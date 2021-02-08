import argparse
import pathlib
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import mlflow

from data import OfflineData
from preprocessing import MinigridPreprocess
from models import VAE
from modules import MinigridEncoder, MinigridDecoderCE


DEFAULT_CONFIG = dict(
    run_name='adhoc',
    input_dir='./data',
    log_interval=40,
    save_path='./log/model.pt',
    save_interval=400,
    # Training
    n_steps=2_000_000,
    batch_length=10,
    batch_size=25,
    # Model
    stoch_dim=10
)


def run(conf):

    mlflow.start_run(run_name=conf.run_name)
    mlflow.log_params(vars(conf))

    data = OfflineData(conf.input_dir)

    preprocess = MinigridPreprocess(categorical=True)

    model = VAE(
        encoder=MinigridEncoder(in_channels=preprocess.img_channels),
        decoder=MinigridDecoderCE(in_dim=conf.stoch_dim)
    )
    print(f'Model: {sum(p.numel() for p in model.parameters() if p.requires_grad)} parameters')
    mlflow.set_tag(mlflow.utils.mlflow_tags.MLFLOW_RUN_NOTE, f'```\n{model}\n```')

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    metrics = defaultdict(list)
    step = 0
    batches = 0
    for batch in data.iterate(conf.batch_length, conf.batch_size):

        # Predict

        image = preprocess(batch)
        output = model(image, None)
        loss, loss_metrics = model.loss(*output, image)

        # Grad step

        optimizer.zero_grad()
        loss.backward()
        # nn.utils.clip_grad_norm_(self._dreamer.model_params(), self._grad_clip)
        optimizer.step()

        # Metrics

        step += image.size(0) * image.size(1)
        batches += 1
        metrics['loss'].append(loss.item())
        for k, v in loss_metrics.items():
            metrics[k].append(v.item())

        # Log

        if batches % conf.log_interval == 0:
            metrics = {k: np.mean(v) for k, v in metrics.items()}
            metrics['_step'] = step
            metrics['_loss'] = metrics['loss']
            metrics['batch'] = batches

            print(f"[{step:07}]"
                  f"  loss: {metrics['loss']:.3f}"
                  f"  loss_kl: {metrics['loss_kl']:.3f}"
                  f"  loss_obs: {metrics['loss_obs']:.3f}"
                  )
            mlflow.log_metrics(metrics, step=step)
            metrics = defaultdict(list)

        # Save

        if conf.save_path and batches % conf.save_interval == 0:
            pathlib.Path(conf.save_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), conf.save_path)
            print(f'Saved to {conf.save_path}')

        # Stop

        if step >= conf.n_steps:
            print('Stopping')
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for key, value in DEFAULT_CONFIG.items():
        parser.add_argument(f'--{key}', type=type(value), default=value)
    conf = parser.parse_args()
    run(conf)
