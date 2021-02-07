import argparse
import inspect
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
from modules import MinigridEncoder, MinigridDecoder


def main(run_name='adhoc',
         input_dir='./data',
         batch_length=10,
         batch_size=25,
         log_interval=40,
         save_path='./log/model.pt',
         save_interval=400,
         ):

    mlflow.start_run(run_name=run_name)

    data = OfflineData(input_dir)

    preprocess = MinigridPreprocess()

    model = VAE(
        encoder=MinigridEncoder(),
        decoder=MinigridDecoder())
    print(f'Model: {sum(p.numel() for p in model.parameters() if p.requires_grad)} parameters')

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    metrics = defaultdict(list)
    frames = 0
    batches = 0
    for batch in data.iterate(batch_length, batch_size):

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

        frames += image.size(0) * image.size(1)
        batches += 1
        metrics['loss'].append(loss.item())
        for k, v in loss_metrics.items():
            metrics[k].append(v.item())

        # Log

        if batches % log_interval == 0:
            metrics = {k: np.mean(v) for k, v in metrics.items()}
            metrics['_step'] = frames
            metrics['_loss'] = metrics['loss']
            metrics['batch'] = batches

            print(f"[{frames:07}]"
                  f"  loss: {metrics['loss']:.3f}")
            mlflow.log_metrics(metrics, step=frames)
            metrics = defaultdict(list)

        # Save

        if batches % save_interval == 0:
            pathlib.Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), save_path)
            print(f'Saved to {save_path}')


if __name__ == '__main__':
    # Use main() kwargs as config
    parser = argparse.ArgumentParser()
    argspec = inspect.getfullargspec(main)
    for key, value in zip(argspec.args, argspec.defaults):
        parser.add_argument(f'--{key}', type=type(value), default=value)
    config = parser.parse_args()
    main(**vars(config))
