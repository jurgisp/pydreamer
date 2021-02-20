import argparse
import pathlib
from collections import defaultdict
import numpy as np
import torch
import mlflow

import tools
from data import OfflineData
from preprocessing import MinigridPreprocess
from models import VAE, RSSM
from modules import ConvEncoder, ConvDecoderCat


def run(conf):

    mlflow.start_run(run_name=conf.run_name)
    mlflow.log_params(vars(conf))

    data = OfflineData(conf.input_dir)

    device = torch.device(conf.device)
    preprocess = MinigridPreprocess(categorical=33, device=device)

    # model = VAE(
    #     encoder=ConvEncoder(in_channels=preprocess.img_channels, out_dim=conf.embed_dim, stride=1, kernels=(1, 3, 3, 3)),
    #     decoder=ConvDecoderCat(in_dim=conf.stoch_dim, out_channels=preprocess.img_channels, stride=1, kernels=(3, 3, 3, 1))
    # )
    model = RSSM(
        encoder=ConvEncoder(in_channels=preprocess.img_channels, out_dim=conf.embed_dim, stride=1, kernels=(1, 3, 3, 3)),
        decoder=ConvDecoderCat(in_dim=conf.stoch_dim, out_channels=preprocess.img_channels, stride=1, kernels=(3, 3, 3, 1)),
        deter_dim=conf.deter_dim,
        stoch_dim=conf.stoch_dim,
        hidden_dim=conf.hidden_dim,
    )
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    print(f'Model: {sum(p.numel() for p in model.parameters() if p.requires_grad)} parameters')
    mlflow.set_tag(mlflow.utils.mlflow_tags.MLFLOW_RUN_NOTE, f'```\n{model}\n```')
    metrics = defaultdict(list)
    batches = 0
    for batch in data.iterate(conf.batch_length, conf.batch_size):

        image, action, reset = preprocess(batch)

        # Predict

        state = model.init_state(image.size(1))
        output = model(image, action, reset, state)
        loss, loss_metrics = model.loss(*output, image)

        # Grad step

        optimizer.zero_grad()
        loss.backward()
        # nn.utils.clip_grad_norm_(self._dreamer.model_params(), self._grad_clip)
        optimizer.step()

        # Metrics

        batches += 1
        metrics['loss'].append(loss.item())
        for k, v in loss_metrics.items():
            metrics[k].append(v.item())

        # Log

        if batches % conf.log_interval == 0:
            metrics = {k: np.mean(v) for k, v in metrics.items()}
            metrics['_step'] = batches
            metrics['_loss'] = metrics['loss']

            print(f"[{batches:06}]"
                  f"  loss: {metrics['loss']:.3f}"
                  f"  loss_kl: {metrics['loss_kl']:.3f}"
                  f"  loss_image: {metrics['loss_image']:.3f}"
                  )
            mlflow.log_metrics(metrics, step=batches)
            metrics = defaultdict(list)

        # Save

        if conf.save_path and batches % conf.save_interval == 0:
            pathlib.Path(conf.save_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), conf.save_path)
            print(f'Saved to {conf.save_path}')

        # Stop

        if batches >= conf.n_steps:
            print('Stopping')
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--configs', nargs='+', required=True)
    args, remaining = parser.parse_known_args()

    # Config from YAML
    conf = {}
    configs = tools.read_yamls('./config')    
    for name in args.configs:
        conf.update(configs[name])

    # Override config from command-line
    parser = argparse.ArgumentParser()
    for key, value in conf.items():
        parser.add_argument(f'--{key}', type=type(value), default=value)
    conf = parser.parse_args(remaining)

    run(conf)
