import argparse
import pathlib
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import mlflow

import tools
from data import OfflineDataSequential, OfflineDataRandom
from preprocessing import MinigridPreprocess
from models import VAE, RSSM
from modules import ConvEncoder, ConvDecoderCat, DenseDecoder


def run(conf):

    mlflow.start_run(run_name=conf.run_name)
    mlflow.log_params(vars(conf))
    device = torch.device(conf.device)

    data = (OfflineDataSequential(conf.input_dir) if conf.data_seq else
            OfflineDataRandom(conf.input_dir))
    data_eval = OfflineDataSequential(conf.eval_dir)

    preprocess = MinigridPreprocess(categorical=conf.channels,
                                    image_key=conf.image_key,
                                    map_key=conf.map_key,
                                    device=device)

    encoder = ConvEncoder(in_channels=conf.channels,
                          out_dim=conf.embed_dim,
                          stride=1,
                          kernels=(1, 3, 3, 3))
    decoder = (ConvDecoderCat(in_dim=conf.deter_dim + conf.stoch_dim,
                              out_channels=conf.channels,
                              stride=1,
                              kernels=(3, 3, 3, 1))
               if conf.image_decoder == 'cnn' else
               DenseDecoder(in_dim=conf.deter_dim + conf.stoch_dim,
                            out_shape=(conf.channels, 7, 7))
               )
    decoder_map = DenseDecoder(in_dim=conf.deter_dim + conf.stoch_dim,
                               out_shape=(conf.channels, conf.map_size, conf.map_size))
    model = RSSM(
        encoder=encoder,
        decoder_image=decoder,
        decoder_map=decoder_map,
        deter_dim=conf.deter_dim,
        stoch_dim=conf.stoch_dim,
        hidden_dim=conf.hidden_dim,
    )

    model.to(device)
    print(f'Model: {sum(p.numel() for p in model.parameters() if p.requires_grad)} parameters')
    mlflow.set_tag(mlflow.utils.mlflow_tags.MLFLOW_RUN_NOTE, f'```\n{model}\n```')

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, eps=1e-5)

    metrics = defaultdict(list)
    batches = 0
    grad_norm = None
    data_eval_iter = data_eval.iterate(500, 100)
    # data_eval_iter = data_eval.iterate(conf.batch_length, 1000)
    
    for batch in data.iterate(conf.batch_length, conf.batch_size):

        image, action, reset, map = preprocess(batch)

        # Predict

        state = model.init_state(image.size(1))
        output = model(image, action, reset, state)
        loss, loss_metrics, loss_tensors = model.loss(*output, image, map)

        # Grad step

        optimizer.zero_grad()
        loss.backward()
        if conf.grad_clip:
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), conf.grad_clip)
        optimizer.step()

        # Metrics

        batches += 1
        metrics['loss'].append(loss.item())
        for k, v in loss_metrics.items():
            metrics[k].append(v.item())
        if grad_norm:
            metrics['grad_norm'].append(grad_norm.item())

        # Log metrics

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

        # Log artifacts

        def log_batch_npz(batch, loss_tensors, image_pred, image_rec, map_rec, top=-1, subdir='d2_wm_predict'):
            data = batch.copy()
            data.update({k: v.cpu().numpy() for k, v in loss_tensors.items()})
            data['image_pred'] = image_pred.sample().cpu().numpy()
            data['image_rec'] = image_rec.sample().cpu().numpy()
            data['map_rec'] = map_rec.sample().cpu().numpy()
            data['image_pred_p'] = image_pred.probs.cpu().numpy()
            data['image_rec_p'] = image_rec.probs.cpu().numpy()
            data['map_rec_p'] = map_rec.probs.cpu().numpy()
            data = {k: v.swapaxes(0, 1)[:top] for k, v in data.items()}  # (N,B,...) => (B,N,...)
            tools.mlflow_log_npz(data, f'{batches:07}.npz', subdir)

        if batches % conf.log_interval == 0:
            with torch.no_grad():
                image_pred, image_rec, map_rec = model.predict_obs(*output)
            log_batch_npz(batch, loss_tensors, image_pred, image_rec, map_rec)
            

        # Save model

        if conf.save_path and batches % conf.save_interval == 0:
            pathlib.Path(conf.save_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), conf.save_path)
            print(f'Saved to {conf.save_path}')

        # Stop

        if batches >= conf.n_steps:
            print('Stopping')
            break

        # Evaluate

        if batches % conf.eval_interval == 0:
            batch = next(data_eval_iter)
            image, action, reset, map = preprocess(batch)
            print(f'Eval batch: {image.shape}')

            for _ in range(1):
                with torch.no_grad():
                    output = model(image, action, reset, model.init_state(image.size(1)))
                    loss, loss_metrics, loss_tensors = model.loss(*output, image, map)
                    image_pred, image_rec, map_rec = model.predict_obs(*output)

            metrics_eval = {f'eval/{k}': v.item() for k, v in loss_metrics.items()}
            mlflow.log_metrics(metrics_eval, step=batches)
            log_batch_npz(batch, loss_tensors, image_pred, image_rec, map_rec, top=10, subdir='d2_wm_predict_eval')


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
