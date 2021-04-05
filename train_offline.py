import argparse
import pathlib
import subprocess
from collections import defaultdict
import numpy as np
import time
import torch
import torch.nn as nn
import torch.distributions as D
import mlflow

import tools
from tools import mlflow_start_or_resume
from data import OfflineDataSequential, OfflineDataRandom
from preprocessing import MinigridPreprocess
from models import *
from modules import *


def run(conf):
    assert not(conf.keep_state and not conf.data_seq), "Should train sequentially if keeping state"

    if conf.generator_run:
        # Start train
        cmd = f'python3 generator.py {conf.generator_env} --num_steps 1000000000 --seed 1 --output_dir {conf.input_dir} --delete_old {conf.generator_buffer}'
        print(f'Starting data generator:\n{cmd}')
        p1 = subprocess.Popen(cmd.split(' '), stdout=subprocess.DEVNULL)
        # Start eval
        cmd = f'python3 generator.py {conf.generator_env} --num_steps 1000000000 --seed 2 --output_dir {conf.eval_dir} --delete_old {conf.generator_buffer} --sleep 20'
        print(f'Starting data generator:\n{cmd}')
        p2 = subprocess.Popen(cmd.split(' '), stdout=subprocess.DEVNULL)
        # Check
        time.sleep(5)
        assert (p1.poll() is None) and (p2.poll() is None), 'Process has exited'
        # Wait
        print(f'Waiting for {conf.generator_wait} sec for initial data')
        time.sleep(conf.generator_wait)
        # Check again
        assert (p1.poll() is None) and (p2.poll() is None), 'Process has exited'

    mlflow_start_or_resume(conf.run_name, conf.resume_id)
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
                               out_shape=(conf.channels, conf.map_size, conf.map_size),
                               hidden_layers=4)
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

    resume_step = tools.mlflow_load_checkpoint(model, optimizer)
    if resume_step:
        print(f'Loaded model from checkpoint epoch {resume_step}')

    eval_iter = data_eval.iterate(conf.batch_length, 1000)
    eval_iter_full = data_eval.iterate(500, 100)

    start_time = time.time()
    steps = resume_step or 0
    last_time = start_time
    last_steps = steps
    metrics = defaultdict(list)

    persist_state = None

    for batch in data.iterate(conf.batch_length, conf.batch_size):

        image, action, reset, map = preprocess(batch)
        if persist_state is None and conf.keep_state:
            persist_state = model.init_state(image.size(1))

        # Predict

        state = persist_state if conf.keep_state else model.init_state(image.size(1))
        output = model(image, action, reset, state)
        loss, loss_metrics, loss_tensors = model.loss(*output, image, map)
        persist_state = output[-1][-1].detach()

        # Grad step

        optimizer.zero_grad()
        loss.backward()
        if conf.grad_clip:
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), conf.grad_clip)
            metrics['grad_norm'].append(grad_norm.item())
        optimizer.step()

        # Metrics

        steps += 1
        metrics['loss'].append(loss.item())
        for k, v in loss_metrics.items():
            metrics[k].append(v.item())

        # Log metrics

        if steps % conf.log_interval == 0:
            metrics = {k: np.mean(v) for k, v in metrics.items()}
            metrics['_step'] = steps
            metrics['_loss'] = metrics['loss']

            t = time.time()
            fps = (steps - last_steps) / (t - last_time)
            metrics['fps'] = fps
            last_time, last_steps = t, steps

            print(f"T:{t-start_time:05.0f}  "
                  f"[{steps:06}]"
                  f"  loss: {metrics['loss']:.3f}"
                  f"  loss_kl: {metrics['loss_kl']:.3f}"
                  f"  loss_image: {metrics['loss_image']:.3f}"
                  f"  fps: {metrics['fps']:.3f}"
                  )
            mlflow.log_metrics(metrics, step=steps)
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
            tools.mlflow_log_npz(data, f'{steps:07}.npz', subdir)

        if steps % conf.log_interval == 0:
            with torch.no_grad():
                image_pred, image_rec, map_rec = model.predict_obs(*output)
            log_batch_npz(batch, loss_tensors, image_pred, image_rec, map_rec)

        # Save model

        if steps % conf.save_interval == 0:
            tools.mlflow_save_checkpoint(model, optimizer, steps)
            print(f'Saved model checkpoint')

        # Stop

        if steps >= conf.n_steps:
            print('Stopping')
            break

        # Evaluate single batch (does not make sense if keeping state)

        # if steps % conf.eval_interval == 0 and not conf.keep_state:
        #     batch = next(eval_iter)
        #     image, action, reset, map = preprocess(batch)
        #     print(f'Eval batch: {image.shape}')

        #     with torch.no_grad():
        #         output = model(image, action, reset, model.init_state(image.size(1)))
        #         loss, loss_metrics, loss_tensors = model.loss(*output, image, map)
        #         image_pred, image_rec, map_rec = model.predict_obs(*output)

        #     metrics_eval = {f'eval/{k}': v.item() for k, v in loss_metrics.items()}
        #     mlflow.log_metrics(metrics_eval, step=steps)
        #     # log_batch_npz(batch, loss_tensors, image_pred, image_rec, map_rec, top=10, subdir='d2_wm_predict_eval')

        # Evaluate full

        if steps % conf.eval_interval == 0:
            batch = next(eval_iter_full)
            image, action, reset, map = preprocess(batch)
            image_cat = image.argmax(axis=-3)
            print(f'Eval full {image.shape} for {conf.full_eval_samples} samples')

            map_losses = []
            img_losses = []
            metrics_eval = defaultdict(list)
            image_pred_sum = None
            image_rec_sum = None
            map_rec_sum = None

            # Sample loss several times and do log E[p(map|state)] = log avg[exp(loss)]
            with tools.Timer('eval_sampling'):
                for _ in range(conf.full_eval_samples):
                    with torch.no_grad():
                        output = model(image, action, reset, model.init_state(image.size(1)))
                        loss, loss_metrics, loss_tensors = model.loss(*output, image, map)
                        image_pred, image_rec, map_rec = model.predict_obs(*output)

                    map_losses.append(map_rec.log_prob(map).sum(dim=[-1, -2]))          # Keep (N,B) dim
                    img_losses.append(image_pred.log_prob(image_cat).sum(dim=[-1, -2]))

                    for k, v in loss_metrics.items():
                        metrics_eval[k].append(v.item())

                    if image_pred_sum is None:
                        image_pred_sum = image_pred.probs
                        image_rec_sum = image_rec.probs
                        map_rec_sum = map_rec.probs
                    else:
                        image_pred_sum += image_pred.probs
                        image_rec_sum += image_rec.probs
                        map_rec_sum += map_rec.probs
                    # TODO: loss_tensors should be aggregated too

            metrics_eval = {f'eval_full/{k}': np.mean(v) for k, v in metrics_eval.items()}

            map_losses = torch.stack(map_losses)  # (S,N,B)
            img_losses = torch.stack(img_losses)
            map_losses_exp = torch.logsumexp(map_losses, dim=0) - np.log(conf.full_eval_samples)  # log avg[exp(loss)] = log sum[exp(loss)] - log(S)
            img_losses_exp = torch.logsumexp(img_losses, dim=0) - np.log(conf.full_eval_samples)  # log avg[exp(loss)] = log sum[exp(loss)] - log(S)
            metrics_eval['eval_full/loss_map_exp'] = -map_losses_exp.mean().item()
            metrics_eval['eval_full/loss_image_pred_exp'] = -img_losses_exp.mean().item()

            image_pred = D.Categorical(probs=image_pred_sum / conf.full_eval_samples)  # Average image predictions over samples
            image_rec = D.Categorical(probs=image_rec_sum / conf.full_eval_samples)
            map_rec = D.Categorical(probs=map_rec_sum / conf.full_eval_samples)

            mlflow.log_metrics(metrics_eval, step=steps)
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
        parser.add_argument(f'--{key}', type=type(value) if value is not None else str, default=value)
    conf = parser.parse_args(remaining)

    run(conf)
