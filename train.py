import os
from datetime import datetime
from typing import Optional
import argparse
from pathlib import Path
from multiprocessing import Process
from collections import defaultdict
from typing import Iterator
from itertools import chain
import numpy as np
import time
import torch
from torch import tensor, Tensor
import torch.nn as nn
import torch.distributions as D
from torch.profiler import ProfilerActivity
import mlflow
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import scipy.special

import tools
from tools import *
from data import OfflineDataSequential
from preprocessing import MinigridPreprocess, WorkerInfoPreprocess
from models import *
import generator


torch.distributions.Distribution.set_default_validate_args(False)
torch.backends.cudnn.benchmark = True  # type: ignore


def run(conf):
    mlflow_start_or_resume(conf.run_name, conf.resume_id)
    mlflow.log_params(vars(conf))
    device = torch.device(conf.device)

    # Generator / Agent

    data_reload_interval = 0
    if conf.generator_run:
        conf.input_dir = mlflow.active_run().info.artifact_uri.replace('file://', '') + '/episodes'  # type: ignore
        conf.eval_dir = mlflow.active_run().info.artifact_uri.replace('file://', '') + '/episodes_eval'  # type: ignore
        data_reload_interval = 60
        print(f'Generator prefilling random data ({conf.generator_prefill_steps} steps)...')
        run_generator(conf, seed=0, policy='random', num_steps=conf.generator_prefill_steps, block=True)
        print('Generator random prefill done, starting agent generator...')
        run_generator(conf, seed=1, policy='network', episodes_dir='episodes', metrics_prefix='agent')
        run_generator(conf, seed=2, policy='network', episodes_dir='episodes_eval', metrics_prefix='agent_eval')

    # Data

    data = OfflineDataSequential(conf.input_dir, conf.batch_length, conf.batch_size, skip_first=True, reload_interval=data_reload_interval)
    preprocess = MinigridPreprocess(image_categorical=conf.image_channels if conf.image_categorical else None,
                                    image_key=conf.image_key,
                                    map_categorical=conf.map_channels if conf.map_categorical else None,
                                    map_key=conf.map_key,
                                    amp=conf.device.startswith('cuda') and conf.amp)

    # MODEL

    model = Dreamer(conf)
    model.to(device)

    # elif conf.model == 'map_rnn':
    #     model = MapPredictModel(
    #         encoder=encoder,
    #         decoder=decoder,
    #         map_model=map_model,
    #         action_dim=conf.action_dim,
    #         state_dim=state_dim,
    #     )  # type: ignore

    print(f'Model: {param_count(model)} parameters')
    for submodel in [model.wm._encoder, model.wm._decoder_image, model.wm._core, model.wm._input_rnn, model.map_model]:
        if submodel is not None:
            print(f'  {type(submodel).__name__:<15}: {param_count(submodel)} parameters')
    # print(model)
    mlflow_log_text(str(model), 'architecture.txt')

    # Training

    optimizer_wm = torch.optim.AdamW(model.wm.parameters(), lr=conf.adam_lr, eps=conf.adam_eps)  # type: ignore
    optimizer_map = torch.optim.AdamW(model.map_model.parameters(), lr=conf.adam_lr, eps=conf.adam_eps)  # type: ignore
    optimizer_actor = torch.optim.AdamW(model.ac._actor.parameters(), lr=conf.adam_lr_actor, eps=conf.adam_eps)  # type: ignore
    optimizer_critic = torch.optim.AdamW(model.ac._critic.parameters(), lr=conf.adam_lr_critic, eps=conf.adam_eps)  # type: ignore

    resume_step = tools.mlflow_load_checkpoint(model, optimizer_wm, optimizer_map, optimizer_actor, optimizer_critic)
    if resume_step:
        print(f'Loaded model from checkpoint epoch {resume_step}')

    start_time = time.time()
    steps = resume_step or 0
    last_time = start_time
    last_steps = steps
    metrics = defaultdict(list)
    metrics_max = defaultdict(list)

    timers = {}

    def timer(name):
        if name not in timers:
            timers[name] = Timer('total', False)
        return timers[name]

    states = {}  # by worker
    data_iter = iter(DataLoader(WorkerInfoPreprocess(preprocess(data)),
                                batch_size=None,
                                num_workers=conf.data_workers,
                                prefetch_factor=20 if conf.data_workers else 2,  # GCS download has to be shorter than this many batches (e.g. 1sec < 20*300ms)
                                pin_memory=True))

    scaler = GradScaler(enabled=conf.amp)

    with get_profiler(conf) as profiler:
        while True:
            with timer('total'):
                profiler.step()
                steps += 1

                # Make batch

                with timer('data'):

                    batch, wid = next(data_iter)
                    image = batch['image'].to(device)
                    reward = batch['reward'].to(device)
                    terminal = batch['terminal'].to(device)
                    action = batch['action'].to(device)
                    reset = batch['reset'].to(device)
                    map = batch['map'].to(device)
                    map_coord = batch['map_coord'].to(device)

                # Forward

                with timer('forward'):
                    with autocast(enabled=conf.amp):

                        state = states.get(wid) or model.wm.init_state(image.size(1) * conf.iwae_samples)
                        losses, loss_metrics, loss_tensors, new_state, out_tensors, dream_tensors = \
                            model.train(image, reward, terminal, action, reset, map, map_coord, state,
                                        I=conf.iwae_samples,
                                        H=conf.imag_horizon,
                                        do_output_tensors=steps % conf.logbatch_interval == 1,
                                        do_dream_tensors=steps % conf.logbatch_interval == 1)
                        if conf.keep_state:
                            states[wid] = new_state

                # Backward

                with timer('backward'):

                    loss_wm, loss_map, loss_actor, loss_critic = losses
                    optimizer_wm.zero_grad()
                    optimizer_map.zero_grad()
                    optimizer_actor.zero_grad()
                    optimizer_critic.zero_grad()
                    scaler.scale(loss_wm).backward()
                    scaler.scale(loss_map).backward()
                    scaler.scale(loss_actor).backward()
                    scaler.scale(loss_critic).backward()

                # Grad step

                with timer('gradstep'):  # CUDA wait happens here

                    scaler.unscale_(optimizer_wm)
                    scaler.unscale_(optimizer_map)
                    scaler.unscale_(optimizer_actor)
                    scaler.unscale_(optimizer_critic)
                    grad_norm_model = nn.utils.clip_grad_norm_(model.wm.parameters(), conf.grad_clip)
                    grad_norm_map = nn.utils.clip_grad_norm_(model.map_model.parameters(), conf.grad_clip)
                    grad_norm_actor = nn.utils.clip_grad_norm_(model.ac._actor.parameters(), conf.grad_clip)
                    grad_norm_critic = nn.utils.clip_grad_norm_(model.ac._critic.parameters(), conf.grad_clip)
                    grad_metrics = {
                        'grad_norm': grad_norm_model, 
                        'grad_norm_map': grad_norm_map, 
                        'grad_norm_actor': grad_norm_actor,
                        'grad_norm_critic': grad_norm_critic,
                        }
                    scaler.step(optimizer_wm)
                    scaler.step(optimizer_map)
                    scaler.step(optimizer_actor)
                    scaler.step(optimizer_critic)
                    scaler.update()

                with timer('other'):

                    # Metrics

                    for k, v in loss_metrics.items():
                        if not np.isnan(v.item()):
                            metrics[k].append(v.item())
                    for k, v in grad_metrics.items():
                        if np.isfinite(v.item()):  # It's ok for grad norm to be inf, when using amp
                            metrics[k].append(v.item())
                            metrics_max[k].append(v.item())

                    # Log sample

                    if out_tensors:
                        log_batch_npz(batch, loss_tensors, out_tensors, f'{steps:07}.npz', subdir='d2_wm_closed', verbose=conf.verbose)
                    if dream_tensors:
                        log_batch_npz(batch, loss_tensors, dream_tensors, f'{steps:07}.npz', subdir='d2_wm_dream', verbose=conf.verbose)

                    # Log metrics

                    if steps % conf.log_interval == 0:
                        metrics = {f'train/{k}': np.mean(v) for k, v in metrics.items()}
                        metrics.update({f'train/{k}_max': np.max(v) for k, v in metrics_max.items()})
                        metrics['train/steps'] = steps  # type: ignore
                        metrics['_step'] = steps  # type: ignore
                        metrics['_loss'] = metrics['train/loss']
                        metrics['_timestamp'] = datetime.now().timestamp()  # type: ignore

                        t = time.time()
                        fps = (steps - last_steps) / (t - last_time)
                        metrics['train/fps'] = fps  # type: ignore
                        last_time, last_steps = t, steps

                        print(f"[{steps:06}]"
                              f"  loss_wm: {metrics.get('train/loss_wm', 0):.3f}"
                              f"  loss_wm_kl: {metrics.get('train/loss_wm_kl', 0):.3f}"
                              f"  loss_critic: {metrics.get('train/loss_critic', 0):.3f}"
                              f"  loss_map: {metrics.get('train/loss_map', 0):.3f}"
                              f"  policy_value: {metrics.get('train/policy_value',0):.3f}"
                              f"  policy_entropy: {metrics.get('train/policy_entropy',0):.3f}"
                              f"  fps: {metrics['train/fps']:.3f}"
                              )
                        if steps > conf.log_interval:  # Skip the first batch, because the losses are very high and mess up y axis
                            mlflow.log_metrics(metrics, step=steps)
                        metrics = defaultdict(list)
                        metrics_max = defaultdict(list)

                    # Save model

                    if steps % conf.save_interval == 0:
                        tools.mlflow_save_checkpoint(model, optimizer_wm, optimizer_map, optimizer_actor, optimizer_critic, steps)
                        print(f'Saved model checkpoint {steps}')

                    # Stop

                    if steps >= conf.n_steps:
                        print('Stopping')
                        break

                # Evaluate

                with timer('eval'):
                    if conf.eval_interval and steps % conf.eval_interval == 0:

                        # Same batch as train
                        data_test = OfflineDataSequential(conf.eval_dir, conf.batch_length, conf.test_batch_size, skip_first=False)
                        test_iter = iter(DataLoader(preprocess(data_test), batch_size=None))
                        evaluate('test', steps, model, test_iter, device, conf.test_batches, conf.iwae_samples, conf.keep_state, conf)

                        # Full episodes
                        data_eval = OfflineDataSequential(conf.eval_dir, conf.batch_length, conf.eval_batch_size, skip_first=False)
                        eval_iter = iter(DataLoader(preprocess(data_eval), batch_size=None))
                        evaluate('eval', steps, model, eval_iter, device, conf.eval_batches, conf.eval_samples, True, conf)

            for k, v in timers.items():
                metrics[f'timer_{k}'].append(v.dt_ms)

            if conf.verbose:
                print(f"[{steps:06}] timers"
                      f"  TOTAL: {timer('total').dt_ms:>4}"
                      f"  data: {timer('data').dt_ms:>4}"
                      f"  forward: {timer('forward').dt_ms:>4}"
                      f"  backward: {timer('backward').dt_ms:>4}"
                      f"  gradstep: {timer('gradstep').dt_ms:>4}"
                      f"  eval: {timer('eval').dt_ms:>4}"
                      f"  other: {timer('other').dt_ms:>4}"
                      )


def evaluate(prefix: str,
             steps: int,
             model: Dreamer,
             data_iterator: Iterator,
             device,
             eval_batches: int,
             eval_samples: int,
             keep_state: bool,
             conf):

    start_time = time.time()
    metrics_eval = defaultdict(list)
    state = None
    loss_tensors = None
    npz_datas = []
    n_finished_episodes = np.zeros(1)
    do_output_tensors = True

    for i_batch in range(eval_batches):
        with torch.no_grad():

            batch = next(data_iterator)
            image = batch['image'].to(device)
            reward = batch['reward'].to(device)
            terminal = batch['terminal'].to(device)
            action = batch['action'].to(device)
            reset = batch['reset'].to(device)
            map = batch['map'].to(device)
            map_coord = batch['map_coord'].to(device)
            N, B = image.shape[:2]

            if i_batch == 0:
                print(f'Evaluation ({prefix}): batches: {eval_batches},  size(N,B,I): {tuple(image.shape[0:2])+(eval_samples,)}')

            reset_episodes = reset.any(dim=0)  # (B,)
            n_reset_episodes = reset_episodes.sum().item()
            if i_batch == 0:
                n_finished_episodes = np.zeros(B)
            else:
                n_finished_episodes += reset_episodes.cpu().numpy()

            # Log _last predictions from the last batch of previous episode

            if n_reset_episodes > 0 and loss_tensors is not None and 'loss_map' in loss_tensors:
                logprob_map_last = (loss_tensors['loss_map'].mean(dim=0) * reset_episodes).sum() / reset_episodes.sum()
                metrics_eval['logprob_map_last'].append(logprob_map_last.item())

            # Open loop & unseen logprob

            if n_reset_episodes == 0 and i_batch == 1:  # just one batch
                with autocast(enabled=conf.amp):
                    _, _, loss_tensors_im, _, out_tensors_im, _ = \
                        model.train(image, reward, terminal,  # (image, reward, terminal) will be ignored in forward pass because of imagine=True
                                    action, reset, map, map_coord, state,
                                    I=eval_samples,
                                    H=conf.imag_horizon,
                                    imagine=True,
                                    do_image_pred=True,
                                    do_output_tensors=True)

                    log_batch_npz(batch, loss_tensors_im, out_tensors_im, f'{steps:07}.npz', subdir=f'd2_wm_open_{prefix}', verbose=True)

                    # if 'logprob_img' in loss_tensors_im:
                    #     metrics_eval['logprob_img_1step'].append(loss_tensors_im['logprob_img'][0].mean().item())
                    #     metrics_eval['logprob_img_2step'].append(loss_tensors_im['logprob_img'][1].mean().item())

            # Closed loop & loss

            with autocast(enabled=conf.amp):
                if state is None or not keep_state:
                    state = model.wm.init_state(image.size(1) * eval_samples)

                _, loss_metrics, loss_tensors, state, out_tensors, _ = \
                    model.train(image, reward, terminal, action, reset, map, map_coord, state,
                                I=eval_samples,
                                H=conf.imag_horizon,
                                do_image_pred=True,
                                do_output_tensors=do_output_tensors)

                for k, v in loss_metrics.items():
                    if not np.isnan(v.item()):
                        metrics_eval[k].append(v.item())

            # Log one episode batch

            if out_tensors:
                npz_datas.append(prepare_batch_npz(dict(**batch, **loss_tensors, **out_tensors), take_b=1))
            if n_finished_episodes[0] > 0:
                # log predictions until first episode is finished
                do_output_tensors = False

    metrics_eval = {f'{prefix}/{k}': np.mean(v) for k, v in metrics_eval.items()}
    mlflow.log_metrics(metrics_eval, step=steps)

    npz_data = {k: np.concatenate([d[k] for d in npz_datas], 1) for k in npz_datas[0]}
    print_once(f'Saving batch d2_wm_closed_{prefix}: ', {k: tuple(v.shape) for k, v in npz_data.items()})
    tools.mlflow_log_npz(npz_data, f'{steps:07}.npz', subdir=f'd2_wm_closed_{prefix}', verbose=True)

    print(f'Evaluation ({prefix}): done in {(time.time()-start_time):.0f} sec, recorded {n_finished_episodes.sum()} episodes')


def log_batch_npz(batch: Dict[str, Tensor],
                  loss_tensors: Dict[str, Tensor],
                  out_tensors: Dict[str, Tensor],
                  filename: str,
                  subdir: str,
                  verbose=False):

    data = dict(**batch, **loss_tensors, **out_tensors)
    print_once(f'Saving batch {subdir} (input): ', {k: tuple(v.shape) for k, v in data.items()})
    data = prepare_batch_npz(data)
    print_once(f'Saving batch {subdir} (proc.): ', {k: tuple(v.shape) for k, v in data.items()})
    tools.mlflow_log_npz(data, filename, subdir, verbose=verbose)


def prepare_batch_npz(data: Dict[str, Tensor], take_b=999):

    def unpreprocess(key: str, val: Tensor) -> np.ndarray:
        if take_b < val.shape[1]:
            val = val[:, :take_b]

        x = val.cpu().numpy()  # (N,B,*)
        if x.dtype in [np.float16, np.float64]:
            x = x.astype(np.float32)

        if len(x.shape) == 2:  # Scalar
            pass

        elif len(x.shape) == 3:  # 1D vector
            assert key in ['action', 'action_pred', 'map_coord', 'agent_pos', 'agent_dir'], \
                f'Unexpected 1D tensor: {key}: {x.shape}, {x.dtype}'

        elif len(x.shape) == 4:  # 2D tensor - categorical image
            assert x.dtype == np.int64 and key.startswith('map'), \
                f'Unexpected 2D tensor: {key}: {x.shape}, {x.dtype}'

        elif len(x.shape) == 5:  # 3D tensor - image
            assert x.dtype == np.float32 and (key.startswith('image') or key.startswith('map')), \
                f'Unexpected 3D tensor: {key}: {x.shape}, {x.dtype}'

            if x.shape[-1] == x.shape[-2]:  # (N,B,C,W,W)
                x = x.transpose(0, 1, 3, 4, 2)  # => (N,B,W,W,C)
            assert x.shape[-2] == x.shape[-3], 'Assuming rectangular images, otherwise need to improve logic'

            if x.shape[-1] in [1, 3]:
                # RGB or grayscale
                x = ((x + 0.5) * 255.0).clip(0, 255).astype('uint8')
            elif np.allclose(x.sum(axis=-1), 1.0) and np.allclose(x.max(axis=-1), 1.0):
                # One-hot
                x = x.argmax(axis=-1)
            else:
                # Categorical logits
                assert key in ['map_rec'], f'Unexpected 3D categorical logits: {key}: {x.shape}'
                x = scipy.special.softmax(x, axis=-1)

        x = x.swapaxes(0, 1)  # (N,B,*) => (B,N,*)
        return x

    return {k: unpreprocess(k, v) for k, v in data.items()}


def run_generator(conf, policy='network', seed=0, num_steps=int(1e9), block=False, episodes_dir='episodes', metrics_prefix='agent'):
    os.environ['MLFLOW_RUN_ID'] = mlflow.active_run().info.run_id  # type: ignore
    p = Process(target=generator.main,
                daemon=True,
                kwargs=dict(
                    env_id=conf.env_id,
                    # env_max_steps=conf.env_max_steps,
                    policy=policy,
                    num_steps=num_steps,
                    seed=seed,
                    model_conf=conf,
                    log_mlflow_metrics=(policy == 'network'),  # Don't log for initial random prefill
                    episodes_dir=episodes_dir,
                    metrics_prefix=metrics_prefix
                ))
    p.start()
    if block:
        p.join()


def get_profiler(conf):
    if conf.enable_profiler:
        return torch.profiler.profile(
            # activities=[ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=10, warmup=10, active=1, repeat=3),
            on_trace_ready=torch.profiler.tensorboard_trace_handler('./log'),
        )
    else:
        return NoProfiler()


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
