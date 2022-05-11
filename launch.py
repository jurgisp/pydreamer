import argparse
import os
import time
from logging import info
from distutils.util import strtobool
from multiprocessing import Process
from typing import List

import generator
import train
from pydreamer.tools import (configure_logging, mlflow_log_params,
                             mlflow_init, read_yamls)


def launch():
    configure_logging('[launcher]')
    parser = argparse.ArgumentParser()
    parser.add_argument('--configs', nargs='+', required=True)
    args, remaining = parser.parse_known_args()

    # Config from YAML

    conf = {}
    configs = read_yamls('./config')
    for name in args.configs:
        if ',' in name:
            for n in name.split(','):
                conf.update(configs[n])
        else:
            conf.update(configs[name])

    # Override config from command-line

    parser = argparse.ArgumentParser()
    for key, value in conf.items():
        type_ = type(value) if value is not None else str
        if type_ == bool:
            type_ = lambda x: bool(strtobool(x))
        parser.add_argument(f'--{key}', type=type_, default=value)
    conf = parser.parse_args(remaining)

    # Mlflow

    mlrun = mlflow_init(conf.run_name or conf.resume_id, conf.resume_id)
    artifact_uri = mlrun.info.artifact_uri
    mlflow_log_params(vars(conf))

    # Launch train+eval generators

    subprocesses: List[Process] = []
    for i in range(conf.generator_workers):
        info(f'Launching train+eval generator {i}')
        p = run_generator(
            conf.env_id,
            conf,
            save_uri=f'{artifact_uri}/episodes/{i}',
            save_uri2=f'{artifact_uri}/episodes_eval/{i}',
            num_steps=conf.n_env_steps // conf.env_action_repeat // conf.generator_workers,
            limit_step_ratio=conf.limit_step_ratio / conf.generator_workers,
            worker_id=i,
            policy_main='network',
            policy_prefill=conf.generator_prefill_policy,
            num_steps_prefill=conf.generator_prefill_steps // conf.generator_workers,
            split_fraction=0.05,
        )
        subprocesses.append(p)

    # Launch train generators

    for i in range(conf.generator_workers_train):
        info(f'Launching train generator {i}')
        p = run_generator(
            conf.env_id,
            conf,
            f'{artifact_uri}/episodes/{i}',
            num_steps=conf.n_env_steps // conf.env_action_repeat // conf.generator_workers,
            limit_step_ratio=conf.limit_step_ratio / conf.generator_workers,
            worker_id=i,
            policy_main='network',
            policy_prefill=conf.generator_prefill_policy,
            num_steps_prefill=conf.generator_prefill_steps // conf.generator_workers,
        )
        subprocesses.append(p)

    # Launch eval generators

    for i in range(conf.generator_workers_eval):
        info(f'Launching eval generator {i}')
        p = run_generator(
            conf.env_id_eval or conf.env_id,
            conf,
            f'{artifact_uri}/episodes_eval/{i}',
            worker_id=conf.generator_workers + i,
            policy_main='network',
            metrics_prefix='agent_eval')
        subprocesses.append(p)

    # Launch trainer

    info('Launching trainer')
    p = run_trainer(conf)
    subprocesses.append(p)

    # Wait & watch

    while len(subprocesses) > 0:
        check_subprocesses(subprocesses)
        time.sleep(1)


def run_trainer(conf):
    p = Process(target=train.run, daemon=True, args=[conf])
    p.start()
    return p


def run_generator(env_id,
                  conf,
                  save_uri,
                  save_uri2=None,
                  policy_main='network',
                  policy_prefill='random',
                  worker_id=0,
                  num_steps=int(1e9),
                  num_steps_prefill=0,
                  limit_step_ratio=0,
                  split_fraction=0.0,
                  metrics_prefix='agent',
                  log_mlflow_metrics=True,
                  ):
    p = Process(target=generator.main,
                daemon=True,
                kwargs=dict(
                    env_id=env_id,
                    save_uri=save_uri,
                    save_uri2=save_uri2,
                    env_time_limit=conf.env_time_limit,
                    env_action_repeat=conf.env_action_repeat,
                    env_no_terminal=conf.env_no_terminal,
                    limit_step_ratio=limit_step_ratio,
                    policy_main=policy_main,
                    policy_prefill=policy_prefill,
                    num_steps=num_steps,
                    num_steps_prefill=num_steps_prefill,
                    worker_id=worker_id,
                    model_conf=conf,
                    log_mlflow_metrics=log_mlflow_metrics,
                    split_fraction=split_fraction,
                    metrics_prefix=metrics_prefix,
                    metrics_gamma=conf.gamma,
                ))
    p.start()
    return p


def check_subprocesses(subprocesses):
    subp_finished = []
    for p in subprocesses:
        if not p.is_alive():
            if p.exitcode == 0:
                subp_finished.append(p)
                info(f'Generator process {p.pid} finished')
            else:
                raise Exception(f'Generator process {p.pid} died with exitcode {p.exitcode}')
    for p in subp_finished:
        subprocesses.remove(p)


if __name__ == '__main__':
    launch()
