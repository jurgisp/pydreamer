import json
import os
import random
import warnings
import yaml
from pathlib import Path

warnings.filterwarnings("ignore", "Your application has authenticated using end user credentials")

from absl import app, flags
from xmanager import xm, xm_local


CWD = Path(__file__).parent.parent
EXPORT_VARS = ['MLFLOW_TRACKING_URI', 'MLFLOW_TRACKING_USERNAME', 'MLFLOW_TRACKING_PASSWORD', 'MLFLOW_EXPERIMENT_NAME']
ENV_VARS = {k: os.environ[k] for k in EXPORT_VARS}

# Xlauncher params
flags.DEFINE_string('dockerfile', 'Dockerfile', '')
# Launch from flagsfile
flags.DEFINE_string('flagsfile', '', 'JSON flags file')
# Launch with explicit PyDreamer parameters
flags.DEFINE_string('configs', '', '')
flags.DEFINE_string('env_id', '', '')
flags.DEFINE_string('run_name', '', '')
flags.DEFINE_string('resume_id', '', '')
# Launch multiple configs
flags.DEFINE_string('configlist', '', 'Comma-separated list of config to launch, appended to --configs')
# Launch from configfile
flags.DEFINE_string('configfile', '', 'YAML config file, from which to launch all configs (only read headers, not actual config)')

FLAGS = flags.FLAGS


def main(_):
    dockerpath = CWD
    dockerfile = CWD / Path(FLAGS.dockerfile)

    # Parse flags list

    if FLAGS.flagsfile:
        flagsfile = CWD / Path(FLAGS.flagsfile)
        expid = flagsfile.stem
        with flagsfile.open('r') as f:
            flagslist = json.load(f)
        print(f'Launching {expid} ({len(flagslist)} runs) from {flagsfile}')

    else:
        assert FLAGS.configs, "--configs must be set"
        assert not (bool(FLAGS.configlist) and bool(FLAGS.configfile)), "Maximum one of --configfile, --configlist should be set"

        if FLAGS.configlist:
            configlist = FLAGS.configlist.split(',')
            expid = FLAGS.run_name or FLAGS.configlist.replace(',', '_')

        elif FLAGS.configfile:
            with open(FLAGS.configfile, 'r') as f:
                conf_yaml = yaml.safe_load(f)
            # Only reads config list, not actual configs
            configlist = [k for k in conf_yaml if not k.startswith('.')]
            expid = Path(FLAGS.configfile).stem

        else:
            configlist = ['']
            expid = FLAGS.run_name or FLAGS.configs.split(',')[-1]

        print('configlist:', configlist)

        flagslist = []
        for config in configlist:
            flagslist.append({
                'configs': f'{FLAGS.configs},{config}' if config else FLAGS.configs,
                'env_id': FLAGS.env_id,
                'run_name': FLAGS.run_name or config or expid,
                'resume_id': FLAGS.resume_id or '${rnd}',
            })

    # Launch experiment

    with xm_local.create_experiment(expid) as exp:
        [executable] = exp.package([
            xm.Packageable(
                executable_spec=xm.Dockerfile(str(dockerpath), str(dockerfile)),
                executor_spec=xm_local.Caip.Spec(),
            ),
        ])

        for flags_conf in flagslist:
            context = {
                'expid': expid,
                'rnd': str(random.randint(10000, 99999))
            }
            flags = {k: replace(v, context) for k, v in flags_conf.items()}
            flags = {k: v for k, v in flags.items() if v}  # Remove empty parameters
            print(f'Launching: {flags}')

            # Launch run

            exp.add(xm.Job(
                executable=executable,
                executor=xm_local.Caip(xm.JobRequirements(a100=1)),  # A100
                # executor=xm_local.Caip(xm.JobRequirements(p100=1, cpu=16 * xm.vCPU, ram=10 * xm.GiB)),  # P100
                # executor=xm_local.Caip(xm.JobRequirements(v100=2, cpu=16 * xm.vCPU, ram=10 * xm.GiB)),  # V100
                # executor=xm_local.Caip(xm.JobRequirements(t4=1, cpu=16 * xm.vCPU, ram=10 * xm.GiB)),  # T4
                args=flags,  # type: ignore
                env_vars=ENV_VARS
            ))

    print('--- LAUNCHER DONE ---')


def replace(s, vars):
    # Replaces ${expid}, etc with value
    for k, v in vars.items():
        s = s.replace(f'${{{k}}}', v)
    return s


if __name__ == '__main__':
    app.run(main)
