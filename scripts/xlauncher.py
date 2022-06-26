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
EXPORT_VARS = [
    'MLFLOW_TRACKING_URI',
    'MLFLOW_TRACKING_USERNAME',
    'MLFLOW_TRACKING_PASSWORD',
    'MLFLOW_EXPERIMENT_NAME',
]
ENV_VARS = {k: os.environ[k] for k in EXPORT_VARS}
EXPID_PREFIX = os.environ.get('EXPID_PREFIX', '')

# Xlauncher params
flags.DEFINE_string('dockerfile', 'Dockerfile', '')
flags.DEFINE_string('dockerfile_actor', '', '')
flags.DEFINE_integer('a100', 0, '')
flags.DEFINE_integer('v100', 0, '')
flags.DEFINE_integer('p100', 0, '')
flags.DEFINE_integer('num_actors', 0, 'How many actors to launch')
flags.DEFINE_integer('cpu_actors', 4, 'Number of CPUs per actor')
flags.DEFINE_integer('seeds', 1, 'How many times to launch each run')
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
flags.DEFINE_string('run_name_suffix', '', '')
flags.DEFINE_bool('fixed_resume', False, 'Sets resume_id automatically based on run name, so it can be rerun and resumed.')

FLAGS = flags.FLAGS


def main(_):
    run_name_suffix = f'_{FLAGS.run_name_suffix}' if FLAGS.run_name_suffix else ''

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
            expid = FLAGS.run_name or make_name(configlist)

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
            for i in range(FLAGS.seeds):
                flagslist.append({
                    'configs': f'{FLAGS.configs},{config}' if config else FLAGS.configs,
                    'env_id': FLAGS.env_id,
                    'MLFLOW_RUN_NAME': FLAGS.run_name or f'{config}{run_name_suffix}',
                    'MLFLOW_RESUME_ID': FLAGS.resume_id or (f'{config}{run_name_suffix}_{i}' if FLAGS.fixed_resume else random_string()),
                })

    # Launch experiment

    with xm_local.create_experiment(EXPID_PREFIX + expid + run_name_suffix) as exp:
        [executable] = exp.package([
            xm.Packageable(
                executable_spec=xm.Dockerfile(str(CWD), str(CWD / FLAGS.dockerfile)),
                executor_spec=xm_local.Caip.Spec(),
            ),
        ])
        if FLAGS.dockerfile_actor:
            [exec_actor] = exp.package([
                xm.Packageable(
                    executable_spec=xm.Dockerfile(str(CWD), str(CWD / FLAGS.dockerfile_actor)),
                    executor_spec=xm_local.Caip.Spec(),
                ),
            ])
        else:
            exec_actor = executable

        for flags_conf in flagslist:
            flags = {k: v for k, v in flags_conf.items() if v}  # Remove empty parameters
            print(f'Launching: {flags}')
            
            # ALL_CAPS flags => env variables
            env_flags = {k: v for k, v in flags.items() if k.upper() == k}
            flags = {k: v for k, v in flags.items() if k not in env_flags}

            # Launch run

            job = xm.Job(
                executable=executable,
                executor=xm_local.Caip(
                    xm.JobRequirements(v100=FLAGS.v100, cpu=8 * FLAGS.v100, ram=10 * xm.GiB) if FLAGS.v100  # 1xV100 => n1-standard-8
                    else xm.JobRequirements(p100=FLAGS.p100, cpu=16 * FLAGS.p100, ram=10 * xm.GiB) if FLAGS.p100  # 1xP100 => n1-standard-16
                    else xm.JobRequirements(a100=FLAGS.a100 or 1),  # 1xA100 => a2-highgpu-1g (12CPU)
                ),
                args=flags,  # type: ignore
                env_vars=dict(**ENV_VARS, **env_flags),
            )

            if FLAGS.num_actors:
                # Multi-worker job
                job = xm.JobGroup(
                    learner=job,
                    actor=xm.Job(
                        executable=exec_actor,
                        executor=xm_local.Caip(xm.JobRequirements(cpu=FLAGS.cpu_actors * xm.vCPU, ram=8 * xm.GiB, replicas=FLAGS.num_actors)),
                        args=flags,  # type: ignore
                        env_vars=dict(**ENV_VARS, **env_flags),
                    ),
                )

            exp.add(job)

    print('--- LAUNCHER DONE ---')


def random_string(length=5):
    import string
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=length))


def make_name(flagslist):
    # Builds experiment name from a list of configs

    def longest_prefix(strings):
        first = strings[0]
        i = 0
        while i < len(first):
            prefix = first[:i + 1]
            if any(s[:i + 1] != prefix for s in strings):
                break
            i += 1
        return first[:i]

    def longest_suffix(strings):
        first = strings[0]
        i = 0
        while i < len(first):
            suffix = first[-i - 1:]
            if any(s[-i - 1:] != suffix for s in strings):
                break
            i += 1
        if i == 0:
            return ''
        return first[-i:]

    if len(flagslist) == 1:
        return flagslist[0]

    prefix = longest_prefix(flagslist)
    if prefix:
        flagslist = [s[len(prefix):] for s in flagslist]
    suffix = longest_suffix(flagslist)
    if suffix:
        flagslist = [s[:-len(suffix)] for s in flagslist]
    name = prefix + '(' + '|'.join(flagslist) + ')' + suffix
    return name[:100]


if __name__ == '__main__':
    app.run(main)
