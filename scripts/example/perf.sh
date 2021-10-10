## Profiling

pip install py-spy
py-spy top --pid 1
py-spy record --pid 1 --duration 30 --threads --native
py-spy record --pid 1 --duration 30 --threads --native --idle


## Profiles

tensorboard --logdir=./log --bind_all
