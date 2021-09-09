# Download data

mkdir -p ./data/train
mkdir -p ./data/eval
gsutil -m cp gs://humanui-mlflow-west4/artifacts/29/5e20b74069d74cfabe8bf3b7e7c177ff/artifacts/episodes/* ./data/train/
gsutil -m cp gs://humanui-mlflow-west4/artifacts/29/4225b9398bc14a4fb484af5c9455d1b7/artifacts/episodes/* ./data/eval/

# Run 

python train.py --configs defaults miniworld --input_dir ./data/train --eval_dir ./data/eval --data_workers 2

# Profiles

tensorboard --logdir=./log --bind_all
