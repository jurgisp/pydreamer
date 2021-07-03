conda activate pydreamer

## Generate MiniWorld-MazeS5A4

./kubernetes/run_generator_xvfb.sh dreamer2_episodes MiniWorld-MazeS5A4-v0 0

## Generate MiniWorld-MazeS5N

./kubernetes/run_generator_xvfb.sh dreamer2_episodes MiniWorld-MazeS5N-v0 0
./kubernetes/run_generator_xvfb.sh dreamer2_episodes MiniWorld-MazeS5N-v0 1
./kubernetes/run_generator_xvfb.sh dreamer2_episodes MiniWorld-MazeS5N-v0 2
./kubernetes/run_generator_xvfb.sh dreamer2_episodes MiniWorld-MazeS5N-v0 3
./kubernetes/run_generator_xvfb.sh dreamer2_episodes MiniWorld-MazeS5N-v0 4
./kubernetes/run_generator_xvfb.sh dreamer2_episodes MiniWorld-MazeS5N-v0 5
./kubernetes/run_generator_xvfb.sh dreamer2_episodes MiniWorld-MazeS5N-v0 6
./kubernetes/run_generator_xvfb.sh dreamer2_episodes MiniWorld-MazeS5N-v0 7
./kubernetes/run_generator_xvfb.sh dreamer2_episodes MiniWorld-MazeS5N-v0 8
./kubernetes/run_generator_xvfb.sh dreamer2_episodes MiniWorld-MazeS5N-v0 9

# gs://humanui-mlflow-west4/artifacts/29/c7587b1391c8449f85d2b037ff96b356/artifacts/episodes
gsutil cp gs://humanui-mlflow-west4/artifacts/29/c7587b1391c8449f85d2b037ff96b356/artifacts/episodes/* ./


## Generate MiniWorld

./kubernetes/run_generator_xvfb.sh dreamer2_episodes MiniWorld-MazeS5GridN-v0 0
./kubernetes/run_generator_xvfb.sh dreamer2_episodes MiniWorld-MazeS5GridN-v0 1
./kubernetes/run_generator_xvfb.sh dreamer2_episodes MiniWorld-MazeS5GridN-v0 2
./kubernetes/run_generator_xvfb.sh dreamer2_episodes MiniWorld-MazeS5GridN-v0 3
./kubernetes/run_generator_xvfb.sh dreamer2_episodes MiniWorld-MazeS5GridN-v0 4
./kubernetes/run_generator_xvfb.sh dreamer2_episodes MiniWorld-MazeS5GridN-v0 5
./kubernetes/run_generator_xvfb.sh dreamer2_episodes MiniWorld-MazeS5GridN-v0 6
./kubernetes/run_generator_xvfb.sh dreamer2_episodes MiniWorld-MazeS5GridN-v0 7
./kubernetes/run_generator_xvfb.sh dreamer2_episodes MiniWorld-MazeS5GridN-v0 8
./kubernetes/run_generator_xvfb.sh dreamer2_episodes MiniWorld-MazeS5GridN-v0 9

# Test copy local
gsutil ls gs://humanui-mlflow-west4/artifacts/29/5e20b74069d74cfabe8bf3b7e7c177ff/artifacts/episodes | wc -l
gsutil -m cp gs://humanui-mlflow-west4/artifacts/29/5e20b74069d74cfabe8bf3b7e7c177ff/artifacts/episodes/* ./data/MiniWorld-MazeS5GridN-v0/train2

## Copy data to shared disk

gcloud compute instances attach-disk "adhoc-jurgis" --disk "gke-shared-disk"
gcloud compute instances start adhoc-jurgis
gcloud compute ssh adhoc-jurgis
df -h
sudo mount -o discard,defaults /dev/sdb /data
# sudo resize2fs /dev/sdb

mkdir /data/MiniWorld-MazeS5GridN-v0_100M
mkdir /data/MiniWorld-MazeS5GridN-v0_100M/train
mkdir /data/MiniWorld-MazeS5GridN-v0_100M/eval
cat << EOF > script.sh
gsutil -m cp gs://humanui-mlflow-west4/artifacts/29/4225b9398bc14a4fb484af5c9455d1b7/artifacts/episodes/* /data/MiniWorld-MazeS5GridN-v0_100M/train
gsutil -m cp gs://humanui-mlflow-west4/artifacts/29/5faa791c945f4b6689471bc78a2ae6b9/artifacts/episodes/* /data/MiniWorld-MazeS5GridN-v0_100M/train
gsutil -m cp gs://humanui-mlflow-west4/artifacts/29/5cca20dc701d40b8bbcde63a01bc8dbc/artifacts/episodes/* /data/MiniWorld-MazeS5GridN-v0_100M/train
gsutil -m cp gs://humanui-mlflow-west4/artifacts/29/2fad880802c341ef88fccc615be27b76/artifacts/episodes/* /data/MiniWorld-MazeS5GridN-v0_100M/train
gsutil -m cp gs://humanui-mlflow-west4/artifacts/29/d23a95aeafae478895f70dd0768aa05e/artifacts/episodes/* /data/MiniWorld-MazeS5GridN-v0_100M/train
gsutil -m cp gs://humanui-mlflow-west4/artifacts/29/0cf0d1f0b06847339f2ae053fa10f8d6/artifacts/episodes/* /data/MiniWorld-MazeS5GridN-v0_100M/train
gsutil -m cp gs://humanui-mlflow-west4/artifacts/29/6f775b1283b74c2cb635784d0ee7ad57/artifacts/episodes/* /data/MiniWorld-MazeS5GridN-v0_100M/train
gsutil -m cp gs://humanui-mlflow-west4/artifacts/29/a9627040f28c463a8b6e1801009b491a/artifacts/episodes/* /data/MiniWorld-MazeS5GridN-v0_100M/train
gsutil -m cp gs://humanui-mlflow-west4/artifacts/29/5983031215bf4c769eb94c89473512f3/artifacts/episodes/* /data/MiniWorld-MazeS5GridN-v0_100M/train
gsutil -m cp gs://humanui-mlflow-west4/artifacts/29/5e20b74069d74cfabe8bf3b7e7c177ff/artifacts/episodes/* /data/MiniWorld-MazeS5GridN-v0_100M/train
EOF
chmod u+x script.sh
./script.sh & disown

# Move 10k episodes (5M/100M steps) to eval
mv /data/MiniWorld-MazeS5GridN-v0_100M/train/s0-ep00* /data/MiniWorld-MazeS5GridN-v0_100M/eval/

ls /data/MiniWorld-MazeS5GridN-v0_100M/train | wc -l
ls /data/MiniWorld-MazeS5GridN-v0_100M/eval | wc -l

sudo umount /dev/sdb
gcloud compute instances stop adhoc-jurgis
gcloud compute instances detach-disk "adhoc-jurgis" --disk "gke-shared-disk"


## MiniGrid

# python generator.py MiniGrid-MazeS11-v0 --num_steps 1_000_000_000 --seed 1 --delete_old 20 --output_dir ./data_bak/train --sleep 1
# python generator.py MiniGrid-MazeS11-v0 --num_steps 5_000_000 --seed 2

./kubernetes/run_generator.sh dreamer2_episodes MiniGrid-MazeS11N-v0 1
./kubernetes/run_generator.sh dreamer2_episodes MiniGrid-MazeS11N-v0 2
./kubernetes/run_generator.sh dreamer2_episodes MiniGrid-MazeS11N-v0 3
./kubernetes/run_generator.sh dreamer2_episodes MiniGrid-MazeS11N-v0 4
./kubernetes/run_generator.sh dreamer2_episodes MiniGrid-MazeS11N-v0 5
./kubernetes/run_generator.sh dreamer2_episodes MiniGrid-MazeS11N-v0 6
./kubernetes/run_generator.sh dreamer2_episodes MiniGrid-MazeS11N-v0 7
./kubernetes/run_generator.sh dreamer2_episodes MiniGrid-MazeS11N-v0 8
./kubernetes/run_generator.sh dreamer2_episodes MiniGrid-MazeS11N-v0 9
./kubernetes/run_generator.sh dreamer2_episodes MiniGrid-MazeS11N-v0 10

./kubernetes/run_generator.sh dreamer2_episodes MiniGrid-MazeS11N-v0 11
./kubernetes/run_generator.sh dreamer2_episodes MiniGrid-MazeS11N-v0 12
./kubernetes/run_generator.sh dreamer2_episodes MiniGrid-MazeS11N-v0 13
./kubernetes/run_generator.sh dreamer2_episodes MiniGrid-MazeS11N-v0 14
./kubernetes/run_generator.sh dreamer2_episodes MiniGrid-MazeS11N-v0 15
./kubernetes/run_generator.sh dreamer2_episodes MiniGrid-MazeS11N-v0 16
./kubernetes/run_generator.sh dreamer2_episodes MiniGrid-MazeS11N-v0 17
./kubernetes/run_generator.sh dreamer2_episodes MiniGrid-MazeS11N-v0 18
./kubernetes/run_generator.sh dreamer2_episodes MiniGrid-MazeS11N-v0 19
./kubernetes/run_generator.sh dreamer2_episodes MiniGrid-MazeS11N-v0 20


## Move data

gsutil ls gs://humanui-mlflow/artifacts/29/502ab44135f24211ba757966e340ab35/artifacts/episodes | wc -l
gsutil ls gs://humanui-mlflow/artifacts/29/d2f5c3b8cae8450a8502ff492021bf65/artifacts/episodes | wc -l
gsutil ls gs://humanui-mlflow/artifacts/29/0a2563b816b244c0a1057c37b38fd55d/artifacts/episodes | wc -l
gsutil ls gs://humanui-mlflow/artifacts/29/ab178fffc53b4fd5b5a55d3b946da881/artifacts/episodes | wc -l
gsutil ls gs://humanui-mlflow/artifacts/29/c94a229eae9c46ff85bbcba155633d94/artifacts/episodes | wc -l
gsutil ls gs://humanui-mlflow/artifacts/29/e5a2a835983d4f559f890905bda8913c/artifacts/episodes | wc -l
gsutil ls gs://humanui-mlflow/artifacts/29/ba874a1a439445578db4ed1ac0dc9a1b/artifacts/episodes | wc -l
gsutil ls gs://humanui-mlflow/artifacts/29/5697c6251aa04ccca0e9082d70f04773/artifacts/episodes | wc -l
gsutil ls gs://humanui-mlflow/artifacts/29/ef2f79d1cf6947eda02c19af07a4d0f4/artifacts/episodes | wc -l
gsutil ls gs://humanui-mlflow/artifacts/29/cfa3319f56244e948dbe6df5e212a96d/artifacts/episodes | wc -l


gsutil -m cp -r gs://humanui-mlflow/artifacts/29/502ab44135f24211ba757966e340ab35/artifacts/episodes/ gs://humanui-mlflow/artifacts/episodes/MiniGrid-MazeS11N-v0_100M/
gsutil cp -r gs://humanui-mlflow/artifacts/29/d2f5c3b8cae8450a8502ff492021bf65/artifacts/episodes/ gs://humanui-mlflow/artifacts/episodes/MiniGrid-MazeS11N-v0_100M/
gsutil cp -r gs://humanui-mlflow/artifacts/29/0a2563b816b244c0a1057c37b38fd55d/artifacts/episodes/ gs://humanui-mlflow/artifacts/episodes/MiniGrid-MazeS11N-v0_100M/
gsutil cp -r gs://humanui-mlflow/artifacts/29/ab178fffc53b4fd5b5a55d3b946da881/artifacts/episodes/ gs://humanui-mlflow/artifacts/episodes/MiniGrid-MazeS11N-v0_100M/
gsutil cp -r gs://humanui-mlflow/artifacts/29/c94a229eae9c46ff85bbcba155633d94/artifacts/episodes/ gs://humanui-mlflow/artifacts/episodes/MiniGrid-MazeS11N-v0_100M/
gsutil cp -r gs://humanui-mlflow/artifacts/29/e5a2a835983d4f559f890905bda8913c/artifacts/episodes/ gs://humanui-mlflow/artifacts/episodes/MiniGrid-MazeS11N-v0_100M/
gsutil cp -r gs://humanui-mlflow/artifacts/29/ba874a1a439445578db4ed1ac0dc9a1b/artifacts/episodes/ gs://humanui-mlflow/artifacts/episodes/MiniGrid-MazeS11N-v0_100M/
gsutil cp -r gs://humanui-mlflow/artifacts/29/5697c6251aa04ccca0e9082d70f04773/artifacts/episodes/ gs://humanui-mlflow/artifacts/episodes/MiniGrid-MazeS11N-v0_100M/
gsutil cp -r gs://humanui-mlflow/artifacts/29/ef2f79d1cf6947eda02c19af07a4d0f4/artifacts/episodes/ gs://humanui-mlflow/artifacts/episodes/MiniGrid-MazeS11N-v0_100M/
gsutil -m cp -r gs://humanui-mlflow/artifacts/29/cfa3319f56244e948dbe6df5e212a96d/artifacts/episodes/ gs://humanui-mlflow/artifacts/episodes/MiniGrid-MazeS11N-v0_100M/

gsutil ls gs://humanui-mlflow/artifacts/episodes/MiniGrid-MazeS11N-v0_100M/episodes/* | wc -l

# Move 10k episodes (5M/100M steps) to eval
gsutil ls gs://humanui-mlflow/artifacts/episodes/MiniGrid-MazeS11N-v0_100M/episodes/s20-ep01* | wc -l
gsutil -m mv gs://humanui-mlflow/artifacts/episodes/MiniGrid-MazeS11N-v0_100M/episodes/s20-ep01* gs://humanui-mlflow/artifacts/episodes/MiniGrid-MazeS11N-v0_100M/eval/
gsutil ls gs://humanui-mlflow/artifacts/episodes/MiniGrid-MazeS11N-v0_100M/eval/* | wc -l

# Move 100 episodes to test
gsutil ls gs://humanui-mlflow/artifacts/episodes/MiniGrid-MazeS11N-v0_100M/episodes/s20-ep0000* | wc -l
gsutil -m cp gs://humanui-mlflow/artifacts/episodes/MiniGrid-MazeS11N-v0_100M/episodes/s20-ep0000* gs://humanui-mlflow/artifacts/episodes/MiniGrid-MazeS11N-v0_small/episodes/
gsutil -m cp gs://humanui-mlflow/artifacts/episodes/MiniGrid-MazeS11N-v0_100M/episodes/s20-ep0000* gs://humanui-mlflow/artifacts/episodes/MiniGrid-MazeS11N-v0_small/eval/
gsutil ls gs://humanui-mlflow/artifacts/episodes/MiniGrid-MazeS11N-v0_small/episodes/

# Copy to new bucket
gsutil -m cp -r gs://humanui-mlflow/artifacts/episodes/MiniGrid-MazeS11N-v0_100M gs://humanui-mlflow-episodes/

# Copy local
gsutil -m cp -r gs://humanui-mlflow/artifacts/episodes/MiniGrid-MazeS11N-v0_100M ./data/
