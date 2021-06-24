conda activate pydreamer

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


## Copy data to shared disk

gcloud compute instances attach-disk "adhoc-jurgis" --disk "gke-shared-disk"
gcloud compute instances start adhoc-jurgis
gcloud compute ssh adhoc-jurgis
sudo mount -o discard,defaults /dev/sdb /data

mkdir /data/MiniWorld-MazeS5GridN-v0_100M
mkdir /data/MiniWorld-MazeS5GridN-v0_100M/train
mkdir /data/MiniWorld-MazeS5GridN-v0_100M/eval

gsutil ls gs://humanui-mlflow-west4/artifacts/29/92a026d91fee4aa6b1df06e4a0b16750/artifacts/episodes | wc -l
cat << EOF > script.sh
# gsutil -m cp gs://humanui-mlflow-west4/artifacts/29/92a026d91fee4aa6b1df06e4a0b16750/artifacts/episodes/* /data/MiniWorld-MazeS5GridN-v0_100M/train
gsutil -m cp gs://humanui-mlflow-west4/artifacts/29/4dd3e103003b4348bb3fe77b7bf59f44/artifacts/episodes/* /data/MiniWorld-MazeS5GridN-v0_100M/train
gsutil -m cp gs://humanui-mlflow-west4/artifacts/29/e1ac3198da6a4d7ca8f69e0e18f88867/artifacts/episodes/* /data/MiniWorld-MazeS5GridN-v0_100M/train
gsutil -m cp gs://humanui-mlflow-west4/artifacts/29/e4eec3178632498094274c4be820cefc/artifacts/episodes/* /data/MiniWorld-MazeS5GridN-v0_100M/train
gsutil -m cp gs://humanui-mlflow-west4/artifacts/29/cc65392bba694374906fe2893bdb55f6/artifacts/episodes/* /data/MiniWorld-MazeS5GridN-v0_100M/train
gsutil -m cp gs://humanui-mlflow-west4/artifacts/29/345d158cd5ce47e6a2b8f9bbc061ea66/artifacts/episodes/* /data/MiniWorld-MazeS5GridN-v0_100M/train
gsutil -m cp gs://humanui-mlflow-west4/artifacts/29/a47ec03621a1491180e801c2279a6ed8/artifacts/episodes/* /data/MiniWorld-MazeS5GridN-v0_100M/train
gsutil -m cp gs://humanui-mlflow-west4/artifacts/29/474fa57a831f4f0b93886889a7ad5b75/artifacts/episodes/* /data/MiniWorld-MazeS5GridN-v0_100M/train
gsutil -m cp gs://humanui-mlflow-west4/artifacts/29/b8d366c826544b699832f5dec5794c13/artifacts/episodes/* /data/MiniWorld-MazeS5GridN-v0_100M/train
gsutil -m cp gs://humanui-mlflow-west4/artifacts/29/89a3ccabf3f744cfb8eda99b98c6ae52/artifacts/episodes/* /data/MiniWorld-MazeS5GridN-v0_100M/train
EOF
chmod u+x script.sh
./script.sh & disown

ls /data/MiniWorld-MazeS5GridN-v0_100M/train | wc -l

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
