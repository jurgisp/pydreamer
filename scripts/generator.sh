conda activate pydreamer

## CLEANUP

gsutil -m rm -r gs://humanui-mlflow-west4/artifacts/29/*

## Generate MiniWorld-MazeS5A4S-dijkstra

for i in {0..19}; do
    echo $i
    ./kubernetes/run_generator_xvfb.sh dreamer2_episodes MiniWorld-MazeS5A4S-v0 maze_dijkstra 1500 5_000_000 $i
done					

cat << EOF > scripts/_tmp_copy.sh
gsutil -m mv -r gs://humanui-mlflow-west4/artifacts/29/343051f92785426a90c37e2908f5dba0/artifacts/episodes/* gs://humanui-mlflow-episodes/MiniWorld-MazeS5A4S-dijkstra/eval  # 10% for eval
gsutil -m mv -r gs://humanui-mlflow-west4/artifacts/29/92fc5ac6dd2642d395b8b3cdf34feb2d/artifacts/episodes/* gs://humanui-mlflow-episodes/MiniWorld-MazeS5A4S-dijkstra/eval  # 10% for eval
gsutil -m mv -r gs://humanui-mlflow-west4/artifacts/29/59908cc1ead44635b97eaaab54947b0e/artifacts/episodes/* gs://humanui-mlflow-episodes/MiniWorld-MazeS5A4S-dijkstra/train
gsutil -m mv -r gs://humanui-mlflow-west4/artifacts/29/0a9146d3b2904411904634c43434db62/artifacts/episodes/* gs://humanui-mlflow-episodes/MiniWorld-MazeS5A4S-dijkstra/train
gsutil -m mv -r gs://humanui-mlflow-west4/artifacts/29/2fe2b760adae44b287ab1e407e57a037/artifacts/episodes/* gs://humanui-mlflow-episodes/MiniWorld-MazeS5A4S-dijkstra/train
gsutil -m mv -r gs://humanui-mlflow-west4/artifacts/29/4c41d14209ff49aeb9d20c8ca0420c58/artifacts/episodes/* gs://humanui-mlflow-episodes/MiniWorld-MazeS5A4S-dijkstra/train
gsutil -m mv -r gs://humanui-mlflow-west4/artifacts/29/7a450ba170e44ede9c0e23e5437bc261/artifacts/episodes/* gs://humanui-mlflow-episodes/MiniWorld-MazeS5A4S-dijkstra/train
gsutil -m mv -r gs://humanui-mlflow-west4/artifacts/29/aeb0badbc1f34aa9a507c2c6fe78ee6c/artifacts/episodes/* gs://humanui-mlflow-episodes/MiniWorld-MazeS5A4S-dijkstra/train
gsutil -m mv -r gs://humanui-mlflow-west4/artifacts/29/bcdcdbcfabeb446188f9cf809bedd88b/artifacts/episodes/* gs://humanui-mlflow-episodes/MiniWorld-MazeS5A4S-dijkstra/train
gsutil -m mv -r gs://humanui-mlflow-west4/artifacts/29/d588bb2966554db882463cd82a889720/artifacts/episodes/* gs://humanui-mlflow-episodes/MiniWorld-MazeS5A4S-dijkstra/train
gsutil -m mv -r gs://humanui-mlflow-west4/artifacts/29/b0e4e5acf28a4f8daecea8bf58385f4f/artifacts/episodes/* gs://humanui-mlflow-episodes/MiniWorld-MazeS5A4S-dijkstra/train
gsutil -m mv -r gs://humanui-mlflow-west4/artifacts/29/4335ef144797427689c7bb9344dcd3ee/artifacts/episodes/* gs://humanui-mlflow-episodes/MiniWorld-MazeS5A4S-dijkstra/train
gsutil -m mv -r gs://humanui-mlflow-west4/artifacts/29/d3a9c12f05a346cbb0319c66e765be6e/artifacts/episodes/* gs://humanui-mlflow-episodes/MiniWorld-MazeS5A4S-dijkstra/train
gsutil -m mv -r gs://humanui-mlflow-west4/artifacts/29/05d37cc9a2194718b82b47945b123f15/artifacts/episodes/* gs://humanui-mlflow-episodes/MiniWorld-MazeS5A4S-dijkstra/train
gsutil -m mv -r gs://humanui-mlflow-west4/artifacts/29/f5dd3a4bc34f4cafb6cfa2bfee634b7e/artifacts/episodes/* gs://humanui-mlflow-episodes/MiniWorld-MazeS5A4S-dijkstra/train
gsutil -m mv -r gs://humanui-mlflow-west4/artifacts/29/6654af93ab80421fb71f6a8d5073581f/artifacts/episodes/* gs://humanui-mlflow-episodes/MiniWorld-MazeS5A4S-dijkstra/train
gsutil -m mv -r gs://humanui-mlflow-west4/artifacts/29/2fadee9381704b92a2e1c099119f7730/artifacts/episodes/* gs://humanui-mlflow-episodes/MiniWorld-MazeS5A4S-dijkstra/train
gsutil -m mv -r gs://humanui-mlflow-west4/artifacts/29/038e5b299b3746e9a28da3d4983906de/artifacts/episodes/* gs://humanui-mlflow-episodes/MiniWorld-MazeS5A4S-dijkstra/train
gsutil -m mv -r gs://humanui-mlflow-west4/artifacts/29/e9e3456d9ba54273858e64c747dccd55/artifacts/episodes/* gs://humanui-mlflow-episodes/MiniWorld-MazeS5A4S-dijkstra/train
EOF
scripts/_tmp_copy.sh


gsutil ls gs://humanui-mlflow-episodes/MiniWorld-MazeS5A4S-dijkstra/eval | wc -l
gsutil ls gs://humanui-mlflow-episodes/MiniWorld-MazeS5A4S-dijkstra/train | wc -l


## Generate MiniWorld-MazeS5NS-dijkstra-pixmap

for i in {0..19}; do
    echo $i
    ./kubernetes/run_generator_xvfb.sh dreamer2_episodes MiniWorld-MazeS5NS-v0 maze_dijkstra 1500 5_000_000 $i
done

cat << EOF > scripts/_tmp_copy.sh
gsutil -m mv -r gs://humanui-mlflow-west4/artifacts/29/3160e2128f174bba82505524d6d955dd/artifacts/episodes/* gs://humanui-mlflow-episodes/MiniWorld-MazeS5NS-dijkstra-pixmap/eval  # 10% for eval
gsutil -m mv -r gs://humanui-mlflow-west4/artifacts/29/b2cf00479c644223a5c1e2e4d1d699a5/artifacts/episodes/* gs://humanui-mlflow-episodes/MiniWorld-MazeS5NS-dijkstra-pixmap/eval  # 10% for eval
gsutil -m mv -r gs://humanui-mlflow-west4/artifacts/29/c753e038b7a349f0b4d189dcfef6eca3/artifacts/episodes/* gs://humanui-mlflow-episodes/MiniWorld-MazeS5NS-dijkstra-pixmap/train
gsutil -m mv -r gs://humanui-mlflow-west4/artifacts/29/c714d4ccabbc49c19ed0965ea5eac4c9/artifacts/episodes/* gs://humanui-mlflow-episodes/MiniWorld-MazeS5NS-dijkstra-pixmap/train
gsutil -m mv -r gs://humanui-mlflow-west4/artifacts/29/9e2d6491d0c54e19a8f0b69069f60912/artifacts/episodes/* gs://humanui-mlflow-episodes/MiniWorld-MazeS5NS-dijkstra-pixmap/train
gsutil -m mv -r gs://humanui-mlflow-west4/artifacts/29/18fec7930bb643e8b666ba91ed324664/artifacts/episodes/* gs://humanui-mlflow-episodes/MiniWorld-MazeS5NS-dijkstra-pixmap/train
gsutil -m mv -r gs://humanui-mlflow-west4/artifacts/29/3c248a43c14641699964c13ba3c7ffb7/artifacts/episodes/* gs://humanui-mlflow-episodes/MiniWorld-MazeS5NS-dijkstra-pixmap/train
gsutil -m mv -r gs://humanui-mlflow-west4/artifacts/29/a1dc519fbe3a4f9e9072ab8528a47515/artifacts/episodes/* gs://humanui-mlflow-episodes/MiniWorld-MazeS5NS-dijkstra-pixmap/train
gsutil -m mv -r gs://humanui-mlflow-west4/artifacts/29/6296e822a0b1465f88d72b6c36715629/artifacts/episodes/* gs://humanui-mlflow-episodes/MiniWorld-MazeS5NS-dijkstra-pixmap/train
gsutil -m mv -r gs://humanui-mlflow-west4/artifacts/29/4c030d25f450411084cd2aeebf428e3c/artifacts/episodes/* gs://humanui-mlflow-episodes/MiniWorld-MazeS5NS-dijkstra-pixmap/train
gsutil -m mv -r gs://humanui-mlflow-west4/artifacts/29/9366e73c61234a83bafd54ab000c3fd0/artifacts/episodes/* gs://humanui-mlflow-episodes/MiniWorld-MazeS5NS-dijkstra-pixmap/train
gsutil -m mv -r gs://humanui-mlflow-west4/artifacts/29/f770d9986d57435182d102820661fec5/artifacts/episodes/* gs://humanui-mlflow-episodes/MiniWorld-MazeS5NS-dijkstra-pixmap/train
gsutil -m mv -r gs://humanui-mlflow-west4/artifacts/29/70fb12d5d5d84489b99e16bcb256bebe/artifacts/episodes/* gs://humanui-mlflow-episodes/MiniWorld-MazeS5NS-dijkstra-pixmap/train
gsutil -m mv -r gs://humanui-mlflow-west4/artifacts/29/5fbcb63e5d3a454d980ad7c65f960dab/artifacts/episodes/* gs://humanui-mlflow-episodes/MiniWorld-MazeS5NS-dijkstra-pixmap/train
gsutil -m mv -r gs://humanui-mlflow-west4/artifacts/29/b55b766f3ef34957836b7d0f0c13c838/artifacts/episodes/* gs://humanui-mlflow-episodes/MiniWorld-MazeS5NS-dijkstra-pixmap/train
gsutil -m mv -r gs://humanui-mlflow-west4/artifacts/29/7d0c21ca4ed6413c9ccdfab601d26a28/artifacts/episodes/* gs://humanui-mlflow-episodes/MiniWorld-MazeS5NS-dijkstra-pixmap/train
gsutil -m mv -r gs://humanui-mlflow-west4/artifacts/29/1d44146ed5c546e3b4def676d687b32c/artifacts/episodes/* gs://humanui-mlflow-episodes/MiniWorld-MazeS5NS-dijkstra-pixmap/train
gsutil -m mv -r gs://humanui-mlflow-west4/artifacts/29/09710649cf684e67bb3e72d693c1e56c/artifacts/episodes/* gs://humanui-mlflow-episodes/MiniWorld-MazeS5NS-dijkstra-pixmap/train
gsutil -m mv -r gs://humanui-mlflow-west4/artifacts/29/f43290cd12e7426787f23d6173d96b9d/artifacts/episodes/* gs://humanui-mlflow-episodes/MiniWorld-MazeS5NS-dijkstra-pixmap/train
gsutil -m mv -r gs://humanui-mlflow-west4/artifacts/29/f58a60c435c441eb85698fddd739b903/artifacts/episodes/* gs://humanui-mlflow-episodes/MiniWorld-MazeS5NS-dijkstra-pixmap/train
gsutil -m mv -r gs://humanui-mlflow-west4/artifacts/29/315e5d3619744690980664dbbcc1ea7f/artifacts/episodes/* gs://humanui-mlflow-episodes/MiniWorld-MazeS5NS-dijkstra-pixmap/train
gsutil -m mv -r gs://humanui-mlflow-west4/artifacts/29/c898caa4b7a5432b877db9818573a5bc/artifacts/episodes/* gs://humanui-mlflow-episodes/MiniWorld-MazeS5NS-dijkstra-pixmap/train
gsutil -m mv -r gs://humanui-mlflow-west4/artifacts/29/b84a240b5caf4e7eaefd5f430f336bcb/artifacts/episodes/* gs://humanui-mlflow-episodes/MiniWorld-MazeS5NS-dijkstra-pixmap/train
gsutil -m mv -r gs://humanui-mlflow-west4/artifacts/29/8dd99d3b48f7486488f1faccee667c66/artifacts/episodes/* gs://humanui-mlflow-episodes/MiniWorld-MazeS5NS-dijkstra-pixmap/train
EOF
scripts/_tmp_copy.sh



## Generate MiniWorld-MazeS5NS-dijkstra 

./kubernetes/run_generator_xvfb.sh dreamer2_episodes MiniWorld-MazeS5NS-v0 maze_dijkstra 1500 5_000_000 0
./kubernetes/run_generator_xvfb.sh dreamer2_episodes MiniWorld-MazeS5NS-v0 maze_dijkstra 1500 5_000_000 1
./kubernetes/run_generator_xvfb.sh dreamer2_episodes MiniWorld-MazeS5NS-v0 maze_dijkstra 1500 5_000_000 2
./kubernetes/run_generator_xvfb.sh dreamer2_episodes MiniWorld-MazeS5NS-v0 maze_dijkstra 1500 5_000_000 3
./kubernetes/run_generator_xvfb.sh dreamer2_episodes MiniWorld-MazeS5NS-v0 maze_dijkstra 1500 5_000_000 4
./kubernetes/run_generator_xvfb.sh dreamer2_episodes MiniWorld-MazeS5NS-v0 maze_dijkstra 1500 5_000_000 5
./kubernetes/run_generator_xvfb.sh dreamer2_episodes MiniWorld-MazeS5NS-v0 maze_dijkstra 1500 5_000_000 6
./kubernetes/run_generator_xvfb.sh dreamer2_episodes MiniWorld-MazeS5NS-v0 maze_dijkstra 1500 5_000_000 7
./kubernetes/run_generator_xvfb.sh dreamer2_episodes MiniWorld-MazeS5NS-v0 maze_dijkstra 1500 5_000_000 8
./kubernetes/run_generator_xvfb.sh dreamer2_episodes MiniWorld-MazeS5NS-v0 maze_dijkstra 1500 5_000_000 9
./kubernetes/run_generator_xvfb.sh dreamer2_episodes MiniWorld-MazeS5NS-v0 maze_dijkstra 1500 5_000_000 10
./kubernetes/run_generator_xvfb.sh dreamer2_episodes MiniWorld-MazeS5NS-v0 maze_dijkstra 1500 5_000_000 11
./kubernetes/run_generator_xvfb.sh dreamer2_episodes MiniWorld-MazeS5NS-v0 maze_dijkstra 1500 5_000_000 12
./kubernetes/run_generator_xvfb.sh dreamer2_episodes MiniWorld-MazeS5NS-v0 maze_dijkstra 1500 5_000_000 13
./kubernetes/run_generator_xvfb.sh dreamer2_episodes MiniWorld-MazeS5NS-v0 maze_dijkstra 1500 5_000_000 14
./kubernetes/run_generator_xvfb.sh dreamer2_episodes MiniWorld-MazeS5NS-v0 maze_dijkstra 1500 5_000_000 15
./kubernetes/run_generator_xvfb.sh dreamer2_episodes MiniWorld-MazeS5NS-v0 maze_dijkstra 1500 5_000_000 16
./kubernetes/run_generator_xvfb.sh dreamer2_episodes MiniWorld-MazeS5NS-v0 maze_dijkstra 1500 5_000_000 17
./kubernetes/run_generator_xvfb.sh dreamer2_episodes MiniWorld-MazeS5NS-v0 maze_dijkstra 1500 5_000_000 18
./kubernetes/run_generator_xvfb.sh dreamer2_episodes MiniWorld-MazeS5NS-v0 maze_dijkstra 1500 5_000_000 19

cat << EOF > scripts/_tmp_copy.sh
gsutil -m mv -r gs://humanui-mlflow-west4/artifacts/29/0db09a5ac91947f5839d5ad427c6ac14/artifacts/episodes/* gs://humanui-mlflow-episodes/MiniWorld-MazeS5NS-dijkstra/eval  # 10% for eval
gsutil -m mv -r gs://humanui-mlflow-west4/artifacts/29/2dc2b03f66034b0abc5e0ead34054110/artifacts/episodes/* gs://humanui-mlflow-episodes/MiniWorld-MazeS5NS-dijkstra/eval  # 10% for eval
gsutil -m mv -r gs://humanui-mlflow-west4/artifacts/29/6c3cc45c4f6241028604ede4929ed904/artifacts/episodes/* gs://humanui-mlflow-episodes/MiniWorld-MazeS5NS-dijkstra/train
gsutil -m mv -r gs://humanui-mlflow-west4/artifacts/29/02a1236656e64c0090914cd5d7a52e52/artifacts/episodes/* gs://humanui-mlflow-episodes/MiniWorld-MazeS5NS-dijkstra/train
gsutil -m mv -r gs://humanui-mlflow-west4/artifacts/29/f8378da51a72427585931ce24cfe0dc0/artifacts/episodes/* gs://humanui-mlflow-episodes/MiniWorld-MazeS5NS-dijkstra/train
gsutil -m mv -r gs://humanui-mlflow-west4/artifacts/29/f80058ad3bf64d03a648e35ec08d203e/artifacts/episodes/* gs://humanui-mlflow-episodes/MiniWorld-MazeS5NS-dijkstra/train
gsutil -m mv -r gs://humanui-mlflow-west4/artifacts/29/1a0297c405014349baef45d11916790f/artifacts/episodes/* gs://humanui-mlflow-episodes/MiniWorld-MazeS5NS-dijkstra/train
gsutil -m mv -r gs://humanui-mlflow-west4/artifacts/29/5d28b34a1e394161953f75e70a6d7474/artifacts/episodes/* gs://humanui-mlflow-episodes/MiniWorld-MazeS5NS-dijkstra/train
gsutil -m mv -r gs://humanui-mlflow-west4/artifacts/29/771dac17a41c41b5bd8955b34b78cf6b/artifacts/episodes/* gs://humanui-mlflow-episodes/MiniWorld-MazeS5NS-dijkstra/train
gsutil -m mv -r gs://humanui-mlflow-west4/artifacts/29/490f9a3efb124aa6848ad4c613ee2dc8/artifacts/episodes/* gs://humanui-mlflow-episodes/MiniWorld-MazeS5NS-dijkstra/train
gsutil -m mv -r gs://humanui-mlflow-west4/artifacts/29/0101817631b84237af8fb83d7b93f55d/artifacts/episodes/* gs://humanui-mlflow-episodes/MiniWorld-MazeS5NS-dijkstra/train
gsutil -m mv -r gs://humanui-mlflow-west4/artifacts/29/8cd694af2029456691fc24f650a4cca1/artifacts/episodes/* gs://humanui-mlflow-episodes/MiniWorld-MazeS5NS-dijkstra/train
gsutil -m mv -r gs://humanui-mlflow-west4/artifacts/29/bd5bf7d807744a27a3fdf67a9dd2241b/artifacts/episodes/* gs://humanui-mlflow-episodes/MiniWorld-MazeS5NS-dijkstra/train
gsutil -m mv -r gs://humanui-mlflow-west4/artifacts/29/a4db87af215b4c9eb1761fc766b76c97/artifacts/episodes/* gs://humanui-mlflow-episodes/MiniWorld-MazeS5NS-dijkstra/train
gsutil -m mv -r gs://humanui-mlflow-west4/artifacts/29/a0efb08d3c564a41a199fde555a849cc/artifacts/episodes/* gs://humanui-mlflow-episodes/MiniWorld-MazeS5NS-dijkstra/train
gsutil -m mv -r gs://humanui-mlflow-west4/artifacts/29/815c152636014317be10381d822ed39b/artifacts/episodes/* gs://humanui-mlflow-episodes/MiniWorld-MazeS5NS-dijkstra/train
gsutil -m mv -r gs://humanui-mlflow-west4/artifacts/29/fd06cfc10d2f428bbb6dc14c86a406c3/artifacts/episodes/* gs://humanui-mlflow-episodes/MiniWorld-MazeS5NS-dijkstra/train
gsutil -m mv -r gs://humanui-mlflow-west4/artifacts/29/36ed549295be4abe98183d0d71045aaa/artifacts/episodes/* gs://humanui-mlflow-episodes/MiniWorld-MazeS5NS-dijkstra/train
gsutil -m mv -r gs://humanui-mlflow-west4/artifacts/29/3dbc006b714b4c7e8ea566c59ba4a58e/artifacts/episodes/* gs://humanui-mlflow-episodes/MiniWorld-MazeS5NS-dijkstra/train
gsutil -m mv -r gs://humanui-mlflow-west4/artifacts/29/b3686ea08e4c4913909c857988638ca2/artifacts/episodes/* gs://humanui-mlflow-episodes/MiniWorld-MazeS5NS-dijkstra/train
gsutil -m mv -r gs://humanui-mlflow-west4/artifacts/29/7b7b8b60ec0b4f5b9ef33b7fca0562d2/artifacts/episodes/* gs://humanui-mlflow-episodes/MiniWorld-MazeS5NS-dijkstra/train
gsutil -m mv -r gs://humanui-mlflow-west4/artifacts/29/2bc71e9c1a414e66b122accb8bfcdf91/artifacts/episodes/* gs://humanui-mlflow-episodes/MiniWorld-MazeS5NS-dijkstra/train
gsutil -m mv -r gs://humanui-mlflow-west4/artifacts/29/e7531fcfeb70480aafa139ff087758f3/artifacts/episodes/* gs://humanui-mlflow-episodes/MiniWorld-MazeS5NS-dijkstra/train
gsutil -m mv -r gs://humanui-mlflow-west4/artifacts/29/a4a1705aed2a401681bb2a401d35b1ba/artifacts/episodes/* gs://humanui-mlflow-episodes/MiniWorld-MazeS5NS-dijkstra/train
EOF
scripts/_tmp_copy.sh

gsutil ls gs://humanui-mlflow-episodes/MiniWorld-MazeS5NS-dijkstra/eval | wc -l
gsutil ls gs://humanui-mlflow-episodes/MiniWorld-MazeS5NS-dijkstra/train | wc -l

## Generate MiniWorld-MazeS5GridNS-dijkstra 

./kubernetes/run_generator_xvfb.sh dreamer2_episodes MiniWorld-MazeS5GridNS-v0 maze_dijkstra 500 10_000_000 0
./kubernetes/run_generator_xvfb.sh dreamer2_episodes MiniWorld-MazeS5GridNS-v0 maze_dijkstra 500 10_000_000 1
./kubernetes/run_generator_xvfb.sh dreamer2_episodes MiniWorld-MazeS5GridNS-v0 maze_dijkstra 500 10_000_000 2
./kubernetes/run_generator_xvfb.sh dreamer2_episodes MiniWorld-MazeS5GridNS-v0 maze_dijkstra 500 10_000_000 3
./kubernetes/run_generator_xvfb.sh dreamer2_episodes MiniWorld-MazeS5GridNS-v0 maze_dijkstra 500 10_000_000 4
./kubernetes/run_generator_xvfb.sh dreamer2_episodes MiniWorld-MazeS5GridNS-v0 maze_dijkstra 500 10_000_000 5
./kubernetes/run_generator_xvfb.sh dreamer2_episodes MiniWorld-MazeS5GridNS-v0 maze_dijkstra 500 10_000_000 6
./kubernetes/run_generator_xvfb.sh dreamer2_episodes MiniWorld-MazeS5GridNS-v0 maze_dijkstra 500 10_000_000 7
./kubernetes/run_generator_xvfb.sh dreamer2_episodes MiniWorld-MazeS5GridNS-v0 maze_dijkstra 500 10_000_000 8
./kubernetes/run_generator_xvfb.sh dreamer2_episodes MiniWorld-MazeS5GridNS-v0 maze_dijkstra 500 10_000_000 9
./kubernetes/run_generator_xvfb.sh dreamer2_episodes MiniWorld-MazeS5GridNS-v0 maze_dijkstra 500 10_000_000 10

cat << EOF > scripts/_copy.sh
gsutil -m mv -r gs://humanui-mlflow-west4/artifacts/29/3740983d76314017bf4f7aeb2c5745e3/artifacts/episodes/* gs://humanui-mlflow-episodes/MiniWorld-MazeS5GridNS-dijkstra/eval  # 10% for eval
gsutil -m mv -r gs://humanui-mlflow-west4/artifacts/29/f2b723d71d114c76941f69491e29fc68/artifacts/episodes/* gs://humanui-mlflow-episodes/MiniWorld-MazeS5GridNS-dijkstra/train
gsutil -m mv -r gs://humanui-mlflow-west4/artifacts/29/5c28ca2290614048b255046505046a31/artifacts/episodes/* gs://humanui-mlflow-episodes/MiniWorld-MazeS5GridNS-dijkstra/train
gsutil -m mv -r gs://humanui-mlflow-west4/artifacts/29/76c9f0dcd491427b9bc1e0db7d23d03f/artifacts/episodes/* gs://humanui-mlflow-episodes/MiniWorld-MazeS5GridNS-dijkstra/train
gsutil -m mv -r gs://humanui-mlflow-west4/artifacts/29/3c742a89262044e4ae0bbcdc77074193/artifacts/episodes/* gs://humanui-mlflow-episodes/MiniWorld-MazeS5GridNS-dijkstra/train
gsutil -m mv -r gs://humanui-mlflow-west4/artifacts/29/11695288a3884e40835c33597c2d1d0c/artifacts/episodes/* gs://humanui-mlflow-episodes/MiniWorld-MazeS5GridNS-dijkstra/train
gsutil -m mv -r gs://humanui-mlflow-west4/artifacts/29/0f2c4fdd41874b8e9a43113d589f738f/artifacts/episodes/* gs://humanui-mlflow-episodes/MiniWorld-MazeS5GridNS-dijkstra/train
gsutil -m mv -r gs://humanui-mlflow-west4/artifacts/29/a20b022dfc824f0f92ebb891f5f9d5d9/artifacts/episodes/* gs://humanui-mlflow-episodes/MiniWorld-MazeS5GridNS-dijkstra/train
gsutil -m mv -r gs://humanui-mlflow-west4/artifacts/29/7eb8cd22f882485eb079c65710ac3504/artifacts/episodes/* gs://humanui-mlflow-episodes/MiniWorld-MazeS5GridNS-dijkstra/train
gsutil -m mv -r gs://humanui-mlflow-west4/artifacts/29/eec716a5918a4761bb073210732be085/artifacts/episodes/* gs://humanui-mlflow-episodes/MiniWorld-MazeS5GridNS-dijkstra/train
EOF
scripts/_copy.sh


## Generate MiniWorld-MazeS5GridNS-wander500 (no apples, top start, max 500)

./kubernetes/run_generator_xvfb.sh dreamer2_episodes MiniWorld-MazeS5GridNS-v0 minigrid_wander 500 10_000_000 0
./kubernetes/run_generator_xvfb.sh dreamer2_episodes MiniWorld-MazeS5GridNS-v0 minigrid_wander 500 10_000_000 1
./kubernetes/run_generator_xvfb.sh dreamer2_episodes MiniWorld-MazeS5GridNS-v0 minigrid_wander 500 10_000_000 2
./kubernetes/run_generator_xvfb.sh dreamer2_episodes MiniWorld-MazeS5GridNS-v0 minigrid_wander 500 10_000_000 3
./kubernetes/run_generator_xvfb.sh dreamer2_episodes MiniWorld-MazeS5GridNS-v0 minigrid_wander 500 10_000_000 4
./kubernetes/run_generator_xvfb.sh dreamer2_episodes MiniWorld-MazeS5GridNS-v0 minigrid_wander 500 10_000_000 5
./kubernetes/run_generator_xvfb.sh dreamer2_episodes MiniWorld-MazeS5GridNS-v0 minigrid_wander 500 10_000_000 6
./kubernetes/run_generator_xvfb.sh dreamer2_episodes MiniWorld-MazeS5GridNS-v0 minigrid_wander 500 10_000_000 7
./kubernetes/run_generator_xvfb.sh dreamer2_episodes MiniWorld-MazeS5GridNS-v0 minigrid_wander 500 10_000_000 8
./kubernetes/run_generator_xvfb.sh dreamer2_episodes MiniWorld-MazeS5GridNS-v0 minigrid_wander 500 10_000_000 9
./kubernetes/run_generator_xvfb.sh dreamer2_episodes MiniWorld-MazeS5GridNS-v0 minigrid_wander 500 10_000_000 10

gsutil -m mv -r gs://humanui-mlflow-west4/artifacts/29/bfc3f89e73ef41b08ebe433d852ce75d/artifacts/episodes/* gs://humanui-mlflow-episodes/MiniWorld-MazeS5GridNS-wander500/eval  # 10% for eval
gsutil -m mv -r gs://humanui-mlflow-west4/artifacts/29/222ecb30070947909cda3e6aea3c8645/artifacts/episodes/* gs://humanui-mlflow-episodes/MiniWorld-MazeS5GridNS-wander500/train
gsutil -m mv -r gs://humanui-mlflow-west4/artifacts/29/e8dba76536db44e4b30165265f06fccf/artifacts/episodes/* gs://humanui-mlflow-episodes/MiniWorld-MazeS5GridNS-wander500/train
gsutil -m mv -r gs://humanui-mlflow-west4/artifacts/29/3d2a5f38565042feafe4f1d8388eaf99/artifacts/episodes/* gs://humanui-mlflow-episodes/MiniWorld-MazeS5GridNS-wander500/train
gsutil -m mv -r gs://humanui-mlflow-west4/artifacts/29/597ee58494264741badd5e4b25f5f73f/artifacts/episodes/* gs://humanui-mlflow-episodes/MiniWorld-MazeS5GridNS-wander500/train
gsutil -m mv -r gs://humanui-mlflow-west4/artifacts/29/40bc6a1f9b6b41e79507e2693b449a31/artifacts/episodes/* gs://humanui-mlflow-episodes/MiniWorld-MazeS5GridNS-wander500/train
gsutil -m mv -r gs://humanui-mlflow-west4/artifacts/29/4b178a5377f44e1ab32b692f2ae119b6/artifacts/episodes/* gs://humanui-mlflow-episodes/MiniWorld-MazeS5GridNS-wander500/train
gsutil -m mv -r gs://humanui-mlflow-west4/artifacts/29/ffc68607d3724490a1a6e88f7c9f3860/artifacts/episodes/* gs://humanui-mlflow-episodes/MiniWorld-MazeS5GridNS-wander500/train
gsutil -m mv -r gs://humanui-mlflow-west4/artifacts/29/a39810cc20b249ad92fe3127aa0d5718/artifacts/episodes/* gs://humanui-mlflow-episodes/MiniWorld-MazeS5GridNS-wander500/train
gsutil -m mv -r gs://humanui-mlflow-west4/artifacts/29/ecabd84d0b464941a1dfead6390228b2/artifacts/episodes/* gs://humanui-mlflow-episodes/MiniWorld-MazeS5GridNS-wander500/train
gsutil -m mv -r gs://humanui-mlflow-west4/artifacts/29/71d2e909e2cb435abb9bd48befb0d9ab/artifacts/episodes/* gs://humanui-mlflow-episodes/MiniWorld-MazeS5GridNS-wander500/train
gsutil -m mv -r gs://humanui-mlflow-west4/artifacts/29/2d73e14cffd34d12bd8863d7cfb28801/artifacts/episodes/* gs://humanui-mlflow-episodes/MiniWorld-MazeS5GridNS-wander500/train
gsutil -m mv -r gs://humanui-mlflow-west4/artifacts/29/33c693fcdea84faba49bb2585ca10201/artifacts/episodes/* gs://humanui-mlflow-episodes/MiniWorld-MazeS5GridNS-wander500/train
gsutil -m mv -r gs://humanui-mlflow-west4/artifacts/29/a06dc3fbcf0f4e338c69f1cb376512ba/artifacts/episodes/* gs://humanui-mlflow-episodes/MiniWorld-MazeS5GridNS-wander500/train

gsutil ls gs://humanui-mlflow-episodes/MiniWorld-MazeS5GridNS-wander500/eval | wc -l
gsutil ls gs://humanui-mlflow-episodes/MiniWorld-MazeS5GridNS-wander500/train | wc -l

gsutil -m cp -r gs://humanui-mlflow-episodes/MiniWorld-MazeS5GridNS-wander500/train/s3-* gs://humanui-mlflow-episodes/MiniWorld-MazeS5GridNS-wander500/train40k
gsutil -m cp -r gs://humanui-mlflow-episodes/MiniWorld-MazeS5GridNS-wander500/train/s4-* gs://humanui-mlflow-episodes/MiniWorld-MazeS5GridNS-wander500/train40k
gsutil ls gs://humanui-mlflow-episodes/MiniWorld-MazeS5GridNS-wander500/train40k | wc -l


## Generate MiniWorld-MazeS5GridNS-wander (no apples, top start)

./kubernetes/run_generator_xvfb.sh dreamer2_episodes MiniWorld-MazeS5GridNS-v0 minigrid_wander 2500 10_000_000 0
./kubernetes/run_generator_xvfb.sh dreamer2_episodes MiniWorld-MazeS5GridNS-v0 minigrid_wander 2500 10_000_000 1
./kubernetes/run_generator_xvfb.sh dreamer2_episodes MiniWorld-MazeS5GridNS-v0 minigrid_wander 2500 10_000_000 2
./kubernetes/run_generator_xvfb.sh dreamer2_episodes MiniWorld-MazeS5GridNS-v0 minigrid_wander 2500 10_000_000 3
./kubernetes/run_generator_xvfb.sh dreamer2_episodes MiniWorld-MazeS5GridNS-v0 minigrid_wander 2500 10_000_000 4
./kubernetes/run_generator_xvfb.sh dreamer2_episodes MiniWorld-MazeS5GridNS-v0 minigrid_wander 2500 10_000_000 5
./kubernetes/run_generator_xvfb.sh dreamer2_episodes MiniWorld-MazeS5GridNS-v0 minigrid_wander 2500 10_000_000 6
./kubernetes/run_generator_xvfb.sh dreamer2_episodes MiniWorld-MazeS5GridNS-v0 minigrid_wander 2500 10_000_000 7
./kubernetes/run_generator_xvfb.sh dreamer2_episodes MiniWorld-MazeS5GridNS-v0 minigrid_wander 2500 10_000_000 8
./kubernetes/run_generator_xvfb.sh dreamer2_episodes MiniWorld-MazeS5GridNS-v0 minigrid_wander 2500 10_000_000 9
./kubernetes/run_generator_xvfb.sh dreamer2_episodes MiniWorld-MazeS5GridNS-v0 minigrid_wander 2500 10_000_000 10

./kubernetes/run_generator_xvfb.sh dreamer2_episodes MiniWorld-MazeS5GridNS-v0 minigrid_wander 2500 10_000_000 11
./kubernetes/run_generator_xvfb.sh dreamer2_episodes MiniWorld-MazeS5GridNS-v0 minigrid_wander 2500 10_000_000 12
./kubernetes/run_generator_xvfb.sh dreamer2_episodes MiniWorld-MazeS5GridNS-v0 minigrid_wander 2500 10_000_000 13
./kubernetes/run_generator_xvfb.sh dreamer2_episodes MiniWorld-MazeS5GridNS-v0 minigrid_wander 2500 10_000_000 14
./kubernetes/run_generator_xvfb.sh dreamer2_episodes MiniWorld-MazeS5GridNS-v0 minigrid_wander 2500 10_000_000 15
./kubernetes/run_generator_xvfb.sh dreamer2_episodes MiniWorld-MazeS5GridNS-v0 minigrid_wander 2500 10_000_000 16
./kubernetes/run_generator_xvfb.sh dreamer2_episodes MiniWorld-MazeS5GridNS-v0 minigrid_wander 2500 10_000_000 17
./kubernetes/run_generator_xvfb.sh dreamer2_episodes MiniWorld-MazeS5GridNS-v0 minigrid_wander 2500 10_000_000 18
./kubernetes/run_generator_xvfb.sh dreamer2_episodes MiniWorld-MazeS5GridNS-v0 minigrid_wander 2500 10_000_000 19

./kubernetes/run_generator_xvfb.sh dreamer2_episodes MiniWorld-MazeS5GridNS-v0 minigrid_wander 2500 10_000_000 20
./kubernetes/run_generator_xvfb.sh dreamer2_episodes MiniWorld-MazeS5GridNS-v0 minigrid_wander 2500 10_000_000 21
./kubernetes/run_generator_xvfb.sh dreamer2_episodes MiniWorld-MazeS5GridNS-v0 minigrid_wander 2500 10_000_000 22
./kubernetes/run_generator_xvfb.sh dreamer2_episodes MiniWorld-MazeS5GridNS-v0 minigrid_wander 2500 10_000_000 23
./kubernetes/run_generator_xvfb.sh dreamer2_episodes MiniWorld-MazeS5GridNS-v0 minigrid_wander 2500 10_000_000 24
./kubernetes/run_generator_xvfb.sh dreamer2_episodes MiniWorld-MazeS5GridNS-v0 minigrid_wander 2500 10_000_000 25
./kubernetes/run_generator_xvfb.sh dreamer2_episodes MiniWorld-MazeS5GridNS-v0 minigrid_wander 2500 10_000_000 26
./kubernetes/run_generator_xvfb.sh dreamer2_episodes MiniWorld-MazeS5GridNS-v0 minigrid_wander 2500 10_000_000 27
./kubernetes/run_generator_xvfb.sh dreamer2_episodes MiniWorld-MazeS5GridNS-v0 minigrid_wander 2500 10_000_000 28
./kubernetes/run_generator_xvfb.sh dreamer2_episodes MiniWorld-MazeS5GridNS-v0 minigrid_wander 2500 10_000_000 29

./kubernetes/run_generator_xvfb.sh dreamer2_episodes MiniWorld-MazeS5GridNS-v0 minigrid_wander 2500 10_000_000 30
./kubernetes/run_generator_xvfb.sh dreamer2_episodes MiniWorld-MazeS5GridNS-v0 minigrid_wander 2500 10_000_000 31
./kubernetes/run_generator_xvfb.sh dreamer2_episodes MiniWorld-MazeS5GridNS-v0 minigrid_wander 2500 10_000_000 32
./kubernetes/run_generator_xvfb.sh dreamer2_episodes MiniWorld-MazeS5GridNS-v0 minigrid_wander 2500 10_000_000 33
./kubernetes/run_generator_xvfb.sh dreamer2_episodes MiniWorld-MazeS5GridNS-v0 minigrid_wander 2500 10_000_000 34
./kubernetes/run_generator_xvfb.sh dreamer2_episodes MiniWorld-MazeS5GridNS-v0 minigrid_wander 2500 10_000_000 35
./kubernetes/run_generator_xvfb.sh dreamer2_episodes MiniWorld-MazeS5GridNS-v0 minigrid_wander 2500 10_000_000 36
./kubernetes/run_generator_xvfb.sh dreamer2_episodes MiniWorld-MazeS5GridNS-v0 minigrid_wander 2500 10_000_000 37
./kubernetes/run_generator_xvfb.sh dreamer2_episodes MiniWorld-MazeS5GridNS-v0 minigrid_wander 2500 10_000_000 38
./kubernetes/run_generator_xvfb.sh dreamer2_episodes MiniWorld-MazeS5GridNS-v0 minigrid_wander 2500 10_000_000 39

gsutil -m mv -r gs://humanui-mlflow-west4/artifacts/29/b77b45ec26e9435ba30a91accdd0960f/artifacts/episodes/* gs://humanui-mlflow-episodes/MiniWorld-MazeS5GridNS-wander/eval  # 10% for eval
gsutil -m mv -r gs://humanui-mlflow-west4/artifacts/29/eb54883be903471288bdcbcd2365f03d/artifacts/episodes/* gs://humanui-mlflow-episodes/MiniWorld-MazeS5GridNS-wander/eval  # 10% for eval
gsutil -m mv -r gs://humanui-mlflow-west4/artifacts/29/43a88019ec4147d7b7212efe7d18be15/artifacts/episodes/* gs://humanui-mlflow-episodes/MiniWorld-MazeS5GridNS-wander/train
gsutil -m mv -r gs://humanui-mlflow-west4/artifacts/29/01e8a1239e74473ba1363b1bb3c75d6b/artifacts/episodes/* gs://humanui-mlflow-episodes/MiniWorld-MazeS5GridNS-wander/train
gsutil -m mv -r gs://humanui-mlflow-west4/artifacts/29/460fc0fb54844fcd9f0ff2f1086f5190/artifacts/episodes/* gs://humanui-mlflow-episodes/MiniWorld-MazeS5GridNS-wander/train
gsutil -m mv -r gs://humanui-mlflow-west4/artifacts/29/30c8f7609dc5429384b43b2c6655d67d/artifacts/episodes/* gs://humanui-mlflow-episodes/MiniWorld-MazeS5GridNS-wander/train
gsutil -m mv -r gs://humanui-mlflow-west4/artifacts/29/c77c52477a9b4db0bb0adc8f86efd04e/artifacts/episodes/* gs://humanui-mlflow-episodes/MiniWorld-MazeS5GridNS-wander/train
gsutil -m mv -r gs://humanui-mlflow-west4/artifacts/29/7726dd1f2eca405cb970da52bee8255a/artifacts/episodes/* gs://humanui-mlflow-episodes/MiniWorld-MazeS5GridNS-wander/train
gsutil -m mv -r gs://humanui-mlflow-west4/artifacts/29/e4c344d8ed894f03ad32c09329cffa3e/artifacts/episodes/* gs://humanui-mlflow-episodes/MiniWorld-MazeS5GridNS-wander/train
gsutil -m mv -r gs://humanui-mlflow-west4/artifacts/29/f4f9afce2fb8428cb150814003f4cc30/artifacts/episodes/* gs://humanui-mlflow-episodes/MiniWorld-MazeS5GridNS-wander/train
gsutil -m mv -r gs://humanui-mlflow-west4/artifacts/29/042bb4d330b04792894b2f3171cadbfe/artifacts/episodes/* gs://humanui-mlflow-episodes/MiniWorld-MazeS5GridNS-wander/train


gsutil ls gs://humanui-mlflow-episodes/MiniWorld-MazeS5GridNS-wander/eval | wc -l
gsutil ls gs://humanui-mlflow-episodes/MiniWorld-MazeS5GridNS-wander/train | wc -l


## Generate MiniWorld-MazeS5GridNS (no apples, top start)

./kubernetes/run_generator_xvfb.sh dreamer2_episodes MiniWorld-MazeS5GridNS-v0 2500 0
./kubernetes/run_generator_xvfb.sh dreamer2_episodes MiniWorld-MazeS5GridNS-v0 2500 1
./kubernetes/run_generator_xvfb.sh dreamer2_episodes MiniWorld-MazeS5GridNS-v0 2500 2
./kubernetes/run_generator_xvfb.sh dreamer2_episodes MiniWorld-MazeS5GridNS-v0 2500 3
./kubernetes/run_generator_xvfb.sh dreamer2_episodes MiniWorld-MazeS5GridNS-v0 2500 4
./kubernetes/run_generator_xvfb.sh dreamer2_episodes MiniWorld-MazeS5GridNS-v0 2500 5
./kubernetes/run_generator_xvfb.sh dreamer2_episodes MiniWorld-MazeS5GridNS-v0 2500 6
./kubernetes/run_generator_xvfb.sh dreamer2_episodes MiniWorld-MazeS5GridNS-v0 2500 7
./kubernetes/run_generator_xvfb.sh dreamer2_episodes MiniWorld-MazeS5GridNS-v0 2500 8
./kubernetes/run_generator_xvfb.sh dreamer2_episodes MiniWorld-MazeS5GridNS-v0 2500 9
./kubernetes/run_generator_xvfb.sh dreamer2_episodes MiniWorld-MazeS5GridNS-v0 2500 10

gsutil -m mv -r gs://humanui-mlflow-west4/artifacts/29/51f7d080ee544cef87c34a3ca7fdcaea/artifacts/episodes/* gs://humanui-mlflow-episodes/MiniWorld-MazeS5GridNS-v0/eval  # 10% for eval
gsutil -m mv -r gs://humanui-mlflow-west4/artifacts/29/295a4c809171450fb42258c8be734796/artifacts/episodes/* gs://humanui-mlflow-episodes/MiniWorld-MazeS5GridNS-v0/eval  # 10% for eval
gsutil -m mv -r gs://humanui-mlflow-west4/artifacts/29/3548d352de814d0da6e65e5dc072ccb7/artifacts/episodes/* gs://humanui-mlflow-episodes/MiniWorld-MazeS5GridNS-v0/eval  # 10% for eval
gsutil -m mv -r gs://humanui-mlflow-west4/artifacts/29/f4013c40a5f24deba54855327ab979ae/artifacts/episodes/* gs://humanui-mlflow-episodes/MiniWorld-MazeS5GridNS-v0/eval  # 10% for eval
gsutil -m mv -r gs://humanui-mlflow-west4/artifacts/29/e44974a07e5f45118c198e7d3ab3f26a/artifacts/episodes/* gs://humanui-mlflow-episodes/MiniWorld-MazeS5GridNS-v0/train
gsutil -m mv -r gs://humanui-mlflow-west4/artifacts/29/33358928beed41ce87013832485f80a2/artifacts/episodes/* gs://humanui-mlflow-episodes/MiniWorld-MazeS5GridNS-v0/train
gsutil -m mv -r gs://humanui-mlflow-west4/artifacts/29/4e2a849357c64dbebc093804935ec4c1/artifacts/episodes/* gs://humanui-mlflow-episodes/MiniWorld-MazeS5GridNS-v0/train
gsutil -m mv -r gs://humanui-mlflow-west4/artifacts/29/564cbece26fb43718d623b9ce348d9f5/artifacts/episodes/* gs://humanui-mlflow-episodes/MiniWorld-MazeS5GridNS-v0/train
gsutil -m mv -r gs://humanui-mlflow-west4/artifacts/29/df2d219de00c45389cd1d02aa89e8571/artifacts/episodes/* gs://humanui-mlflow-episodes/MiniWorld-MazeS5GridNS-v0/train
gsutil -m mv -r gs://humanui-mlflow-west4/artifacts/29/d31b1673804946109e05c9c04d4dc335/artifacts/episodes/* gs://humanui-mlflow-episodes/MiniWorld-MazeS5GridNS-v0/train
gsutil -m mv -r gs://humanui-mlflow-west4/artifacts/29/035e3a68fe364f8ab172150badeb30a0/artifacts/episodes/* gs://humanui-mlflow-episodes/MiniWorld-MazeS5GridNS-v0/train
gsutil -m mv -r gs://humanui-mlflow-west4/artifacts/29/bc9cf5cd565643a5afa138df5cc90e18/artifacts/episodes/* gs://humanui-mlflow-episodes/MiniWorld-MazeS5GridNS-v0/train
gsutil -m mv -r gs://humanui-mlflow-west4/artifacts/29/7397737aead243b2909215758a324b31/artifacts/episodes/* gs://humanui-mlflow-episodes/MiniWorld-MazeS5GridNS-v0/train
gsutil -m mv -r gs://humanui-mlflow-west4/artifacts/29/e3d7b2ef0eeb4eaf933fd8167f2c3a45/artifacts/episodes/* gs://humanui-mlflow-episodes/MiniWorld-MazeS5GridNS-v0/train
gsutil -m mv -r gs://humanui-mlflow-west4/artifacts/29/5fc80babbf0341afaa07503d8831fad0/artifacts/episodes/* gs://humanui-mlflow-episodes/MiniWorld-MazeS5GridNS-v0/train
gsutil -m mv -r gs://humanui-mlflow-west4/artifacts/29/357d5bafe13f48e19257aac0e6f1503c/artifacts/episodes/* gs://humanui-mlflow-episodes/MiniWorld-MazeS5GridNS-v0/train


gsutil ls gs://humanui-mlflow-episodes/MiniWorld-MazeS5GridNS-v0/eval | wc -l
gsutil ls gs://humanui-mlflow-episodes/MiniWorld-MazeS5GridNS-v0/train | wc -l


## Generate MiniWorld-MazeS5GridNR (no apples, random start)

./kubernetes/run_generator_xvfb.sh dreamer2_episodes MiniWorld-MazeS5GridNR-v0 2500 0
./kubernetes/run_generator_xvfb.sh dreamer2_episodes MiniWorld-MazeS5GridNR-v0 2500 1
./kubernetes/run_generator_xvfb.sh dreamer2_episodes MiniWorld-MazeS5GridNR-v0 2500 2
./kubernetes/run_generator_xvfb.sh dreamer2_episodes MiniWorld-MazeS5GridNR-v0 2500 3
./kubernetes/run_generator_xvfb.sh dreamer2_episodes MiniWorld-MazeS5GridNR-v0 2500 4
./kubernetes/run_generator_xvfb.sh dreamer2_episodes MiniWorld-MazeS5GridNR-v0 2500 5
./kubernetes/run_generator_xvfb.sh dreamer2_episodes MiniWorld-MazeS5GridNR-v0 2500 6
./kubernetes/run_generator_xvfb.sh dreamer2_episodes MiniWorld-MazeS5GridNR-v0 2500 7
./kubernetes/run_generator_xvfb.sh dreamer2_episodes MiniWorld-MazeS5GridNR-v0 2500 8
./kubernetes/run_generator_xvfb.sh dreamer2_episodes MiniWorld-MazeS5GridNR-v0 2500 9
./kubernetes/run_generator_xvfb.sh dreamer2_episodes MiniWorld-MazeS5GridNR-v0 2500 10

gsutil -m mv -r gs://humanui-mlflow-west4/artifacts/29/233ed1c9a89d4308a840ebe5a68519b9/artifacts/episodes/* gs://humanui-mlflow-episodes/MiniWorld-MazeS5GridNR-v0/eval  # 10% for eval
gsutil -m mv -r gs://humanui-mlflow-west4/artifacts/29/2c8da60f1cbd485ca762a75c7f52f20d/artifacts/episodes/* gs://humanui-mlflow-episodes/MiniWorld-MazeS5GridNR-v0/eval  # 2nd eval, because less data generated
gsutil -m mv -r gs://humanui-mlflow-west4/artifacts/29/2a196895a78646a1a384428fd7621be1/artifacts/episodes/* gs://humanui-mlflow-episodes/MiniWorld-MazeS5GridNR-v0/train
gsutil -m mv -r gs://humanui-mlflow-west4/artifacts/29/3dd7471a8ce94b9892f5a5b7d48b10d4/artifacts/episodes/* gs://humanui-mlflow-episodes/MiniWorld-MazeS5GridNR-v0/train
gsutil -m mv -r gs://humanui-mlflow-west4/artifacts/29/f329c53ff98e434883770f349a4ea55f/artifacts/episodes/* gs://humanui-mlflow-episodes/MiniWorld-MazeS5GridNR-v0/train
gsutil -m mv -r gs://humanui-mlflow-west4/artifacts/29/e89ee97368e94613ba5e46c7e3e93c65/artifacts/episodes/* gs://humanui-mlflow-episodes/MiniWorld-MazeS5GridNR-v0/train
gsutil -m mv -r gs://humanui-mlflow-west4/artifacts/29/e5092eda91724c83b11d164e635f4a1c/artifacts/episodes/* gs://humanui-mlflow-episodes/MiniWorld-MazeS5GridNR-v0/train
gsutil -m mv -r gs://humanui-mlflow-west4/artifacts/29/b9ed2ad4e2704936b2b5054f2eaff7bf/artifacts/episodes/* gs://humanui-mlflow-episodes/MiniWorld-MazeS5GridNR-v0/train
gsutil -m mv -r gs://humanui-mlflow-west4/artifacts/29/74bcebf2aa374fe186d2a180d992d1b2/artifacts/episodes/* gs://humanui-mlflow-episodes/MiniWorld-MazeS5GridNR-v0/train
gsutil -m mv -r gs://humanui-mlflow-west4/artifacts/29/5040772353c0492cad5bb6af765b4589/artifacts/episodes/* gs://humanui-mlflow-episodes/MiniWorld-MazeS5GridNR-v0/train
gsutil -m mv -r gs://humanui-mlflow-west4/artifacts/29/b428507fff924f9abbca83fa0aa07ab6/artifacts/episodes/* gs://humanui-mlflow-episodes/MiniWorld-MazeS5GridNR-v0/train
gsutil -m mv -r gs://humanui-mlflow-west4/artifacts/29/7ae7741d7ea44fbe8fa865a2f0b2974f/artifacts/episodes/* gs://humanui-mlflow-episodes/MiniWorld-MazeS5GridNR-v0/train

gsutil ls gs://humanui-mlflow-episodes/MiniWorld-MazeS5GridNR-v0/train | wc -l
gsutil ls gs://humanui-mlflow-episodes/MiniWorld-MazeS5GridNR-v0/eval | wc -l


## Generate MiniWorld-MazeS5A4S (apples, start top-left)

./kubernetes/run_generator_xvfb.sh dreamer2_episodes MiniWorld-MazeS5A4S-v0 2500 0
./kubernetes/run_generator_xvfb.sh dreamer2_episodes MiniWorld-MazeS5A4S-v0 2500 1
./kubernetes/run_generator_xvfb.sh dreamer2_episodes MiniWorld-MazeS5A4S-v0 2500 2
./kubernetes/run_generator_xvfb.sh dreamer2_episodes MiniWorld-MazeS5A4S-v0 2500 3
./kubernetes/run_generator_xvfb.sh dreamer2_episodes MiniWorld-MazeS5A4S-v0 2500 4
./kubernetes/run_generator_xvfb.sh dreamer2_episodes MiniWorld-MazeS5A4S-v0 2500 5
./kubernetes/run_generator_xvfb.sh dreamer2_episodes MiniWorld-MazeS5A4S-v0 2500 6
./kubernetes/run_generator_xvfb.sh dreamer2_episodes MiniWorld-MazeS5A4S-v0 2500 7
./kubernetes/run_generator_xvfb.sh dreamer2_episodes MiniWorld-MazeS5A4S-v0 2500 8
./kubernetes/run_generator_xvfb.sh dreamer2_episodes MiniWorld-MazeS5A4S-v0 2500 9
./kubernetes/run_generator_xvfb.sh dreamer2_episodes MiniWorld-MazeS5A4S-v0 2500 10

gsutil -m mv -r gs://humanui-mlflow-west4/artifacts/29/29fe5e6065bc4ad28e3b4f4475b93193/artifacts/episodes/* gs://humanui-mlflow-episodes/MiniWorld-MazeS5A4S-v0/eval  # 10% for eval
gsutil -m mv -r gs://humanui-mlflow-west4/artifacts/29/1a836916afb74deb9a3e612f9799306e/artifacts/episodes/* gs://humanui-mlflow-episodes/MiniWorld-MazeS5A4S-v0/train
gsutil -m mv -r gs://humanui-mlflow-west4/artifacts/29/d2900dd1194c4b41aebf54b15ab04f9a/artifacts/episodes/* gs://humanui-mlflow-episodes/MiniWorld-MazeS5A4S-v0/train
gsutil -m mv -r gs://humanui-mlflow-west4/artifacts/29/7dcf2a3fdd4f448abefe4389d1755949/artifacts/episodes/* gs://humanui-mlflow-episodes/MiniWorld-MazeS5A4S-v0/train
gsutil -m mv -r gs://humanui-mlflow-west4/artifacts/29/dc1345c3591d453c828987b354ca3aa6/artifacts/episodes/* gs://humanui-mlflow-episodes/MiniWorld-MazeS5A4S-v0/train
gsutil -m mv -r gs://humanui-mlflow-west4/artifacts/29/94f5fde620c84e15b7833312a5e4f785/artifacts/episodes/* gs://humanui-mlflow-episodes/MiniWorld-MazeS5A4S-v0/train
gsutil -m mv -r gs://humanui-mlflow-west4/artifacts/29/be991f28191348ea81bd0d81589d606d/artifacts/episodes/* gs://humanui-mlflow-episodes/MiniWorld-MazeS5A4S-v0/train
gsutil -m mv -r gs://humanui-mlflow-west4/artifacts/29/cc86bba9514a45c0bf82e64079e1386d/artifacts/episodes/* gs://humanui-mlflow-episodes/MiniWorld-MazeS5A4S-v0/train
gsutil -m mv -r gs://humanui-mlflow-west4/artifacts/29/cc957d15f9e646e0b86fa4f2bb77c4ef/artifacts/episodes/* gs://humanui-mlflow-episodes/MiniWorld-MazeS5A4S-v0/train
gsutil -m mv -r gs://humanui-mlflow-west4/artifacts/29/efc5e0e81bf84107af89f23765f517fd/artifacts/episodes/* gs://humanui-mlflow-episodes/MiniWorld-MazeS5A4S-v0/train
gsutil -m mv -r gs://humanui-mlflow-west4/artifacts/29/e6234d5bdb184e21b18135426797e795/artifacts/episodes/* gs://humanui-mlflow-episodes/MiniWorld-MazeS5A4S-v0/train

gsutil ls gs://humanui-mlflow-episodes/MiniWorld-MazeS5A4S-v0/train | wc -l
gsutil ls gs://humanui-mlflow-episodes/MiniWorld-MazeS5A4S-v0/eval | wc -l

## Generate MiniWorld-MazeS5GridA4S (apples, grid, start top-left)

./kubernetes/run_generator_xvfb.sh dreamer2_episodes MiniWorld-MazeS5GridA4S-v0 2500 0
./kubernetes/run_generator_xvfb.sh dreamer2_episodes MiniWorld-MazeS5GridA4S-v0 2500 1
./kubernetes/run_generator_xvfb.sh dreamer2_episodes MiniWorld-MazeS5GridA4S-v0 2500 2
./kubernetes/run_generator_xvfb.sh dreamer2_episodes MiniWorld-MazeS5GridA4S-v0 2500 3
./kubernetes/run_generator_xvfb.sh dreamer2_episodes MiniWorld-MazeS5GridA4S-v0 2500 4
./kubernetes/run_generator_xvfb.sh dreamer2_episodes MiniWorld-MazeS5GridA4S-v0 2500 5
./kubernetes/run_generator_xvfb.sh dreamer2_episodes MiniWorld-MazeS5GridA4S-v0 2500 6
./kubernetes/run_generator_xvfb.sh dreamer2_episodes MiniWorld-MazeS5GridA4S-v0 2500 7
./kubernetes/run_generator_xvfb.sh dreamer2_episodes MiniWorld-MazeS5GridA4S-v0 2500 8
./kubernetes/run_generator_xvfb.sh dreamer2_episodes MiniWorld-MazeS5GridA4S-v0 2500 9
./kubernetes/run_generator_xvfb.sh dreamer2_episodes MiniWorld-MazeS5GridA4S-v0 2500 10

gsutil -m mv -r gs://humanui-mlflow-west4/artifacts/29/b7f496d7c0d94163ae1daa5906a138d1/artifacts/episodes/* gs://humanui-mlflow-episodes/MiniWorld-MazeS5GridA4S-v0/eval  # 10% for eval
gsutil -m mv -r gs://humanui-mlflow-west4/artifacts/29/be7508a90f4345a6a9ce6d1b1e272914/artifacts/episodes/* gs://humanui-mlflow-episodes/MiniWorld-MazeS5GridA4S-v0/train
gsutil -m mv -r gs://humanui-mlflow-west4/artifacts/29/22756dbae16c4f89af55a00f3e7cba49/artifacts/episodes/* gs://humanui-mlflow-episodes/MiniWorld-MazeS5GridA4S-v0/train
gsutil -m mv -r gs://humanui-mlflow-west4/artifacts/29/3756a034f1ae45df9feedc2795ae0bfc/artifacts/episodes/* gs://humanui-mlflow-episodes/MiniWorld-MazeS5GridA4S-v0/train
gsutil -m mv -r gs://humanui-mlflow-west4/artifacts/29/36bb1cbb4ae846dda975522488c138f2/artifacts/episodes/* gs://humanui-mlflow-episodes/MiniWorld-MazeS5GridA4S-v0/train
gsutil -m mv -r gs://humanui-mlflow-west4/artifacts/29/7877bcbcc93f4d58a3affa5ebcf5c0c3/artifacts/episodes/* gs://humanui-mlflow-episodes/MiniWorld-MazeS5GridA4S-v0/train
gsutil -m mv -r gs://humanui-mlflow-west4/artifacts/29/676c284ce17c400fb63aab9ad9fe8951/artifacts/episodes/* gs://humanui-mlflow-episodes/MiniWorld-MazeS5GridA4S-v0/train
gsutil -m mv -r gs://humanui-mlflow-west4/artifacts/29/bd545d905e08494b9b7aaee39a76ae83/artifacts/episodes/* gs://humanui-mlflow-episodes/MiniWorld-MazeS5GridA4S-v0/train
gsutil -m mv -r gs://humanui-mlflow-west4/artifacts/29/2443fbef35f046ea9318bf7e648eeebe/artifacts/episodes/* gs://humanui-mlflow-episodes/MiniWorld-MazeS5GridA4S-v0/train
gsutil -m mv -r gs://humanui-mlflow-west4/artifacts/29/648e529b2dbf46b68a2952e760382efe/artifacts/episodes/* gs://humanui-mlflow-episodes/MiniWorld-MazeS5GridA4S-v0/train
gsutil -m mv -r gs://humanui-mlflow-west4/artifacts/29/f32ede7f180542c688059a5ce5a83a22/artifacts/episodes/* gs://humanui-mlflow-episodes/MiniWorld-MazeS5GridA4S-v0/train

gsutil ls gs://humanui-mlflow-episodes/MiniWorld-MazeS5GridA4S-v0/train | wc -l
gsutil ls gs://humanui-mlflow-episodes/MiniWorld-MazeS5GridA4S-v0/eval | wc -l

## Generate MiniWorld-MazeS5GridA4 (apples, grid)

./kubernetes/run_generator_xvfb.sh dreamer2_episodes MiniWorld-MazeS5GridA4-v0 2500 0
./kubernetes/run_generator_xvfb.sh dreamer2_episodes MiniWorld-MazeS5GridA4-v0 2500 1
./kubernetes/run_generator_xvfb.sh dreamer2_episodes MiniWorld-MazeS5GridA4-v0 2500 2
./kubernetes/run_generator_xvfb.sh dreamer2_episodes MiniWorld-MazeS5GridA4-v0 2500 3
./kubernetes/run_generator_xvfb.sh dreamer2_episodes MiniWorld-MazeS5GridA4-v0 2500 4
./kubernetes/run_generator_xvfb.sh dreamer2_episodes MiniWorld-MazeS5GridA4-v0 2500 5
./kubernetes/run_generator_xvfb.sh dreamer2_episodes MiniWorld-MazeS5GridA4-v0 2500 6
./kubernetes/run_generator_xvfb.sh dreamer2_episodes MiniWorld-MazeS5GridA4-v0 2500 7
./kubernetes/run_generator_xvfb.sh dreamer2_episodes MiniWorld-MazeS5GridA4-v0 2500 8
./kubernetes/run_generator_xvfb.sh dreamer2_episodes MiniWorld-MazeS5GridA4-v0 2500 9
./kubernetes/run_generator_xvfb.sh dreamer2_episodes MiniWorld-MazeS5GridA4-v0 2500 10

gsutil -m cp -r gs://humanui-mlflow-west4/artifacts/29/e1b055b32eec4f43b1b7dd59b0d933f4/artifacts/episodes/* gs://humanui-mlflow-episodes/MiniWorld-MazeS5GridA4-v0/eval  # 10% for eval
gsutil -m cp -r gs://humanui-mlflow-west4/artifacts/29/d28d79fe39324658bbb2d3b9b24a3876/artifacts/episodes/* gs://humanui-mlflow-episodes/MiniWorld-MazeS5GridA4-v0/train
gsutil -m cp -r gs://humanui-mlflow-west4/artifacts/29/030614b973a74efc85618faa0401d4f1/artifacts/episodes/* gs://humanui-mlflow-episodes/MiniWorld-MazeS5GridA4-v0/train
gsutil -m cp -r gs://humanui-mlflow-west4/artifacts/29/4b586380b5654c92b05d6ac0a46314c1/artifacts/episodes/* gs://humanui-mlflow-episodes/MiniWorld-MazeS5GridA4-v0/train
gsutil -m cp -r gs://humanui-mlflow-west4/artifacts/29/2a9460b85ac14595adb045722467ca97/artifacts/episodes/* gs://humanui-mlflow-episodes/MiniWorld-MazeS5GridA4-v0/train
gsutil -m cp -r gs://humanui-mlflow-west4/artifacts/29/5e072ea671814d5587ae77b419ac5051/artifacts/episodes/* gs://humanui-mlflow-episodes/MiniWorld-MazeS5GridA4-v0/train
gsutil -m cp -r gs://humanui-mlflow-west4/artifacts/29/83278c2f7bbd439a911b869b7e8ac251/artifacts/episodes/* gs://humanui-mlflow-episodes/MiniWorld-MazeS5GridA4-v0/train
gsutil -m cp -r gs://humanui-mlflow-west4/artifacts/29/5697d562487444a28b0013b9ea6ba4e0/artifacts/episodes/* gs://humanui-mlflow-episodes/MiniWorld-MazeS5GridA4-v0/train
gsutil -m cp -r gs://humanui-mlflow-west4/artifacts/29/4a3ffc3804cc4929920e3582cb46b7b9/artifacts/episodes/* gs://humanui-mlflow-episodes/MiniWorld-MazeS5GridA4-v0/train
gsutil -m cp -r gs://humanui-mlflow-west4/artifacts/29/c9d88bbe485a4d97b79882f40180f417/artifacts/episodes/* gs://humanui-mlflow-episodes/MiniWorld-MazeS5GridA4-v0/train
gsutil -m cp -r gs://humanui-mlflow-west4/artifacts/29/91d026769f19449f910cca12d0e7d66e/artifacts/episodes/* gs://humanui-mlflow-episodes/MiniWorld-MazeS5GridA4-v0/train

gsutil ls gs://humanui-mlflow-episodes/MiniWorld-MazeS5GridA4-v0/train | wc -l
gsutil ls gs://humanui-mlflow-episodes/MiniWorld-MazeS5GridA4-v0/eval | wc -l

## Generate MiniWorld-MazeS5A4W (apples, white)

./kubernetes/run_generator_xvfb.sh dreamer2_episodes MiniWorld-MazeS5A4W-v0 0
./kubernetes/run_generator_xvfb.sh dreamer2_episodes MiniWorld-MazeS5A4W-v0 1
./kubernetes/run_generator_xvfb.sh dreamer2_episodes MiniWorld-MazeS5A4W-v0 2
./kubernetes/run_generator_xvfb.sh dreamer2_episodes MiniWorld-MazeS5A4W-v0 3
./kubernetes/run_generator_xvfb.sh dreamer2_episodes MiniWorld-MazeS5A4W-v0 4
./kubernetes/run_generator_xvfb.sh dreamer2_episodes MiniWorld-MazeS5A4W-v0 5
./kubernetes/run_generator_xvfb.sh dreamer2_episodes MiniWorld-MazeS5A4W-v0 6
./kubernetes/run_generator_xvfb.sh dreamer2_episodes MiniWorld-MazeS5A4W-v0 7
./kubernetes/run_generator_xvfb.sh dreamer2_episodes MiniWorld-MazeS5A4W-v0 8
./kubernetes/run_generator_xvfb.sh dreamer2_episodes MiniWorld-MazeS5A4W-v0 9
./kubernetes/run_generator_xvfb.sh dreamer2_episodes MiniWorld-MazeS5A4W-v0 10

gsutil -m cp -r gs://humanui-mlflow-west4/artifacts/29/77366abf3a1c4df389deb115b5b8b34d/artifacts/episodes/* gs://humanui-mlflow-episodes/MiniWorld-MazeS5A4W-v0/eval  # 10% for eval
gsutil -m cp -r gs://humanui-mlflow-west4/artifacts/29/5c5d62386b2a44cb9385392bc02c899b/artifacts/episodes/* gs://humanui-mlflow-episodes/MiniWorld-MazeS5A4W-v0/train
gsutil -m cp -r gs://humanui-mlflow-west4/artifacts/29/4ff581abc66349669747a169f4a2a2cd/artifacts/episodes/* gs://humanui-mlflow-episodes/MiniWorld-MazeS5A4W-v0/train
gsutil -m cp -r gs://humanui-mlflow-west4/artifacts/29/fbd2bd06dd344cb4834ff3623e118db1/artifacts/episodes/* gs://humanui-mlflow-episodes/MiniWorld-MazeS5A4W-v0/train
gsutil -m cp -r gs://humanui-mlflow-west4/artifacts/29/1fa2663a68f64326a07dc5f7e38b2abe/artifacts/episodes/* gs://humanui-mlflow-episodes/MiniWorld-MazeS5A4W-v0/train
gsutil -m cp -r gs://humanui-mlflow-west4/artifacts/29/16b0d5f591ef4961a73d6d777b1f92ca/artifacts/episodes/* gs://humanui-mlflow-episodes/MiniWorld-MazeS5A4W-v0/train
gsutil -m cp -r gs://humanui-mlflow-west4/artifacts/29/ed2a4bc12dd84ec6abb054e7f66d2a42/artifacts/episodes/* gs://humanui-mlflow-episodes/MiniWorld-MazeS5A4W-v0/train
gsutil -m cp -r gs://humanui-mlflow-west4/artifacts/29/d6b1daa24a36450aac26df2c714a8813/artifacts/episodes/* gs://humanui-mlflow-episodes/MiniWorld-MazeS5A4W-v0/train
gsutil -m cp -r gs://humanui-mlflow-west4/artifacts/29/8df076a3ea834f4bab9167843b321d65/artifacts/episodes/* gs://humanui-mlflow-episodes/MiniWorld-MazeS5A4W-v0/train
gsutil -m cp -r gs://humanui-mlflow-west4/artifacts/29/e859e48ebd2e4e7bac3f9b9465fccd46/artifacts/episodes/* gs://humanui-mlflow-episodes/MiniWorld-MazeS5A4W-v0/train
gsutil -m cp -r gs://humanui-mlflow-west4/artifacts/29/fae2b7a1eb3c42e09adae0eaf653505f/artifacts/episodes/* gs://humanui-mlflow-episodes/MiniWorld-MazeS5A4W-v0/train

gsutil ls gs://humanui-mlflow-episodes/MiniWorld-MazeS5A4W-v0/train | wc -l
gsutil ls gs://humanui-mlflow-episodes/MiniWorld-MazeS5A4W-v0/eval | wc -l


## Generate MiniWorld-MazeS5A4 (apples)

./kubernetes/run_generator_xvfb.sh dreamer2_episodes MiniWorld-MazeS5A4-v0 0
./kubernetes/run_generator_xvfb.sh dreamer2_episodes MiniWorld-MazeS5A4-v0 1
./kubernetes/run_generator_xvfb.sh dreamer2_episodes MiniWorld-MazeS5A4-v0 2
./kubernetes/run_generator_xvfb.sh dreamer2_episodes MiniWorld-MazeS5A4-v0 3
./kubernetes/run_generator_xvfb.sh dreamer2_episodes MiniWorld-MazeS5A4-v0 4
./kubernetes/run_generator_xvfb.sh dreamer2_episodes MiniWorld-MazeS5A4-v0 5
./kubernetes/run_generator_xvfb.sh dreamer2_episodes MiniWorld-MazeS5A4-v0 6
./kubernetes/run_generator_xvfb.sh dreamer2_episodes MiniWorld-MazeS5A4-v0 7
./kubernetes/run_generator_xvfb.sh dreamer2_episodes MiniWorld-MazeS5A4-v0 8
./kubernetes/run_generator_xvfb.sh dreamer2_episodes MiniWorld-MazeS5A4-v0 9
./kubernetes/run_generator_xvfb.sh dreamer2_episodes MiniWorld-MazeS5A4-v0 10

gsutil -m cp -r gs://humanui-mlflow-west4/artifacts/29/b202fe249cb84d6bbd017a04ef34949e/artifacts/episodes/* gs://humanui-mlflow-episodes/MiniWorld-MazeS5A4-v0/eval  # 10% for eval
gsutil -m cp -r gs://humanui-mlflow-west4/artifacts/29/7267a582727d4b61ae25f2bda155488a/artifacts/episodes/* gs://humanui-mlflow-episodes/MiniWorld-MazeS5A4-v0/train
gsutil -m cp -r gs://humanui-mlflow-west4/artifacts/29/a2f3ffc4291244f09338b5d60071d96a/artifacts/episodes/* gs://humanui-mlflow-episodes/MiniWorld-MazeS5A4-v0/train
gsutil -m cp -r gs://humanui-mlflow-west4/artifacts/29/c4352462ba904b9ea05144632e36f321/artifacts/episodes/* gs://humanui-mlflow-episodes/MiniWorld-MazeS5A4-v0/train
gsutil -m cp -r gs://humanui-mlflow-west4/artifacts/29/d4cab8bedb5b476697090830cacdd5ec/artifacts/episodes/* gs://humanui-mlflow-episodes/MiniWorld-MazeS5A4-v0/train
gsutil -m cp -r gs://humanui-mlflow-west4/artifacts/29/5b14205465174c2e8ed0fd7b32b8e748/artifacts/episodes/* gs://humanui-mlflow-episodes/MiniWorld-MazeS5A4-v0/train
gsutil -m cp -r gs://humanui-mlflow-west4/artifacts/29/37e3f06af34f4fdaafeebdb4a0a32ef4/artifacts/episodes/* gs://humanui-mlflow-episodes/MiniWorld-MazeS5A4-v0/train
gsutil -m cp -r gs://humanui-mlflow-west4/artifacts/29/9e0aba7e54cb4f52b9e46c710123e9cc/artifacts/episodes/* gs://humanui-mlflow-episodes/MiniWorld-MazeS5A4-v0/train
gsutil -m cp -r gs://humanui-mlflow-west4/artifacts/29/f10d81fa977c431eb38ff45eb570a7f6/artifacts/episodes/* gs://humanui-mlflow-episodes/MiniWorld-MazeS5A4-v0/train
gsutil -m cp -r gs://humanui-mlflow-west4/artifacts/29/597f1d9eb1a84320a25e774f1c4204a6/artifacts/episodes/* gs://humanui-mlflow-episodes/MiniWorld-MazeS5A4-v0/train
gsutil -m cp -r gs://humanui-mlflow-west4/artifacts/29/1cc3bb7503f34c61bff176625b1352ad/artifacts/episodes/* gs://humanui-mlflow-episodes/MiniWorld-MazeS5A4-v0/train

gsutil ls gs://humanui-mlflow-episodes/MiniWorld-MazeS5A4-v0/train | wc -l
gsutil ls gs://humanui-mlflow-episodes/MiniWorld-MazeS5A4-v0/eval | wc -l


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
./kubernetes/run_generator_xvfb.sh dreamer2_episodes MiniWorld-MazeS5N-v0 10

gsutil -m cp -r gs://humanui-mlflow-west4/artifacts/29/13365c5813eb412bb7e0babbbfdecb92/artifacts/episodes/* gs://humanui-mlflow-episodes/MiniWorld-MazeS5N-v0/eval  # 10% for eval
gsutil -m cp -r gs://humanui-mlflow-west4/artifacts/29/ac2a127e40544029ae86cce17b9e84e3/artifacts/episodes/* gs://humanui-mlflow-episodes/MiniWorld-MazeS5N-v0/train
gsutil -m cp -r gs://humanui-mlflow-west4/artifacts/29/0d456ba265ec4fa5a7af8b3f16970a76/artifacts/episodes/* gs://humanui-mlflow-episodes/MiniWorld-MazeS5N-v0/train
gsutil -m cp -r gs://humanui-mlflow-west4/artifacts/29/76f6f331a7e543e297618976ebb7ebae/artifacts/episodes/* gs://humanui-mlflow-episodes/MiniWorld-MazeS5N-v0/train
gsutil -m cp -r gs://humanui-mlflow-west4/artifacts/29/67dbd40da7a54fdebe0dd579093ee595/artifacts/episodes/* gs://humanui-mlflow-episodes/MiniWorld-MazeS5N-v0/train
gsutil -m cp -r gs://humanui-mlflow-west4/artifacts/29/bceca40e5ed44ab19b0e7dc9a12a0339/artifacts/episodes/* gs://humanui-mlflow-episodes/MiniWorld-MazeS5N-v0/train
gsutil -m cp -r gs://humanui-mlflow-west4/artifacts/29/006e1c05c2c848d9943607132c911509/artifacts/episodes/* gs://humanui-mlflow-episodes/MiniWorld-MazeS5N-v0/train
gsutil -m cp -r gs://humanui-mlflow-west4/artifacts/29/86e5b88eaff346baa1f00caa98165eba/artifacts/episodes/* gs://humanui-mlflow-episodes/MiniWorld-MazeS5N-v0/train
gsutil -m cp -r gs://humanui-mlflow-west4/artifacts/29/abf0109c085743049bfb5d76695a4722/artifacts/episodes/* gs://humanui-mlflow-episodes/MiniWorld-MazeS5N-v0/train
gsutil -m cp -r gs://humanui-mlflow-west4/artifacts/29/e2a302ccb6974555980370e5295255ed/artifacts/episodes/* gs://humanui-mlflow-episodes/MiniWorld-MazeS5N-v0/train
gsutil -m cp -r gs://humanui-mlflow-west4/artifacts/29/57cf27d90952402888628022addc16e1/artifacts/episodes/* gs://humanui-mlflow-episodes/MiniWorld-MazeS5N-v0/train

gsutil ls gs://humanui-mlflow-episodes/MiniWorld-MazeS5N-v0/train | wc -l
gsutil ls gs://humanui-mlflow-episodes/MiniWorld-MazeS5N-v0/eval | wc -l


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
