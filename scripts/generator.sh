conda activate pydreamer

## CLEANUP

gsutil -m rm -r gs://humanui-mlflow-west4/artifacts/29/*

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
