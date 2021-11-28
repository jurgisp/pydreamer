docker build . -f Dockerfile -t pydreamer

mlflow server &

docker run -it \
    -e MLFLOW_TRACKING_URI="http://docker.for.mac.host.internal:5000" --net=host \
    pydreamer \
    sh scripts/xvfb_run.sh python3 train.py --configs defaults atari debug --run_name debug
