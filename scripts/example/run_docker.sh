## Build

docker build . -f kubernetes/Dockerfile -t pydreamer

export MLFLOW_TRACKING_URI="http://mlflow.threethirds.ai:30000/"
docker run -it pydreamer \
    --env MLFLOW_TRACKING_URI \
    sh scripts/xvfb_run.sh python3 pydreamer/train.py --configs defaults dmlab debug --run_name debug
