docker build . -f Dockerfile -t pydreamer

source .env

docker run -it \
    -e MLFLOW_TRACKING_URI=$MLFLOW_TRACKING_URI \
    -e MLFLOW_TRACKING_USERNAME=$MLFLOW_TRACKING_USERNAME \
    -e MLFLOW_TRACKING_PASSWORD=$MLFLOW_TRACKING_PASSWORD \
    -e AZURE_STORAGE_ACCESS_KEY=$AZURE_STORAGE_ACCESS_KEY \
    pydreamer \
    sh scripts/xvfb_run.sh python3 pydreamer/launch.py --configs defaults miniworld debug --run_name debug
