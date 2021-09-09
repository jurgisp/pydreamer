## Build

docker build . -f kubernetes/Dockerfile -t pydreamer


## Run with local MLflow tracking server

docker run -it pydreamer \
    bash scripts/xvfb_run.sh \
    python3 train.py --configs defaults dmlab debug --run_name debug


## Run with remote MLflow tracking server

EXP_CONFIG="dmlabtmaze_online"  # <=== Select experiment from config/experiments.yaml
BASE_CONFIG="dmlab"
MLFLOW_TRACKING_URI="http://mlflow.threethirds.ai:30000/"
MLFLOW_EXPERIMENT_NAME="d2_google"
GOOGLE_APPLICATION_CREDENTIALS="/app/.gcs_credentials"

docker run \
    --env MLFLOW_TRACKING_URI="$MLFLOW_TRACKING_URI" \
    --env MLFLOW_EXPERIMENT_NAME="$MLFLOW_EXPERIMENT_NAME" \
    --env GOOGLE_APPLICATION_CREDENTIALS="$GOOGLE_APPLICATION_CREDENTIALS" \
    -it pydreamer \
    bash scripts/xvfb_run.sh \
    python3 train.py --configs defaults $BASE_CONFIG $EXP_CONFIG --run_name $EXP_CONFIG
