BASE_CONFIG="dmlab"
EXP_CONFIG="dmlabtmaze_online"

MLFLOW_TRACKING_URI=""
MLFLOW_EXPERIMENT_NAME="Default"
# MLFLOW_TRACKING_URI="http://mlflow.threethirds.ai:30000/"
# MLFLOW_EXPERIMENT_NAME="d2_google"

docker build . -f kubernetes/Dockerfile -t pydreamer

docker run \
    --env MLFLOW_TRACKING_URI="$MLFLOW_TRACKING_URI" \
    --env MLFLOW_EXPERIMENT_NAME="$MLFLOW_EXPERIMENT_NAME" \
    -it pydreamer \
    bash scripts/xvfb_run.sh \
    python3 train.py --configs defaults $BASE_CONFIG $EXP_CONFIG --run_name $EXP_CONFIG
