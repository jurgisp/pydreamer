#!/bin/bash -e

MLFLOW_TRACKING_URI=">>TODO<<"
DOCKER_REPO=">>TODO<<"
DOCKER_FILE="Dockerfile"

if [[ $# -eq 0 ]] ; then
    echo 'Usage: ./run_pydreamer experiment config [mlflow_experiment]'
    exit 0
fi

EXPERIMENT="$1"
CONFIG="${2:-atari}"
MLFLOW_EXPERIMENT_NAME="${3:-Default}"
TIMESTAMP=$(date +"%Y-%m-%d-%H-%M-%S")
TAG=$(git describe --tags | sed 's/-g[a-z0-9]\{7\}//')
RND=$(base64 < /dev/urandom | tr -d '[A-Z/+]' | head -c 6)

docker build . -f $DOCKER_FILE -t $DOCKER_REPO:$TAG
docker push $DOCKER_REPO:$TAG

cat <<EOF | kubectl apply -f -
apiVersion: batch/v1
kind: Job
metadata:
  name: pydreamer-${EXPERIMENT//_/}-$RND
  namespace: default
spec:
  backoffLimit: 5
  ttlSecondsAfterFinished: 86400
  template:
    spec:
      volumes:
        - name: dshm
          emptyDir:
            medium: Memory
      containers:
        - name: pydreamer
          imagePullPolicy: Always
          image: $DOCKER_REPO:$TAG
          env:
            - name: MLFLOW_TRACKING_URI
              value: ${MLFLOW_TRACKING_URI}
            - name: MLFLOW_EXPERIMENT_NAME
              value: ${MLFLOW_EXPERIMENT_NAME}
          volumeMounts:
            - name: dshm
              mountPath: /dev/shm
          command:
            - sh
          args:
            - scripts/xvfb_run.sh 
            - python3
            - pydreamer/train.py
            - --configs
            - defaults
            - $CONFIG
            - $EXPERIMENT
            - --run_name
            - $EXPERIMENT
            - --resume_id
            - $RND
          resources:
            limits:
              nvidia.com/gpu: 1
            requests:
              memory: 8000Mi
              cpu: 4000m
              nvidia.com/gpu: 1
          securityContext:
            capabilities:
              add:
              - SYS_PTRACE
      restartPolicy: Never
EOF
