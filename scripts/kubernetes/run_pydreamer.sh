#!/bin/bash -e

if [[ $# -eq 0 ]] ; then
    echo 'Usage: ./run_pydreamer experiment config'
    exit 1
fi
EXPERIMENT="$1"
CONFIG="${2:-atari}"
RESUMEID="$3"

if [ ! -f ".env" ]; then
    echo ".env file not found - need it to set DOCKER_REPO, MLFLOW_TRACKING_URI"
    exit 1
fi
source .env
if [[ -z "$MLFLOW_TRACKING_URI" ]]; then
    echo "Must set MLFLOW_TRACKING_URI in .env"
    exit 1
fi
echo "Loaded variables from .env: DOCKER_REPO=$DOCKER_REPO, MLFLOW_TRACKING_URI=$MLFLOW_TRACKING_URI"

TAG=$(git rev-parse --short HEAD)
docker build . -f Dockerfile -t $DOCKER_REPO:$TAG
docker push $DOCKER_REPO:$TAG

RND=$(base64 < /dev/urandom | tr -d '[A-Z/+]' | head -c 6)
RESUMEID="${RESUMEID:-$RND}"

cat <<EOF | kubectl apply -f -
apiVersion: batch/v1
kind: Job
metadata:
  name: pydreamer-${EXPERIMENT//_/}-$RESUMEID-$RND
  namespace: default
spec:
  backoffLimit: 3
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
            - name: MLFLOW_TRACKING_USERNAME
              value: ${MLFLOW_TRACKING_USERNAME}
            - name: MLFLOW_TRACKING_PASSWORD
              value: ${MLFLOW_TRACKING_PASSWORD}
            - name: AZURE_STORAGE_ACCESS_KEY
              value: ${AZURE_STORAGE_ACCESS_KEY}
          volumeMounts:
            - name: dshm
              mountPath: /dev/shm
          command:
            - sh
          args:
            - scripts/xvfb_run.sh 
            - python3
            - train.py
            - --configs
            - defaults
            - $CONFIG
            - $EXPERIMENT
            - --run_name
            - $EXPERIMENT
            - --resume_id
            - $RESUMEID
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
      tolerations:  # Allow SPOT instances, if on Azure
        - key: "kubernetes.azure.com/scalesetpriority"
          operator: "Equal"
          value: "spot"
          effect: "NoSchedule"
      nodeSelector: ${K8S_NODE_SELECTOR}
EOF
