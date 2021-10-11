#!/bin/bash -e

if [[ $# -eq 0 ]] ; then
    echo 'Usage: ./run_pydreamer experiment config [mlflow_experiment]'
    exit
fi
EXPERIMENT="$1"
CONFIG="${2:-atari}"
MLFLOW_EXPERIMENT_NAME="${3:-Default}"

if [ ! -f ".env" ]; then
    echo ".env file not found - need it to set DOCKER_REPO, MLFLOW_TRACKING_URI"
    exit
fi
source .env
echo "Loaded variables from .env: DOCKER_REPO=$DOCKER_REPO, MLFLOW_TRACKING_URI=$MLFLOW_TRACKING_URI"

TAG=$(git describe --tags | sed 's/-g[a-z0-9]\{7\}//')
docker build . -f Dockerfile -t $DOCKER_REPO:$TAG
docker push $DOCKER_REPO:$TAG

RND=$(base64 < /dev/urandom | tr -d '[A-Z/+]' | head -c 6)
cat <<EOF | kubectl apply -f -
apiVersion: batch/v1
kind: Job
metadata:
  name: pydreamer-${EXPERIMENT//_/}-$RND
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
      tolerations:  # Allow SPOT instances, if on Azure
        - key: "kubernetes.azure.com/scalesetpriority"
          operator: "Equal"
          value: "spot"
          effect: "NoSchedule"
      nodeSelector: ${K8S_NODE_SELECTOR}
EOF
