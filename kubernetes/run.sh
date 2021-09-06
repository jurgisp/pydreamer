#!/bin/bash -e

kubectl config use-context mlflow-cluster

MLFLOW_EXPERIMENT_NAME=$1
CONFIG=$2
EXPERIMENT=$3
TIMESTAMP=$(date +"%Y-%m-%d-%H-%M-%S")
TAG=$(git describe --tags | sed 's/-g[a-z0-9]\{7\}//')
RND=$(base64 < /dev/urandom | tr -d '[A-Z/+]' | head -c 6)

docker build . -f kubernetes/Dockerfile -t eu.gcr.io/human-ui/pydreamer:$TAG
docker push eu.gcr.io/human-ui/pydreamer:$TAG

cat <<EOF | kubectl apply -f -
apiVersion: batch/v1
kind: Job
metadata:
  name: pydreamer-${EXPERIMENT//_/}-$TIMESTAMP
  namespace: default
spec:
  backoffLimit: 5
  ttlSecondsAfterFinished: 86400
  template:
    spec:
      volumes:
        - name: google-cloud-key
          secret:
            secretName: mlflow-worker-key
        - name: dshm
          emptyDir:
            medium: Memory
      containers:
        - name: dreamer
          imagePullPolicy: Always
          image: eu.gcr.io/human-ui/pydreamer:$TAG
          env:
            - name: GOOGLE_APPLICATION_CREDENTIALS
              value: /var/secrets/google/key.json
            - name: MLFLOW_TRACKING_URI
              value: http://mlflow-service.default.svc.cluster.local
            - name: MLFLOW_EXPERIMENT_NAME
              value: ${MLFLOW_EXPERIMENT_NAME}
          volumeMounts:
            - name: google-cloud-key
              mountPath: /var/secrets/google
            - name: dshm
              mountPath: /dev/shm
          command:
            - bash
          args:
            - kubernetes/xvfb_run.sh 
            - python3
            - train.py
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
              memory: 4000Mi
              cpu: 12000m
              nvidia.com/gpu: 1
          securityContext:
            capabilities:
              add:
              - SYS_PTRACE
      restartPolicy: Never
EOF