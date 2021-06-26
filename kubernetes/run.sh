#!/bin/bash -e

kubectl config use-context gke_human-ui_europe-west4-b_mlflow-cluster

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
  template:
    spec:
      volumes:
        - name: google-cloud-key
          secret:
            secretName: mlflow-worker-key
        - name: gke-shared-disk
          persistentVolumeClaim:
            claimName: gke-shared-disk-pvc
            readOnly: true
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
            - name: gke-shared-disk
              mountPath: /data
              readOnly: true
            - name: dshm
              mountPath: /dev/shm
          command:
            - python3
          args:
            - train_offline.py
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
              cpu: 3000m
              nvidia.com/gpu: 1
          securityContext:
            capabilities:
              add:
              - SYS_PTRACE
      restartPolicy: Never
EOF