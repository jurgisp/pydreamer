#!/bin/bash -e

kubectl config use-context gke_human-ui_europe-west4-b_mlflow-cluster

MLFLOW_EXPERIMENT_NAME=$1
ENVID=$2
SEED=$3

TIMESTAMP=$(date +"%Y-%m-%d-%H-%M-%S")
TAG=$(git describe --tags | sed 's/-g[a-z0-9]\{7\}//')
RND=$(base64 < /dev/urandom | tr -d '[A-Z/+]' | head -c 6)

docker build . -f kubernetes/Dockerfile -t eu.gcr.io/human-ui/pydreamer:$TAG
docker push eu.gcr.io/human-ui/pydreamer:$TAG

cat <<EOF | kubectl apply -f -
apiVersion: batch/v1
kind: Job
metadata:
  name: generator-$TIMESTAMP
  namespace: default
spec:
  backoffLimit: 5
  template:
    spec:
      volumes:
        - name: google-cloud-key
          secret:
            secretName: mlflow-worker-key
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
          command:
            - python3
          args:
            - generator.py
            - --env_id
            - ${ENVID}
            - --num_steps
            - "10_000_000"
            - --seed
            - "${SEED}"
          resources:
            requests:
              memory: 4000Mi
              cpu: 3000m
      restartPolicy: Never
EOF