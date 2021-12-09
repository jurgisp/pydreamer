#!/bin/bash -e

VARIANT=$1

if [ ! -f ".env" ]; then
    echo ".env file not found - need it to set DOCKER_REPO"
    exit 1
fi
source .env
if [[ -z "$DOCKER_REPO_BASE" ]]; then
    echo "Must set DOCKER_REPO_BASE in .env"
    exit 1
fi
echo "Loaded variables from .env: DOCKER_REPO_BASE=$DOCKER_REPO_BASE"

docker build . -f Dockerfile -t $DOCKER_REPO_BASE-$VARIANT --build-arg ENV=$VARIANT --build-arg TYPE=base
docker push $DOCKER_REPO_BASE-$VARIANT
