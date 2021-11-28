docker build . -f Dockerfile -t pydreamer --build-arg ENV=standard

docker run -it \
    pydreamer \
    sh scripts/xvfb_run.sh python3 train.py --configs defaults dmc debug --run_name debug
