docker build . -f Dockerfile -t pydreamer

docker run -it \
    pydreamer \
    sh scripts/xvfb_run.sh python3 pydreamer/train.py --configs defaults dmc debug --run_name debug
