docker build . -f Dockerfile -t pydreamer --build-arg ENV=standard

docker run -it \
    pydreamer \
    --configs defaults atari debug --run_name debug
