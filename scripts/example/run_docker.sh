docker build . -f Dockerfile -t pydreamer

docker run -it pydreamer --configs defaults atari debug
