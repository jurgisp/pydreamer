FROM nvidia/cuda:10.1-cudnn7-runtime-ubuntu18.04
RUN apt-get update -y
RUN apt-get install -y python3-dev python3-pip python3-venv git
RUN pip3 install --upgrade pip

RUN pip3 install torch==1.8.1+cu101 torchvision==0.9.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html

# Atari dependencies
# RUN apt-get install -y libglu1-mesa-dev libgl1-mesa-dev libosmesa6-dev xvfb ffmpeg curl patchelf libglfw3 libglfw3-dev cmake zlib1g zlib1g-dev swig
# MiniWorld dependencies
RUN apt-get install -y freeglut3-dev xvfb mesa-utils

WORKDIR /app
ENV OMP_NUM_THREADS 1
ENV PYTHONUNBUFFERED 1

COPY requirements.txt .
RUN pip3 install -r requirements.txt
RUN pip3 install git+https://github.com/jurgisp/gym-minigrid.git@e979bc77a9377346a6a0311a257e8bbb218e611c#egg=gym-minigrid
RUN pip3 install git+https://github.com/jurgisp/gym-miniworld.git@a3bae10691d91a94056414c40d09a8dcd74fdc89#egg=gym-miniworld

COPY . .
