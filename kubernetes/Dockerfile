FROM nvidia/cuda:10.1-cudnn7-runtime-ubuntu18.04
RUN apt-get update -y
RUN apt-get install -y python3-dev python3-pip python3-venv git
RUN pip3 install --upgrade pip
# Atari dependencies
# RUN apt-get install -y libglu1-mesa-dev libgl1-mesa-dev libosmesa6-dev xvfb ffmpeg curl patchelf libglfw3 libglfw3-dev cmake zlib1g zlib1g-dev swig

RUN pip3 install torch==1.7.1+cu101 torchvision==0.8.2+cu101 -f https://download.pytorch.org/whl/torch_stable.html

WORKDIR /app
ENV OMP_NUM_THREADS 1
ENV PYTHONUNBUFFERED 1

COPY requirements.txt .
RUN pip3 install -r requirements.txt
RUN pip3 uninstall gym-minigrid -y
RUN pip3 install git+https://github.com/jurgisp/gym-minigrid.git@530e8f18380ebb1996076ef8350ff90ae6b165bb#egg=gym-minigrid

COPY . .
