FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel AS base

# Fix nvidia package repo (see https://github.com/NVIDIA/nvidia-docker/issues/1631)
RUN rm /etc/apt/sources.list.d/cuda.list && \
    rm /etc/apt/sources.list.d/nvidia-ml.list && \
    apt-key del 7fa2af80 && \
    apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub && \
    apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

# System packages for Atari, DMLab, MiniWorld... Throw in everything
RUN apt-get update && apt-get install -y \
    git xvfb \
    libglu1-mesa libglu1-mesa-dev libgl1-mesa-dev libosmesa6-dev mesa-utils freeglut3 freeglut3-dev \
    libglew2.0 libglfw3 libglfw3-dev zlib1g zlib1g-dev libsdl2-dev libjpeg-dev lua5.1 liblua5.1-0-dev libffi-dev \
    build-essential cmake g++-4.8 pkg-config software-properties-common gettext \
    ffmpeg patchelf swig unrar unzip zip curl wget tmux \
    && rm -rf /var/lib/apt/lists/*

# ------------------------
# Standard environments
# ------------------------

# Atari

RUN pip3 install gym==0.19.0 atari-py==0.2.9 opencv-python
RUN wget -L -nv http://www.atarimania.com/roms/Roms.rar && \
    unrar x Roms.rar && \
    python3 -m atari_py.import_roms ROMS && \
    rm -rf Roms.rar ROMS.zip ROMS

# DMC MuJoCo

RUN mkdir -p /root/.mujoco && \
    cd /root/.mujoco && \
    wget -nv https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz -O mujoco.tar.gz && \
    tar -xf mujoco.tar.gz && \
    rm mujoco.tar.gz
RUN pip3 install dm_control

# procgen

RUN pip3 install procgen

# ------------------------
# DMLab (optional)
# ------------------------

# # adapted from https://github.com/google-research/seed_rl
# RUN echo "deb [arch=amd64] http://storage.googleapis.com/bazel-apt stable jdk1.8" | \
#     tee /etc/apt/sources.list.d/bazel.list && \
#     curl https://bazel.build/bazel-release.pub.gpg | \
#     apt-key add - && \
#     apt-get update && apt-get install -y bazel
# RUN git clone https://github.com/deepmind/lab.git /dmlab
# WORKDIR /dmlab
# RUN git checkout "937d53eecf7b46fbfc56c62e8fc2257862b907f2"
# # To check where numpy headers are: python3 -c 'import numpy as np; print(np.get_include())'
# RUN ln -s '/opt/conda/lib/python3.8/site-packages/numpy/core/include/numpy' /usr/include/numpy && \
#     ln -s '/opt/conda/include/python3.8' /usr/include/python3.8 && \
#     sed -i 's@glob(\["include/python3.5/\*.h"\])@glob(["include/numpy/**/*.h", "include/python3.8/**/*.h"\])@g' python.BUILD && \
#     sed -i 's@python3.5@python3.8@g' python.BUILD && \
#     sed -i 's@: \[@: ["include/numpy", @g' python.BUILD && \
#     sed -i 's@650250979303a649e21f87b5ccd02672af1ea6954b911342ea491f351ceb7122@682aee469c3ca857c4c38c37a6edadbfca4b04d42e56613b11590ec6aa4a278d@g' WORKSPACE && \
#     sed -i 's@rules_cc-master@rules_cc-main@g' WORKSPACE && \
#     sed -i 's@rules_cc/archive/master@rules_cc/archive/main@g' WORKSPACE && \
#     sed -i 's@abseil-cpp-master@abseil-cpp-lts_2021_11_02@g' WORKSPACE && \
#     sed -i 's@abseil-cpp/archive/master@abseil-cpp/archive/lts_2021_11_02@g' WORKSPACE
# RUN bazel build -c opt python/pip_package:build_pip_package --incompatible_remove_legacy_whole_archive=0
# RUN pip3 install wheel && \
#     PYTHON_BIN_PATH=$(which python3) && \
#     ./bazel-bin/python/pip_package/build_pip_package /tmp/dmlab_pkg && \
#     pip3 install /tmp/dmlab_pkg/DeepMind_Lab-*.whl --force-reinstall && \
#     rm -rf /dmlab
# WORKDIR /app
# COPY scripts/dmlab_data_download.sh .
# RUN sh dmlab_data_download.sh
# ENV DMLAB_DATASET_PATH "/app/dmlab_data"

# ------------------------
# MineRL (optional)
# ------------------------

# RUN apt-get install -y openjdk-8-jdk libx11-6 x11-xserver-utils
# RUN pip3 install minerl==0.4.4

# ------------------------
# My environments
# ------------------------

# Memory maze
RUN pip3 install memory-maze==1.0.2

# ------------------------
# PyDreamer
# ------------------------

WORKDIR /app

COPY requirements.txt .
RUN pip3 install -r requirements.txt

ENV MLFLOW_TRACKING_URI ""
ENV MLFLOW_EXPERIMENT_NAME "Default"
ENV OMP_NUM_THREADS 1
ENV PYTHONUNBUFFERED 1
ENV LANG "C.UTF-8"

COPY . .

ENTRYPOINT ["sh", "scripts/xvfb_run.sh", "python3", "launch.py"]
