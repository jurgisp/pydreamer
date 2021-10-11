FROM pytorch/pytorch:1.8.1-cuda10.2-cudnn7-devel

# System packages for Atari, DMLab, MiniWorld... Throw in everything

RUN apt-get update && apt-get install -y \
    git xvfb \
    libglu1-mesa libglu1-mesa-dev libgl1-mesa-dev libosmesa6-dev mesa-utils freeglut3 freeglut3-dev \
    libglfw3 libglfw3-dev zlib1g zlib1g-dev libsdl2-dev libjpeg-dev lua5.1 liblua5.1-0-dev libffi-dev \
    build-essential cmake g++-4.8 pkg-config software-properties-common gettext \
    ffmpeg patchelf swig unrar unzip zip curl wget tmux \
    && rm -rf /var/lib/apt/lists/*

# Atari

RUN pip3 install atari-py==0.2.9
RUN wget -L -nv http://www.atarimania.com/roms/Roms.rar && \
    unrar x Roms.rar && \
    unzip ROMS.zip && \
    python3 -m atari_py.import_roms ROMS && \
    rm -rf Roms.rar ROMS.zip ROMS

# DMLab (adapted from https://github.com/google-research/seed_rl)

# RUN echo "deb [arch=amd64] http://storage.googleapis.com/bazel-apt stable jdk1.8" | \
#     tee /etc/apt/sources.list.d/bazel.list && \
#     curl https://bazel.build/bazel-release.pub.gpg | \
#     apt-key add - && \
#     apt-get update && apt-get install -y bazel
# RUN git clone https://github.com/deepmind/lab.git
# RUN NP_INC="$(python3 -c 'import numpy as np; print(np.get_include()[5:])')" && \
#     cd lab && \
#     git checkout 937d53eecf7b46fbfc56c62e8fc2257862b907f2 && \
#     sed -i 's@python3.5@python3.6@g' python.BUILD && \
#     sed -i 's@glob(\[@glob(["'"$NP_INC"'/\*\*/*.h", @g' python.BUILD && \
#     sed -i 's@: \[@: ["'"$NP_INC"'", @g' python.BUILD && \
#     sed -i 's@650250979303a649e21f87b5ccd02672af1ea6954b911342ea491f351ceb7122@1e9793e1c6ba66e7e0b6e5fe7fd0f9e935cc697854d5737adec54d93e5b3f730@g' WORKSPACE && \
#     sed -i 's@rules_cc-master@rules_cc-main@g' WORKSPACE && \
#     sed -i 's@rules_cc/archive/master@rules_cc/archive/main@g' WORKSPACE && \
#     bazel build -c opt python/pip_package:build_pip_package --incompatible_remove_legacy_whole_archive=0 && \
#     pip3 install wheel && \
#     PYTHON_BIN_PATH="/usr/bin/python3" ./bazel-bin/python/pip_package/build_pip_package /tmp/dmlab_pkg && \
#     pip3 install /tmp/dmlab_pkg/DeepMind_Lab-*.whl --force-reinstall && \
#     rm -rf /lab
# WORKDIR /app
# COPY scripts/dmlab_data_download.sh .
# RUN sh dmlab_data_download.sh
# ENV DMLAB_DATASET_PATH "/app/dmlab_data"

# MineRL

RUN apt-get update && apt-get install -y \
    openjdk-8-jdk libx11-6 x11-xserver-utils \
    && rm -rf /var/lib/apt/lists/*
RUN pip3 install minerl==0.4.1a2

# APP

WORKDIR /app

COPY requirements.txt .
RUN pip3 install -r requirements.txt

ENV MLFLOW_TRACKING_URI ""
ENV MLFLOW_EXPERIMENT_NAME "Default"
ENV OMP_NUM_THREADS 1
ENV PYTHONUNBUFFERED 1
ENV LANG "C.UTF-8"

COPY . .
