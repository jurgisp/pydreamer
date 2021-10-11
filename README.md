# PyDreamer

Reimplementation of [DreamerV2](https://danijar.com/project/dreamerv2/) model-based RL algorithm in PyTorch. 

The official DreamerV2 implementation [can be found here](https://danijar.com/project/dreamerv2/).


## Features

...


## Running the code

### Running locally

Install dependencies

    pip3 install -r requirements.txt

Get Atari ROMs

    pip3 install atari-py==0.2.9
    wget -L -nv http://www.atarimania.com/roms/Roms.rar
    apt-get install unrar                                   # brew install unar (Mac)
    unrar x Roms.rar                                        # unar -D Roms.rar  (Mac)
    unzip ROMS.zip
    python3 -m atari_py.import_roms ROMS
    rm -rf Roms.rar *ROMS.zip ROMS

Run training (debug CPU mode)

    python pydreamer/train.py --configs defaults atari debug --env_id Atari-Pong

Run training (full GPU mode)

    python pydreamer/train.py --configs defaults atari atari_pong --run_name atari_pong_1

