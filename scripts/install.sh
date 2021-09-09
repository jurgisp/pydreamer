## Install 

brew install cmake zlib

conda create -n pydreamer python=3.8 -y
conda activate pydreamer
pip install -r requirements.txt
pip install pylint==2.4.4 autopep8 jupyter

pip install -e git+https://github.com/jurgisp/gym-minigrid.git#egg=gym-minigrid --src ../
pip install -e git+https://github.com/jurgisp/gym-miniworld.git#egg=gym-miniworld --src ../
