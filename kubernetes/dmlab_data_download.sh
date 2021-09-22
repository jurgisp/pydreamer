mkdir dmlab_data
cd dmlab_data

curl https://bradylab.ucsd.edu/stimuli/ObjectsAll.zip -o ObjectsAll.zip
unzip ObjectsAll.zip

cd OBJECTSALL
python3 << EOM
import os
from PIL import Image
files = [f for f in os.listdir('.') if f.endswith('jpg') or f.endswith('JPG')]
for i, file in enumerate(sorted(files)):
  print(file)
  im = Image.open(file)
  im.save('../%04d.png' % (i+1))
EOM
cd ..

rm -rf __MACOSX OBJECTSALL ObjectsAll.zip
