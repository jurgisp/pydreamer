FROM tensorflow/tensorflow

RUN pip3 install mlflow google.cloud.storage tfds-nightly Pillow pathy

WORKDIR /app
ENV PYTHONUNBUFFERED 1
COPY . .
