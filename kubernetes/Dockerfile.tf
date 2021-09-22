FROM tensorflow/tensorflow

RUN pip3 install mlflow google.cloud.storage tfds-nightly Pillow

WORKDIR /app
ENV PYTHONUNBUFFERED 1
COPY . .
