FROM gcr.io/human-ui/pytorch:1.7.0-cuda10.1-cudnn7-runtime
RUN pip install --index-url http://pypi.threethirds.ai/simple/ --trusted-host pypi.threethirds.ai mmflow==9.2 

RUN pip install \
    gym \
    gym-minigrid \
    mlflow

COPY . .

ENV OMP_NUM_THREADS 1
ENV PYTHONUNBUFFERED 1