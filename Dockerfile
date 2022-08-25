FROM python:3.10.5-slim-buster

WORKDIR  /app

RUN apt-get update

RUN apt-get install \
    'ffmpeg'\
    'libsm6'\
    'libxext6' -y

COPY . /app

EXPOSE 8080

RUN pip install --upgrade pip

RUN pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu

RUN pip install -r requirements.txt

RUN pip install tensorboard 

RUN pip install python-multipart

CMD ["python", "model_api.py"]