FROM python:3.10.5-slim-buster

WORKDIR  /app

COPY . /app


EXPOSE 8080

RUN pip install -r requirements.txt

#pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu 
CMD ["python", "combined.py"]