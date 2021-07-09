FROM tensorflow/tensorflow:latest

COPY requirements.txt .

RUN python -m pip install -r requirements.txt

WORKDIR /app
COPY . /app