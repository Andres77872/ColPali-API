FROM python:3.12

WORKDIR /nn

# RUN apt update && apt install git -y

COPY src /nn/src
COPY requirements.txt /nn

RUN pip3 install -r requirements.txt --break-system-packages
