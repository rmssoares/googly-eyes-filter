FROM python:3.8-buster
MAINTAINER Ricardo Soares "ricardo.ms.soares@hotmail.com"

RUN apt-get update -y && \
    apt-get install -y \
    # Necessary due to dlib. OpenBLAS for linear algebra optimizations which allows dlib functionality to execute faster
    build-essential \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    && apt-get clean && rm -rf /tmp/* /var/tmp/*

ENV PYTHONPATH .

COPY ./requirements.txt /app/requirements.txt

WORKDIR /app
RUN mkdir /app/googly_images
RUN pip install -r requirements.txt

COPY . /app


ENTRYPOINT [ "python" ]

CMD [ "src/server.py" ]