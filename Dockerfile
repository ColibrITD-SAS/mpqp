FROM python:3.9

RUN mkdir -p /usr/src/app/mpqp
WORKDIR /usr/src/app/mpqp

COPY requirements-dev.txt /usr/src/app/
COPY requirements.txt /usr/src/app/

RUN pip3 install --upgrade pip
RUN pip3 install --no-cache-dir -r ../requirements-dev.txt

COPY .. /usr/src/app/mpqp/