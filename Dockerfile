# Stage 1: Build dependencies
FROM python:3.9 AS builder

WORKDIR /usr/src/app

COPY requirements.txt requirements-dev.txt ./
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements-dev.txt

FROM python:3.9

RUN apt update && \
    apt install -y \
        pandoc \
        texlive-latex-base \
        texlive-fonts-recommended \
        texlive-fonts-extra \
        texlive-latex-extra \
        poppler-utils && \
    apt clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /usr/src/app/mpqp

COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
COPY .. /usr/src/app/mpqp/

COPY requirements.txt requirements-dev.txt /usr/src/app/
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r /usr/src/app/requirements.txt && \
    pip install .

RUN echo "alias pytest='python -m pytest'" >> ~/.bashrc