ARG PYTHON_VERSION=3.13
FROM python:${PYTHON_VERSION}

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        curl \
        pandoc \
        poppler-utils \
        texlive-fonts-extra \
        texlive-fonts-recommended \
        texlive-latex-base \
        texlive-latex-extra \
        unzip && \
    rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://awscli.amazonaws.com/v2/install.sh \
        -o /tmp/install-aws-cli.sh && \
    bash /tmp/install-aws-cli.sh --system && \
    rm /tmp/install-aws-cli.sh && \
    aws --version

WORKDIR /usr/src/app/mpqp
COPY . .

RUN python -m pip install --upgrade pip && \
    python -m pip install -r requirements-dev.txt && \
    python -m pip install ".[all]"

RUN echo "alias pytest='python -m pytest'" >> /root/.bashrc
