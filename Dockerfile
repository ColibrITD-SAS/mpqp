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

COPY mpqp_scripts/awscli_installation/linux_awscli_install.sh /tmp/linux_awscli_install.sh
RUN sed -i 's/\r$//' /tmp/linux_awscli_install.sh && \
    chmod +x /tmp/linux_awscli_install.sh && \
    /tmp/linux_awscli_install.sh && \
    rm /tmp/linux_awscli_install.sh

WORKDIR /usr/src/app/mpqp
COPY . .

RUN python -m pip install --upgrade pip && \
    python -m pip install -r requirements-dev.txt && \
    python -m pip install ".[all]"

RUN echo "alias pytest='python -m pytest'" >> /root/.bashrc
