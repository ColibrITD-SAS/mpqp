ARG PYTHON_VERSION=3.13
ARG AWS_CLI_VERSION=2.39.6

FROM public.ecr.aws/aws-cli/aws-cli:${AWS_CLI_VERSION} AS aws_cli

FROM python:${PYTHON_VERSION}
ARG AWS_CLI_VERSION

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        pandoc \
        poppler-utils \
        texlive-fonts-extra \
        texlive-fonts-recommended \
        texlive-latex-base \
        texlive-latex-extra && \
    rm -rf /var/lib/apt/lists/*

COPY --from=aws_cli /usr/local/aws-cli/ /usr/local/aws-cli/

RUN ln -s /usr/local/aws-cli/v2/current/bin/aws /usr/local/bin/aws && \
    aws --version | grep -F "aws-cli/${AWS_CLI_VERSION} "

WORKDIR /usr/src/app/mpqp
COPY . .

RUN python -m pip install --upgrade pip && \
    python -m pip install -r requirements-dev.txt && \
    python -m pip install ".[all]"

RUN echo "alias pytest='python -m pytest'" >> /root/.bashrc
