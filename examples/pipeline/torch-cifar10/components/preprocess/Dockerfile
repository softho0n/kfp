FROM pytorch/pytorch:latest

COPY requirements.txt .
RUN python3 -m pip install -r \
    requirements.txt --quiet --no-cache-dir \
    && rm -f requirements.txt

ADD src /workspace
WORKDIR /workspace
