FROM docker.io/library/python:3.10 AS install-base

RUN apt-get update -y

RUN pip install --upgrade pip
RUN pip install gunicorn

FROM install-base AS install-requirements

WORKDIR /app
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
COPY './requirements.txt' .
RUN pip --timeout=1000 install -r requirements.txt

FROM install-requirements AS release

WORKDIR /app
COPY . .

CMD ["python", "app.py"]
EXPOSE 5006
