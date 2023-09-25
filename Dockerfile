FROM python:3.11
WORKDIR /simplified_clip2brain
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
COPY . .
