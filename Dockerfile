# FROM python:3.11
# Install light-weight Python 3.11
FROM python:3.11-slim

# Install some basic utilities
RUN apt-get update \
    && apt-get install -y \
    libatlas-base-dev \
    libc-dev \
    python3-dev \
    build-essential \
    gcc \
    wget

# # Create a user so that we don't run as root
# RUN groupadd -r myuser && useradd -m -r -g myuser myuser
# USER myuser

# Create a working directory
RUN mkdir -p /simplified_clip2brain
WORKDIR /simplified_clip2brain

# Install pip3 and make upgrade
RUN apt-get install python3-pip -y
RUN pip3 install --upgrade pip
RUN pip install --upgrade pip setuptools wheel

# # Switch back to root to clean up after the build process
# USER root
# RUN apt-get clean autoclean && apt-get autoremove --yes && rm -rf /var/lib/apt/lists/*

# # Switch back to non-root user and copy Python code
# USER myuser

# Install Python dependencies
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY . /simplified_clip2brain/

# Default command is a python3 interpreter
CMD ["python3"]
