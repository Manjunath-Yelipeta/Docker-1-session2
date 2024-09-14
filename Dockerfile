# our base image
FROM  python:3.8.2-slim-buster

# set working directory inside the image
WORKDIR /workspace

# copy our requirements
COPY requirements.txt .

# install dependencies
RUN pip3 install -r requirements.txt && rm -rf /root/.cache/pip

# copy this folder contents to image
COPY . .



