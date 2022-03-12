FROM pytorch/pytorch

RUN apt-get update && apt-get install -y git gcc

COPY requirements.txt requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

RUN apt-get install -y libglib2.0-0 libsm6 libxrender1 libxext6