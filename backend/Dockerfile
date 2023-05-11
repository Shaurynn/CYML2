FROM python:3.10-slim

WORKDIR /app

# libraries required by OpenCV
RUN apt-get update
RUN apt-get install \
  'ffmpeg'\
  'libsm6'\
  'libxext6'  -y

COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

RUN python fslinstaller.py -d /usr/fsl
ENV FSLDIR=/usr/fsl/
ENV PATH=$PATH:$FSLDIR
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$FSLDIR
ENV FSLOUTPUTTYPE=NIFTI_GZ

COPY . .

# You can add --port $PORT if you need to set PORT as a specific env variable
CMD uvicorn fast_api.api:app --host 0.0.0.0 --port $PORT
