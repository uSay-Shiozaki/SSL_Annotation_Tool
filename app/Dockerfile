FROM ubuntu:20.04

ENV NVIDIA_VISIBLE_DEVICES ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics
ENV TZ=Asia/Tokyo

RUN apt-get update \
  && apt-get install -y \
  python3-pip
RUN apt-get install -y build-essential \
  git \
  python \
  python3.8 \
  python-dev \
  python3.8-dev \
  libffi-dev
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y ffmpeg \
  libsdl2-dev \
  libsdl2-image-dev \
  libsdl2-mixer-dev \
  libsdl2-ttf-dev \
  libportmidi-dev \
  libswscale-dev \
  libavformat-dev \
  libavcodec-dev \
  libmtdev-dev
RUN apt-get update && apt-get install -y zlib1g-dev \
  libgstreamer1.0 \
  gstreamer1.0-plugins-base \
  gstreamer1.0-plugins-good \
  mercurial \
  python3-tk \
  xclip

RUN python3 -m pip install cython kivy==2.1.0 kivymd==1.0.1 python-dotenv\
  && apt-get autoremove -y
COPY ./src/Lib /app/Lib
CMD ["python3", "/app/Lib/main.py"]
