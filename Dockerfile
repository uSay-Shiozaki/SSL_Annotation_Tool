FROM nvidia/cudagl:11.4.2-devel-ubuntu20.04

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
RUN apt-get install -y zlib1g-dev \
  libgstreamer1.0 \
  gstreamer1.0-plugins-base \
  gstreamer1.0-plugins-good \
  mercurial \
  python3-tk \
  xclip

RUN hg clone https://hg.libsdl.org/SDL SDL \
  && cd SDL \
  && mkdir build \
  && cd build \
  && ../configure \
  && make \
  && make install
RUN python3 -m pip install cython kivy \
  https://github.com/kivymd/KivyMD/archive/master.zip \
  && apt-get autoremove -y
COPY ./src/Lib /app/Lib
CMD ["python3", "/app/Lib/main.py"]
