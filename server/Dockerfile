FROM nvcr.io/nvidia/pytorch:22.07-py3
RUN python3 -m pip install fastapi==0.68.1 uvicorn pydantic==1.8.2 \
    timm munkres 
RUN apt-get -y install gdown
RUN gdown https://drive.google.com/uc?id=1nJwAcx8pDRTHo3eeyGz520DkY_dVP24G
RUN gdown httos://drive.google.com/uc?id=1BSnxAYuAAumpK9pgTRnKoC6UJGeDs_GC
RUN gdown httos://drive.google.com/uc?id=15_K21qGvUvEqbkZKu0NHT2VyOtFgn5RR
COPY ./src /src
WORKDIR /src
# CMD [ "uvicorn", "main:app", "--host", "0.0.0.0" ]