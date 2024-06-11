FROM nvcr.io/nvidia/pytorch:22.07-py3
RUN python3 -m pip install fastapi==0.68.1 uvicorn pydantic==1.8.2 \
    timm munkres 
COPY ./src /src
WORKDIR /src
CMD [ "uvicorn", "main:app", "--host", "0.0.0.0" ]