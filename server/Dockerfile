FROM nvcr.io/nvidia/pytorch:23.10-py3
RUN apt remove -y python3-numpy python3-scipy python3-sklearn || true
RUN pip uninstall -y numpy scipy scikit-learn
RUN pip install --upgrade pip
RUN pip install fastapi==0.68.1 uvicorn pydantic==1.8.2 timm munkres torchinfo
RUN pip install --no-cache-dir "numpy<2.0" "scipy<1.12" "scikit-learn<1.4"
COPY ./src /src
WORKDIR /src
CMD [ "uvicorn", "main:app", "--host", "0.0.0.0" ]
