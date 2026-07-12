FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-pip curl \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /service
COPY services/inference_api/requirements.txt ./requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt
COPY services/inference_api/app.py ./app.py
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
