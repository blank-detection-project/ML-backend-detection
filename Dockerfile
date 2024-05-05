FROM python:3.9

WORKDIR /blank-detection

COPY requirements.txt .
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "src.backend:app", "--host", "0.0.0.0", "--port", "8000"]