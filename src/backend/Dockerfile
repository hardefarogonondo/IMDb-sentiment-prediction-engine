FROM python:3.11-slim

ENV PYTHONUNBUFFERED 1

WORKDIR /app

COPY ./src/backend /app
COPY ./models/sentiment_model /app/models/sentiment_model
COPY ./models/tokenizer.pkl /app/models/tokenizer.pkl

RUN pip install --no-cache-dir -r requirements_backend.txt

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]