FROM python:3.11
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt

EXPOSE 80


CMD exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-80}
