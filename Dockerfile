FROM python:3.11
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt

EXPOSE 8080


CMD exec uvicorn app.server:app --host 0.0.0.0 --port 8080
