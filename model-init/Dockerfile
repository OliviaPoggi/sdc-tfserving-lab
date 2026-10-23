FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir tensorflow-cpu==2.17.0

COPY export_model.py .

CMD ["python", "export_model.py"]
