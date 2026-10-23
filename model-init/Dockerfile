FROM python:3.11-slim

WORKDIR /app

# tensorflow-cpu keeps the image far smaller than the full tensorflow wheel,
# and this container only exports the model, it never serves it.
RUN pip install --no-cache-dir tensorflow-cpu==2.17.0

COPY export_model.py .
COPY models.config .

CMD ["python", "export_model.py"]
