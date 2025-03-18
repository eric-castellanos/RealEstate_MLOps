FROM python:3.10-slim

COPY ./ ./

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

CMD ["python", "src/app/app.py"]