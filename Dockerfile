FROM tensorflow/tensorflow:latest

COPY ./ ./

RUN pip install --no-cache-dir --ignore-installed -r requirements.txt

EXPOSE 8000

CMD ["python", "src/app/app.py"]