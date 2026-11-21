FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY model_pipeline.py main.py heart.csv ./

CMD ["python", "main.py", "--step", "all"]
