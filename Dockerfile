FROM python:3.9-slim
WORKDIR /app
RUN apt-get update && apt-get install -y python3-tk
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY connect_four_rl.py .
ENV DISPLAY :0
CMD ["python", "connect_four_rl.py"]
