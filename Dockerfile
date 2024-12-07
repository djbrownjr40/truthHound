FROM python:3.10-slim

WORKDIR /app

# deps for image processing
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev

COPY requirements.txt /app/

RUN pip install --no-cache-dir -r requirements.txt

COPY . /app/

# expose the flask port
EXPOSE 5000

CMD ["python", "server.py"]