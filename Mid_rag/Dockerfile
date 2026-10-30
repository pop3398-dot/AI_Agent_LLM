Dockerfile


FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# [핵심] uvicorn을 직접 명시하지 말고, main.py 내부의 포트 설정 로직을 타도록 실행합니다.
CMD ["python", "main.py"]
