# Python 3.10 슬림 이미지 사용
FROM python:3.10-slim

# 작업 디렉토리 설정
WORKDIR /app

# 시스템 필수 빌드 도구 설치
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 파이썬 의존성 파일 복사 및 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 소스 코드 복사
COPY . .

# 클라우드 런 기본 포트(8080 또는 동적 포트) 개방
EXPOSE 8080

# Uvicorn을 통해 FastAPI 실행 (Cloud Run의 $PORT 환경 변수 동적 수용)
CMD exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}
