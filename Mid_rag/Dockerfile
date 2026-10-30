# Python 3.11 슬림 이미지 사용
FROM python:3.11-slim

# 작업 디렉토리 설정
WORKDIR /app

# 시스템 패키지 업데이트 및 필요 라이브러리 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# requirements.txt 복사 및 라이브러리 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 소스 코드 복사
COPY main.py .

# FastAPI 포트 노출 (Cloud Run 기본 포트 8080)
EXPOSE 8080

# uvicorn 서버 실행
# FastAPI 앱을 실행하고 Cloud Run이 전달하는 PORT 환경변수를 필수로 사용하도록 지정
CMD exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}
