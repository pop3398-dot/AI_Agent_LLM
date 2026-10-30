# 파이썬 공식 슬림 이미지 사용
FROM python:3.10-slim

# 작업 디렉토리 설정
WORKDIR /app

# 시스템 의존성 패키지 설치 (필요한 경우에만 유지)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# requirements.txt 먼저 복사하여 캐시 활용
COPY requirements.txt .

# 패키지 설치 (앞서 언급된 버전 오류가 나지 않도록 유효한 버전만 입력되어야 합니다)
RUN pip install --no-cache-dir -r requirements.txt

# 소스 코드 전체 복사
COPY . .

# [핵심] Cloud Run은 8080 포트를 기본으로 사용하므로 PORT 환경변수를 Uvicorn에 전달합니다.
# main.py 안에서 uvicorn을 실행하고 있다면 CMD ["python", "main.py"]로 설정해도 무방합니다.
CMD exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}
