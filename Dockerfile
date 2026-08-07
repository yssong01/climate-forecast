# Tri-CHEF 기후 예보 대시보드 — 서빙 전용 이미지.
#
# 이 이미지에는 학습된 체크포인트가 포함되지 않는다. 학습(train.py)은
# ~40분간 12개 관측소 API 수집 + GPU 학습이 필요해 이미지 빌드에 포함하기
# 부적합하다 — 코드 이미지와 학습 산출물을 분리하고, checkpoints/ 는
# 런타임에 볼륨으로 마운트한다 (아래 실행 예시 참고).
FROM python:3.12-slim

WORKDIR /app

COPY requirements-app.txt .

# CPU 전용 torch — 대시보드 추론(1회 forward pass)은 GPU가 필요 없고,
# 기본 CUDA 휠은 이미지 용량을 수 GB 불필요하게 키운다.
RUN pip install --no-cache-dir torch torchvision \
      --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements-app.txt

COPY *.py ./

EXPOSE 8501

# curl 없이 파이썬으로 헬스체크 — 이미지에 불필요한 패키지를 추가하지 않는다.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

ENTRYPOINT ["streamlit", "run", "app.py", \
            "--server.port=8501", "--server.address=0.0.0.0"]
