FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1
WORKDIR /app

RUN pip install --upgrade pip \
    && pip install numpy pandas scipy matplotlib seaborn tqdm numba\
    && rm -rf /root/.cache/pip

COPY src /app/src

ENV PYTHONPATH=/app/src
CMD ["python", "/app/src/run_M3.py"]
