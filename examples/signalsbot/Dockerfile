FROM python:3.12-slim

WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -e .
WORKDIR /app/examples/signalsbot
RUN mkdir -p /data
ENV SIGNALS_DB_PATH=/data/signalsbot.sqlite3
ENTRYPOINT ["python", "-m", "signalsbot"]
CMD ["papertrader"]
