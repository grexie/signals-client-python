# Python Signalsbot Example

Paper-trading command line bot for Grexie Signals. It subscribes to `SIGNALS_INSTRUMENTS`, reads OKX candle prices, feeds the Python client `PositionManager`, and persists positions, closed trades, orders, and snapshots in a local SQLite database.

## Run

```sh
cd examples/signalsbot
cp .env.example .env
$EDITOR .env
python -m venv .venv
. .venv/bin/activate
pip install -e ../..
python -m signalsbot papertrader
```

The bot logs position opens, closes with PnL, margin adds/removals, and detailed order sizing. Every five minutes it reports position-manager stats and current PnL.

Clean the local SQLite database with:

```sh
python -m signalsbot clean
```

## Docker

```sh
cd examples/signalsbot
cp .env.example .env
docker compose up --build
```

The compose file stores the SQLite database in the `signalsbot-data` volume.
