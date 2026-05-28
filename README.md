# Grexie Signals Python Client

Typed Python client package for Grexie Signals websocket subscriptions and production-style in-memory position management.

## Grexie Signals - https://signals.grexie.com

Grexie Signals is a real-time crypto trading signal service that streams model-backed market signals with portfolio-aware risk, sizing, and execution context for builders, bots, and trading tools.

```sh
pip install grexie-signals-client
```

## Websocket Client

```python
import asyncio
from grexie_signals_client import SignalsClient, SignalEvent, InfoEvent

async def main():
    client = SignalsClient("ws_your_token")
    await client.connect()
    await client.subscribe("okx", "BTC-USDT-SWAP")

    async for event in client.events():
        if isinstance(event, SignalEvent):
            print(event.signal.instrument, event.signal.side, event.signal.confidence)
        elif isinstance(event, InfoEvent):
            print(event.stage, event.message)

asyncio.run(main())
```

## Position Manager

```python
from grexie_signals_client import (
    InstrumentMetadata,
    PositionManager,
    Signal,
    production_position_manager_config,
)

manager = PositionManager(
    config=production_position_manager_config(
        max_margin_ratio=0.10,
        min_position_size_ratio=0.01,
        min_expected_edge=0.0045,
        min_order_delta=0.20,
        maker_fee_rate=0.0002,
        taker_fee_rate=0.0005,
        max_leverage=3.0,
    )
)
manager.instruments.update_instrument(
    InstrumentMetadata("okx", "BTC-USDT-SWAP", settlement_currency="USDT")
)

orders = manager.handle_signal(
    Signal("okx", "BTC-USDT-SWAP", 0.82, "buy", 0.012, 0.004, price=68000)
)
```

The manager mirrors the server sizing behavior: `max_margin_ratio` is the fraction of `AssetManager` capital that can be allocated as portfolio margin, `min_position_size_ratio` defaults to 1% of capital, positions are signed executable quantities/lots, and emitted orders include quantity, margin, notional, and fee estimates. Portfolio budget is shared by confidence weight, reductions/closes/first-phase flips are emitted before openings or increases, openings are capped by live asset available exposure when asset snapshots are attached, `min_order_delta` scales by the max margin budget, same-side churn can be suppressed, opposite-side flips are allowed, fees feed realized PnL, and leverage is selected from confidence, fee-adjusted edge, and score.

`PositionManager` ignores replay signal events and ignores live signals whose venue/instrument pair has not been configured in its `InstrumentManager`. `SignalsClient.events()` fans out events to independent consumers, so multiple position managers can share one client.

Use `add_position`, `update_position`, `replace_positions`, and `close_position` to hydrate or reconcile the runtime from an exchange account.

## Assets, Instruments, And Stats

`AssetManager` tracks cash, available balance, used margin, and equity by settlement currency. `InstrumentManager` tracks settlement currency, lot size, minimum size, tick size, and max leverage. `PositionManager` uses both to emit concrete quantity, notional, settlement currency, and fee-value estimates.

Call `manager.stats()` for realized and unrealized PnL in account value and percent, grouped by instrument and settlement currency.

## signalsbot Paper Trader Example

The `examples/signalsbot` directory contains a command-line paper trader that reads `.env`, subscribes to `SIGNALS_INSTRUMENTS`, consumes OKX candles, connects with `SIGNALS_WEBSOCKET_TOKEN`, and persists the position manager `initial_state`/`persist` workflow to SQLite.

```sh
cd examples/signalsbot
cp .env.example .env
PYTHONPATH=../../src python3 -m signalsbot papertrader
PYTHONPATH=../../src python3 -m signalsbot clean
docker compose up --build
docker compose run --rm signalsbot clean
```

Set `SIGNALS_WEBSOCKET_URL` to override `wss://signals.grexie.com/ws`. Docker Compose stores the local database in the `signalsbot-data` volume.

## Development

```sh
PYTHONPATH=src python3 -m unittest
```
