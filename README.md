# Grexie Signals Python Client

Typed Python client package for Grexie Signals websocket subscriptions and production-style in-memory position management.

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
    PositionManager,
    Signal,
    production_position_manager_config,
)

manager = PositionManager(
    config=production_position_manager_config(
        position_size=0.10,
        min_expected_edge=0.0045,
        min_order_delta=0.20,
        maker_fee_rate=0.0002,
        taker_fee_rate=0.0005,
        max_leverage=3.0,
    )
)

orders = manager.handle_signal(
    Signal("okx", "BTC-USDT-SWAP", 0.82, "buy", 0.012, 0.004, price=68000)
)
```

The manager mirrors the server sizing behavior: total portfolio budget is shared by confidence weight, `min_order_delta` scales by `position_size`, same-side churn can be suppressed, opposite-side flips are allowed, fees feed realized PnL, and leverage is selected from confidence, fee-adjusted edge, and score.

Use `add_position`, `update_position`, and `close_position` to hydrate or mutate the runtime from an exchange account.

## Assets, Instruments, And Stats

`AssetManager` tracks cash, available balance, used margin, and equity by settlement currency. `InstrumentManager` tracks settlement currency, lot size, minimum size, tick size, and max leverage. `PositionManager` uses both to emit concrete quantity, notional, settlement currency, and fee-value estimates.

Call `manager.stats()` for realized and unrealized PnL in account value and percent, grouped by instrument and settlement currency.

## Development

```sh
PYTHONPATH=src python3 -m unittest
```
