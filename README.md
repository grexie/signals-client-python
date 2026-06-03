# Grexie Signals Python Client

Typed Python client for the Grexie Signals router websocket protocol.

```sh
pip install grexie-signals-client
```

## SignalsManager

`SignalsManager` owns one router basket subscription. It sends your asset and venue-position snapshots to the server, then exposes server-created router intents from the websocket. It does not calculate order management locally.

```python
from grexie_signals_client import (
    AssetSnapshot,
    SignalsClient,
    SignalsManager,
    SignalsManagerConfig,
    SignalsManagerState,
)

client = SignalsClient("ws_your_token", url="wss://signals.grexie.com/ws")
await client.connect()

manager = SignalsManager(
    client,
    SignalsManagerState(
        assets=[AssetSnapshot("USDT", venue="okx", cash=1000, available=1000, equity=1000)]
    ),
    SignalsManagerConfig(
        venue="okx",
        instruments=["BTC-USDT-SWAP", "ETH-USDT-SWAP"],
        risk={"maxMarginRatio": 1, "maxConcurrentPositions": 1, "minLeverage": 1, "maxLeverage": 1},
    ),
)

await manager.subscribe()
```

Router events are available from `manager.intents`, `manager.protection_updates`, `manager.withdrawals`, `manager.backtests`, and `manager.messages`.

## Development

```sh
python -m unittest
```
