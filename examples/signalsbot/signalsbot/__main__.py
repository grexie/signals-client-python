from __future__ import annotations

import asyncio
import os
import signal
from pathlib import Path

from grexie_signals_client import AssetSnapshot, SignalsClient, SignalsManager, SignalsManagerConfig, SignalsManagerState


async def main() -> None:
    load_dotenv(".env")
    token = os.getenv("SIGNALS_WEBSOCKET_TOKEN", "")
    websocket_url = os.getenv("SIGNALS_WEBSOCKET_URL", "wss://signals.grexie.com/ws")
    instruments = [item.strip().upper() for item in os.getenv("SIGNALS_INSTRUMENTS", "BTC-USDT-SWAP").split(",") if item.strip()]
    equity = float(os.getenv("SIGNALS_INITIAL_EQUITY", "1000"))

    client = SignalsClient(token, url=websocket_url)
    await client.connect()

    manager = SignalsManager(
        client,
        SignalsManagerState(assets=[AssetSnapshot("USDT", venue="okx", cash=equity, available=equity, equity=equity)]),
        SignalsManagerConfig(
            venue="okx",
            instruments=instruments,
            risk={"maxMarginRatio": 1, "maxConcurrentPositions": 1, "minLeverage": 1, "maxLeverage": 1},
        ),
    )

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)

    run_task = asyncio.create_task(manager.run())
    print(f"signalsbot listening instruments={','.join(instruments)} ws={websocket_url}")
    try:
        while not stop.is_set():
            done, _ = await asyncio.wait(
                [
                    asyncio.create_task(manager.intents.get()),
                    asyncio.create_task(manager.protection_updates.get()),
                    asyncio.create_task(manager.withdrawals.get()),
                    asyncio.create_task(manager.messages.get()),
                ],
                return_when=asyncio.FIRST_COMPLETED,
            )
            event = done.pop().result()
            if event.type == "create-market-order":
                print(f"intent action={event.action or ''} reason={event.reason or ''} instrument={event.instrument} side={event.side} contracts={event.contract_size} reduce_only={event.reduce_only}")
            elif event.type == "update-tpsl":
                print(f"tpsl instrument={event.instrument} side={event.side} tp={event.take_profit_price} sl={event.stop_loss_price}")
            elif event.type == "withdraw":
                print(f"withdraw currency={event.currency} amount={event.amount}")
            elif event.type == "info":
                print(f'info instrument={event.instrument} stage={event.stage} message="{event.message}"')
    finally:
        run_task.cancel()
        await client.close()


def load_dotenv(path: str) -> None:
    file = Path(path)
    if not file.exists():
        return
    for line in file.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


if __name__ == "__main__":
    asyncio.run(main())
