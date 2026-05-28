from __future__ import annotations

import asyncio
import json
import os
import signal
import sqlite3
import time
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import websockets

from grexie_signals_client import (
    AssetSnapshot,
    ClosedTrade,
    ErrorEvent,
    InfoEvent,
    InstrumentMetadata,
    Order,
    Position,
    PositionManager,
    PositionManagerState,
    ReadyEvent,
    SignalEvent,
    SignalsClient,
    SubscribedEvent,
    UnsubscribedEvent,
    production_position_manager_config,
)

DEFAULT_SIGNALS_WS_URL = "wss://signals.grexie.com/ws"
DEFAULT_OKX_BASE_URL = "https://www.okx.com"
DEFAULT_OKX_WS_URL = "wss://ws.okx.com:8443"
DEFAULT_DB_PATH = "./data/signalsbot.sqlite3"
DEFAULT_EQUITY = 10000.0


async def main() -> None:
    load_dotenv(".env")
    command = "papertrader"
    import sys

    if len(sys.argv) > 1:
        command = sys.argv[1]
    if command == "clean":
        clean_db()
        return
    if command != "papertrader":
        raise SystemExit("usage: signalsbot [papertrader|clean]")

    cfg = load_config()
    store = Store(cfg["db_path"])
    initial_state = store.load_state()

    manager = PositionManager(
        config=production_position_manager_config(initial_state=initial_state, persist=store.save_state)
    )
    bot = SignalsBot(manager, store, cfg["initial_equity"])
    bot.closed_realized = state_closed_realized(initial_state)
    bot.last_closed_count = len(initial_state.closed_trades)
    bot.sync_asset()

    for instrument in cfg["instruments"]:
        metadata = await asyncio.to_thread(fetch_okx_instrument, cfg["okx_base_url"], instrument)
        manager.instruments.update_instrument(metadata)
        tick = await asyncio.to_thread(fetch_latest_candle, cfg["okx_base_url"], cfg["candle_bar"], instrument)
        if tick:
            bot.latest_price_by_key[position_key("okx", instrument)] = tick
            await bot.handle_orders(manager.update_price("okx", instrument, tick["price"], tick["timestamp"]))
        print(
            "Loaded OKX instrument "
            f"instrument={metadata.instrument} settlement={metadata.settlement_currency} "
            f"lot={fmt(metadata.lot_size)} min={fmt(metadata.min_size)} "
            f"tick={fmt(metadata.tick_size)} contract={fmt(metadata.contract_value)}"
        )

    if initial_state.positions or initial_state.closed_trades:
        print(
            f"Hydrated position manager state open_positions={len(initial_state.positions)} "
            f"closed_trades={len(initial_state.closed_trades)}"
        )

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)

    price_task = asyncio.create_task(subscribe_okx_candles(cfg, bot, stop))
    stats_task = asyncio.create_task(report_every(bot, cfg["stats_interval"], stop))

    client = SignalsClient(cfg["token"], url=cfg["websocket_url"])
    await client.connect()
    try:
        for instrument in cfg["instruments"]:
            await client.subscribe("okx", instrument)
            print(f"Subscribed to Grexie Signals venue=okx instrument={instrument}")
        print(
            f"signalsbot running instruments={','.join(cfg['instruments'])} "
            f"db={cfg['db_path']} ws={cfg['websocket_url']}"
        )
        async for event in client.events():
            await bot.handle_signal_event(event)
            if stop.is_set():
                break
    finally:
        price_task.cancel()
        stats_task.cancel()
        await client.close()


class SignalsBot:
    def __init__(self, manager: PositionManager, store: "Store", initial_equity: float) -> None:
        self.manager = manager
        self.store = store
        self.initial_equity = initial_equity
        self.closed_realized = 0.0
        self.last_closed_count = 0
        self.latest_price_by_key: dict[str, dict[str, Any]] = {}
        self.lock = asyncio.Lock()

    async def handle_signal_event(self, event: Any) -> None:
        async with self.lock:
            if isinstance(event, ReadyEvent):
                print(f'Signals websocket ready message="{event.message}"')
                return
            if isinstance(event, InfoEvent):
                print(
                    f'Instrument info instrument={event.instrument} stage={event.stage} '
                    f'replay={event.replay} message="{event.message}"'
                )
                return
            if isinstance(event, ErrorEvent):
                print(f'Signals websocket error code={event.code or ""} message="{event.message or ""}"')
                return
            if isinstance(event, SubscribedEvent):
                print(f"Subscription confirmed subscription={event.subscription_id} instrument={event.instrument}")
                return
            if isinstance(event, UnsubscribedEvent):
                print(
                    f"Subscription removed subscription={event.subscription_id or 0} "
                    f"instrument={event.instrument or ''} code={event.code or ''} message=\"{event.message or ''}\""
                )
                return
            if not isinstance(event, SignalEvent):
                return

            if event.signal.price <= 0:
                tick = self.latest_price_by_key.get(position_key(event.venue or "okx", event.instrument))
                if tick:
                    event.signal.price = tick["price"]
                    event.signal.timestamp = event.signal.timestamp or tick["timestamp"]
            if event.signal.price <= 0:
                print(
                    f"Signal skipped instrument={event.instrument} side={event.signal.side} "
                    f"confidence={fmt(event.signal.confidence)} reason=no OKX candle price yet"
                )
                return

            orders = self.manager.handle_event(event)
            print(
                f"Signal received instrument={event.signal.instrument} side={event.signal.side} "
                f"confidence={fmt(event.signal.confidence)} price={fmt(event.signal.price)} "
                f"replay={event.replay} orders={len(orders)}"
            )
            await self.handle_orders(orders)

    async def handle_price(self, tick: dict[str, Any]) -> None:
        async with self.lock:
            self.latest_price_by_key[position_key("okx", tick["instrument"])] = tick
            await self.handle_orders(self.manager.update_price("okx", tick["instrument"], tick["price"], tick["timestamp"]))

    async def handle_orders(self, orders: Iterable[Order]) -> None:
        orders = list(orders)
        if not orders:
            return
        for order in orders:
            log_order(order)
        trades = self.manager.closed_trades()
        if self.last_closed_count < len(trades):
            new_trades = trades[self.last_closed_count :]
            for trade in new_trades:
                self.closed_realized += trade.realized_pnl
                log_closed_trade(trade, self.initial_equity)
            self.last_closed_count = len(trades)
        self.sync_asset()
        for order in orders:
            self.store.append_order(serialize_dataclass(order))
        self.store.append_snapshot(self.snapshot())

    def sync_asset(self) -> None:
        open_realized = sum(position.realized_pnl for position in self.manager.positions())
        equity = max(self.initial_equity + self.closed_realized + open_realized, 1.0)
        self.manager.assets.update_asset(
            AssetSnapshot(currency="USDT", cash=equity, available=equity, equity=equity, used=0.0)
        )

    def snapshot(self) -> dict[str, Any]:
        stats = self.manager.stats()
        realized = self.closed_realized + stats.realized_pnl
        unrealized = stats.unrealized_pnl
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "equity": self.initial_equity + realized,
            "realizedPnl": realized,
            "unrealizedPnl": unrealized,
            "totalPnl": realized + unrealized,
            "fees": stats.fees,
            "realizedPct": ratio(realized, self.initial_equity),
            "unrealizedPct": ratio(unrealized, self.initial_equity),
            "totalPct": ratio(realized + unrealized, self.initial_equity),
        }

    async def report_stats(self) -> None:
        async with self.lock:
            snapshot = self.snapshot()
            positions = self.manager.positions()
            print(
                "Position manager stats "
                f"equity={money(snapshot['equity'])} realized={money(snapshot['realizedPnl'])} "
                f"unrealized={money(snapshot['unrealizedPnl'])} total={money(snapshot['totalPnl'])} "
                f"fees={money(snapshot['fees'])} open_positions={len(positions)}"
            )
            for position in positions:
                unrealized = position.unrealized_pnl
                print(
                    f"Open position instrument={position.instrument} side={position.side or ''} "
                    f"size={fmt(position.size)} entry={fmt(position.entry_price)} last={fmt(position.last_price)} "
                    f"unrealized={money(unrealized)} pnl={percent(ratio(unrealized, snapshot['equity']))} "
                    f"confidence={fmt(position.confidence)} tp={fmt(position.take_profit)} sl={fmt(position.stop_loss)}"
                )
            self.store.append_snapshot(snapshot)


class Store:
    def __init__(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.execute("create table if not exists manager_state (id integer primary key check (id = 1), data text not null)")
        self.conn.execute("create table if not exists orders (id integer primary key autoincrement, data text not null)")
        self.conn.execute("create table if not exists snapshots (id integer primary key autoincrement, data text not null)")
        self.conn.commit()

    def load_state(self) -> PositionManagerState:
        row = self.conn.execute("select data from manager_state where id = 1").fetchone()
        if row is None:
            return PositionManagerState()
        raw = json.loads(row[0])
        return PositionManagerState(
            positions=[revive_position(item) for item in raw.get("positions", [])],
            closed_trades=[revive_closed_trade(item) for item in raw.get("closed_trades", [])],
        )

    def save_state(self, state: PositionManagerState) -> None:
        data = json.dumps(
            {
                "positions": [serialize_dataclass(position) for position in state.positions],
                "closed_trades": [serialize_dataclass(trade) for trade in state.closed_trades],
            }
        )
        with self.conn:
            self.conn.execute("insert or replace into manager_state (id, data) values (1, ?)", (data,))

    def append_order(self, order: dict[str, Any]) -> None:
        with self.conn:
            self.conn.execute("insert into orders (data) values (?)", (json.dumps(order),))

    def append_snapshot(self, snapshot: dict[str, Any]) -> None:
        with self.conn:
            self.conn.execute("insert into snapshots (data) values (?)", (json.dumps(snapshot),))


async def subscribe_okx_candles(cfg: dict[str, Any], bot: SignalsBot, stop: asyncio.Event) -> None:
    channel = "candle" + cfg["candle_bar"]
    delay = 1.0
    while not stop.is_set():
        try:
            async with websockets.connect(cfg["okx_ws_url"] + "/ws/v5/business") as ws:
                delay = 1.0
                await ws.send(
                    json.dumps(
                        {
                            "op": "subscribe",
                            "args": [{"channel": channel, "instId": instrument} for instrument in cfg["instruments"]],
                        }
                    )
                )
                print(f"Connected OKX candle websocket channel={channel} instruments={','.join(cfg['instruments'])}")
                async for raw in ws:
                    if raw == "ping":
                        await ws.send("pong")
                        continue
                    msg = json.loads(raw)
                    if msg.get("event") == "error" or msg.get("code"):
                        raise RuntimeError(f"okx subscription error {msg.get('code')}: {msg.get('msg')}")
                    for row in msg.get("data") or []:
                        tick = tick_from_okx_candle(msg.get("arg", {}).get("instId", ""), row)
                        if tick:
                            await bot.handle_price(tick)
                    if stop.is_set():
                        return
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            print(f"okx candle websocket: {exc}")
            await asyncio.sleep(delay)
            delay = min(delay * 2, 60.0)


async def report_every(bot: SignalsBot, interval: float, stop: asyncio.Event) -> None:
    while not stop.is_set():
        await asyncio.sleep(interval)
        await bot.report_stats()


def fetch_okx_instrument(base_url: str, instrument: str) -> InstrumentMetadata:
    body = fetch_json(
        base_url + "/api/v5/public/instruments?" + urlencode({"instType": "SWAP", "instId": instrument})
    )
    if body.get("code") != "0" or not body.get("data"):
        raise RuntimeError(f"okx instrument {instrument}: {body.get('code')} {body.get('msg')}")
    row = body["data"][0]
    return InstrumentMetadata(
        venue="okx",
        instrument=row.get("instId", instrument),
        settlement_currency=row.get("settleCcy") or "USDT",
        lot_size=float(row.get("lotSz") or 0),
        min_size=float(row.get("minSz") or 0),
        tick_size=float(row.get("tickSz") or 0),
        contract_value=float(row.get("ctVal") or 0),
        contract_multiplier=float(row.get("ctMult") or 1),
        max_leverage=1.0,
    )


def fetch_latest_candle(base_url: str, bar: str, instrument: str) -> Optional[dict[str, Any]]:
    body = fetch_json(
        base_url + "/api/v5/market/candles?" + urlencode({"instId": instrument, "bar": bar, "limit": "1"})
    )
    if body.get("code") != "0" or not body.get("data"):
        return None
    return tick_from_okx_candle(instrument, body["data"][0])


def fetch_json(url: str) -> dict[str, Any]:
    request = Request(url, headers={"User-Agent": "grexie-signalsbot-python-example/0.1"})
    with urlopen(request, timeout=15) as response:
        return json.loads(response.read().decode())


def tick_from_okx_candle(instrument: str, row: list[str]) -> Optional[dict[str, Any]]:
    if len(row) < 5:
        return None
    price = float(row[4])
    if price <= 0:
        return None
    return {
        "instrument": instrument,
        "price": price,
        "timestamp": datetime.fromtimestamp(int(row[0]) / 1000, tz=timezone.utc),
    }


def log_order(order: Order) -> None:
    action = "Order"
    if abs(order.previous_size) <= 1e-9 and abs(order.target_size) > 1e-9:
        action = "Position Opened"
    elif same_sign(order.previous_size, order.target_size) and abs(order.target_size) > abs(order.previous_size):
        action = "Added margin to position"
    elif same_sign(order.previous_size, order.target_size) and abs(order.target_size) < abs(order.previous_size):
        action = "Removed margin from position"
    elif abs(order.target_size) <= 1e-9 and abs(order.previous_size) > 1e-9:
        action = "Position close order"
    elif not same_sign(order.previous_size, order.target_size):
        action = "Position flip reduction"
    print(
        f"{action} instrument={order.instrument} side={order.side} reason={order.reason} "
        f"delta={fmt(order.size_delta)} previous={fmt(order.previous_size)} target={fmt(order.target_size)} "
        f"price={fmt(order.price)} margin={money(order.margin)} notional={money(order.notional)} "
        f"fee={money(order.estimated_fee_value)} leverage={fmt(order.leverage)} confidence={fmt(order.confidence)} "
        f"expected_edge={fmt(order.expected_edge)} tp={fmt(order.take_profit)} sl={fmt(order.stop_loss)} "
        f"reduce_only={order.reduce_only}"
    )


def log_closed_trade(trade: ClosedTrade, initial_equity: float) -> None:
    print(
        f"Position Closed instrument={trade.instrument} side={trade.side} reason={trade.exit_reason} "
        f"pnl={percent(ratio(trade.realized_pnl, initial_equity))} realized={money(trade.realized_pnl)} "
        f"gross={money(trade.realized_gross)} fees={money(trade.fees)} entry={fmt(trade.entry_price)} "
        f"exit={fmt(trade.exit_price)} size={fmt(trade.size)} move={percent(trade.exit_move)} "
        f"mfe={percent(trade.mfe)} mae={percent(trade.mae)} closed_at={trade.closed_at.isoformat()}"
    )


def serialize_dataclass(value: Any) -> dict[str, Any]:
    raw = asdict(value) if is_dataclass(value) else dict(value)
    return {key: serialize_value(item) for key, item in raw.items()}


def serialize_value(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, list):
        return [serialize_value(item) for item in value]
    if isinstance(value, dict):
        return {key: serialize_value(item) for key, item in value.items()}
    return value


def revive_position(row: dict[str, Any]) -> Position:
    row = dict(row)
    for key in ("opened_at", "last_signal_at"):
        if row.get(key):
            row[key] = datetime.fromisoformat(row[key])
    return Position(**row)


def revive_closed_trade(row: dict[str, Any]) -> ClosedTrade:
    row = dict(row)
    for key in ("closed_at",):
        if row.get(key):
            row[key] = datetime.fromisoformat(row[key])
    return ClosedTrade(**row)


def state_closed_realized(state: PositionManagerState) -> float:
    return sum(trade.realized_pnl for trade in state.closed_trades)


def load_dotenv(path: str) -> None:
    env_path = Path(path)
    if not env_path.exists():
        return
    for raw_line in env_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def load_config() -> dict[str, Any]:
    token = env("SIGNALS_WEBSOCKET_TOKEN", "")
    if not token:
        raise RuntimeError("SIGNALS_WEBSOCKET_TOKEN is required")
    instruments = split_csv(env("SIGNALS_INSTRUMENTS", "DOGE-USDT-SWAP"))
    if not instruments:
        raise RuntimeError("SIGNALS_INSTRUMENTS must contain at least one OKX instrument")
    return {
        "token": token,
        "websocket_url": env("SIGNALS_WEBSOCKET_URL", DEFAULT_SIGNALS_WS_URL),
        "instruments": instruments,
        "db_path": env("SIGNALS_DB_PATH", DEFAULT_DB_PATH),
        "initial_equity": env_float("SIGNALS_INITIAL_EQUITY", DEFAULT_EQUITY),
        "stats_interval": parse_duration_seconds(env("SIGNALS_STATS_INTERVAL", "5m")),
        "okx_base_url": env("SIGNALS_OKX_BASE_URL", DEFAULT_OKX_BASE_URL).rstrip("/"),
        "okx_ws_url": env("SIGNALS_OKX_WEBSOCKET_URL", DEFAULT_OKX_WS_URL).rstrip("/"),
        "candle_bar": env("SIGNALS_OKX_CANDLE_BAR", "1m"),
    }


def clean_db() -> None:
    path = env("SIGNALS_DB_PATH", DEFAULT_DB_PATH)
    try:
        Path(path).unlink()
    except FileNotFoundError:
        pass
    print(f"Cleaned signalsbot local database path={path}")


def env(key: str, fallback: str) -> str:
    return os.environ.get(key, "").strip() or fallback


def env_float(key: str, fallback: float) -> float:
    try:
        value = float(os.environ.get(key, ""))
        return value if value > 0 else fallback
    except ValueError:
        return fallback


def split_csv(value: str) -> list[str]:
    return [item.strip().upper() for item in value.split(",") if item.strip()]


def parse_duration_seconds(value: str) -> float:
    value = value.strip()
    unit = value[-1:] if value[-1:] in {"s", "m", "h"} else "s"
    amount = float(value[:-1] if unit in {"s", "m", "h"} else value)
    return amount * {"s": 1, "m": 60, "h": 3600}[unit]


def position_key(venue: str, instrument: str) -> str:
    return f"{venue.strip().lower()}:{instrument.strip().upper()}"


def same_sign(a: float, b: float) -> bool:
    if abs(a) <= 1e-9 or abs(b) <= 1e-9:
        return True
    return (a < 0) == (b < 0)


def ratio(value: float, basis: float) -> float:
    return value / basis if basis else 0.0


def money(value: float) -> str:
    return f"{value:+.2f} USDT"


def percent(value: float) -> str:
    return f"{value * 100:+.2f}%"


def fmt(value: Optional[float]) -> str:
    return f"{float(value or 0):.8f}"


if __name__ == "__main__":
    asyncio.run(main())
