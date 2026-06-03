"""Typed Python client for the Grexie Signals router websocket protocol."""

from __future__ import annotations

import asyncio
import contextlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Dict, List, Literal, Optional, Protocol, Union

SignalsWebSocketToken = str
Side = Literal["buy", "sell"]


@dataclass
class SignalComponent:
    timeframe: str
    side: Side
    confidence: float
    weight: float
    signed_score: float
    take_profit: float
    stop_loss: float
    probability: List[float] = field(default_factory=list)


@dataclass
class Signal:
    venue: str
    instrument: str
    confidence: float
    side: Side
    take_profit: float
    stop_loss: float
    trailing_stop_activation: float = 0.0
    trailing_stop_distance: float = 0.0
    trailing_stop_min_profit: float = 0.0
    timeframe: Optional[str] = None
    score: float = 0.0
    components: List[SignalComponent] = field(default_factory=list)
    model_variant: Optional[str] = None
    model_version: Optional[str] = None
    prediction_mode: Optional[str] = None
    confidence_mapping: Optional[str] = None
    up_probability: float = 0.0
    down_probability: float = 0.0
    directional_edge: float = 0.0
    normalized_edge: float = 0.0
    expected_value: float = 0.0
    regime: Optional[str] = None
    regime_confidence: float = 0.0
    volatility_state: Optional[str] = None
    squeeze_state: Optional[str] = None
    trend_state: Optional[str] = None
    atr_percent: float = 0.0
    signal_ttl: float = 0.0
    generated_at: Optional[datetime] = None
    artifact_id: Optional[str] = None
    artifact_version: Optional[str] = None
    rejected_reason: Optional[str] = None
    manage_positions_only: bool = False
    timestamp: Optional[datetime] = None
    price: float = 0.0


@dataclass
class ReadyEvent:
    type: Literal["ready"]
    message: str


@dataclass
class SubscribedEvent:
    type: Literal["subscribed"]
    subscription_id: int
    venue: str
    instrument: str


@dataclass
class UnsubscribedEvent:
    type: Literal["unsubscribed"]
    subscription_id: Optional[int] = None
    venue: Optional[str] = None
    instrument: Optional[str] = None
    code: Optional[str] = None
    message: Optional[str] = None


@dataclass
class InfoEvent:
    type: Literal["info"]
    subscription_id: int
    venue: str
    instrument: str
    stage: str
    message: str
    timestamp: Optional[datetime] = None
    replay: bool = False
    replayed_at: Optional[datetime] = None


@dataclass
class BacktestEvent:
    type: Literal["backtest"]
    subscription_id: int
    venue: str
    instrument: str
    backtest: Dict[str, Any]
    timestamp: Optional[datetime] = None


@dataclass
class SignalEvent:
    type: Literal["signal"]
    subscription_id: int
    venue: str
    instrument: str
    signal: Signal
    timestamp: Optional[datetime] = None
    replay: bool = False
    replayed_at: Optional[datetime] = None


@dataclass
class CreateMarketOrderEvent:
    type: Literal["create-market-order"]
    subscription_id: int
    intent_id: Optional[str]
    action: Optional[str]
    reason: Optional[str]
    venue: Optional[str]
    instrument: str
    side: str
    order_type: Optional[str] = None
    contract_size: float = 0.0
    leverage: float = 0.0
    reduce_only: bool = False
    take_profit_price: float = 0.0
    stop_loss_price: float = 0.0
    take_profit: float = 0.0
    stop_loss: float = 0.0
    timestamp: Optional[datetime] = None


@dataclass
class UpdateTPSLEvent:
    type: Literal["update-tpsl"]
    subscription_id: int
    intent_id: Optional[str]
    venue: Optional[str]
    instrument: str
    side: str
    take_profit_price: float = 0.0
    stop_loss_price: float = 0.0
    take_profit: float = 0.0
    stop_loss: float = 0.0
    timestamp: Optional[datetime] = None


@dataclass
class WithdrawEvent:
    type: Literal["withdraw"]
    subscription_id: int
    intent_id: Optional[str]
    venue: Optional[str]
    currency: str
    amount: float
    timestamp: Optional[datetime] = None


@dataclass
class ErrorEvent:
    type: Literal["error"]
    code: Optional[str] = None
    message: Optional[str] = None


SignalsEvent = Union[
    ReadyEvent,
    SubscribedEvent,
    UnsubscribedEvent,
    InfoEvent,
    BacktestEvent,
    SignalEvent,
    CreateMarketOrderEvent,
    UpdateTPSLEvent,
    WithdrawEvent,
    ErrorEvent,
]

Intent = CreateMarketOrderEvent


@dataclass
class AssetSnapshot:
    currency: str
    venue: str = ""
    cash: float = 0.0
    available: float = 0.0
    used: float = 0.0
    equity: float = 0.0
    max_usage: float = 1.0
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class Position:
    instrument: str
    size: float
    venue: str = ""
    status: str = ""
    confidence: float = 0.0
    entry_price: float = 0.0
    last_price: float = 0.0
    take_profit: float = 0.0
    stop_loss: float = 0.0
    take_profit_price: float = 0.0
    stop_loss_price: float = 0.0
    trailing_stop_activation: float = 0.0
    trailing_stop_distance: float = 0.0
    trailing_stop_min_profit: float = 0.0
    leverage: float = 1.0
    mfe: float = 0.0
    mae: float = 0.0
    realized_gross: float = 0.0
    fees: float = 0.0
    realized_pnl: float = 0.0
    opened_at: Optional[datetime] = None
    last_signal_at: Optional[datetime] = None

    @property
    def side(self) -> str:
        if self.size < 0:
            return "sell"
        if self.size > 0:
            return "buy"
        return ""

    @property
    def unrealized_pnl(self) -> float:
        if self.entry_price <= 0 or self.last_price <= 0:
            return 0.0
        move = (self.entry_price - self.last_price) / self.entry_price if self.size < 0 else (self.last_price - self.entry_price) / self.entry_price
        return move * abs(self.size) * (self.entry_price or 1.0)


@dataclass
class SignalsManagerState:
    assets: List[AssetSnapshot] = field(default_factory=list)
    positions: List[Position] = field(default_factory=list)


@dataclass
class SignalsManagerConfig:
    venue: str = "okx"
    instruments: List[str] = field(default_factory=list)
    mode: str = ""
    risk: Dict[str, Any] = field(default_factory=dict)
    profit_withdraw_ratio: float = 0.0


class SignalEventSource(Protocol):
    def events(self) -> AsyncIterator[SignalsEvent]:
        ...


class SignalsClient:
    """Authenticated async websocket client for Grexie Signals."""

    def __init__(
        self,
        token: SignalsWebSocketToken,
        *,
        url: str = "wss://signals.grexie.com/ws",
        headers: Optional[Dict[str, str]] = None,
    ) -> None:
        self.token = token
        self.url = url
        self.headers = dict(headers or {})
        self.websocket: Any = None
        self._receive_queue: asyncio.Queue[Any] = asyncio.Queue()
        self._subscriber_queues: List[asyncio.Queue[Any]] = []
        self._reader_task: Optional[asyncio.Task[None]] = None
        self._closed = False
        self._terminal_error: Optional[BaseException] = None

    async def connect(self) -> None:
        import websockets

        headers = dict(self.headers)
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        self.websocket = await websockets.connect(self.url, extra_headers=headers)
        self._closed = False
        self._terminal_error = None
        self._reader_task = asyncio.create_task(self._read_loop())

    async def close(self) -> None:
        if self._reader_task is not None:
            self._reader_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._reader_task
            self._reader_task = None
        if self.websocket is not None:
            await self.websocket.close()
            self.websocket = None

    async def subscribe(self, venue: str, instrument: str) -> None:
        await self._send({"type": "subscribe", "venue": venue, "instrument": instrument})

    async def subscribe_basket(
        self,
        *,
        venue: str,
        instruments: List[str],
        mode: str = "",
        risk: Optional[Dict[str, Any]] = None,
        profit_withdraw_ratio: float = 0.0,
        assets: Optional[List[AssetSnapshot]] = None,
        positions: Optional[List[Position]] = None,
    ) -> None:
        payload: Dict[str, Any] = {"type": "subscribe", "venue": venue, "instruments": instruments}
        if mode:
            payload["mode"] = mode
        if risk:
            payload["risk"] = risk
        if profit_withdraw_ratio > 0:
            payload["profitWithdrawRatio"] = profit_withdraw_ratio
        if assets:
            payload["assets"] = [_asset_payload(asset) for asset in assets]
        if positions:
            payload["positions"] = [_position_payload(position) for position in positions]
        await self._send(payload)

    async def update_asset(self, subscription_id: int, asset: AssetSnapshot) -> None:
        await self._send({"type": "update-asset", "subscriptionId": subscription_id, **_asset_payload(asset)})

    async def update_position(self, subscription_id: int, position: Position) -> None:
        await self._send({"type": "update-position", "subscriptionId": subscription_id, **_position_payload(position)})

    async def add_instrument(self, subscription_id: int, instrument: str) -> None:
        await self._send({"type": "add-instrument", "subscriptionId": subscription_id, "instrument": instrument})

    async def remove_instrument(self, subscription_id: int, instrument: str) -> None:
        await self._send({"type": "remove-instrument", "subscriptionId": subscription_id, "instrument": instrument})

    async def update_config(self, subscription_id: int, *, profit_withdraw_ratio: float = 0.0) -> None:
        await self._send({"type": "update-config", "subscriptionId": subscription_id, "profitWithdrawRatio": profit_withdraw_ratio})

    async def schedule_withdrawal(self, subscription_id: int, *, currency: str, amount: float, venue: str = "", reason: str = "") -> None:
        await self._send({"type": "schedule-withdrawal", "subscriptionId": subscription_id, "venue": venue, "currency": currency, "amount": amount, "reason": reason})

    async def unsubscribe(self, subscription_id: int) -> None:
        await self._send({"type": "unsubscribe", "subscriptionId": subscription_id})

    async def unsubscribe_instrument(self, venue: str, instrument: str) -> None:
        await self._send({"type": "unsubscribe", "venue": venue, "instrument": instrument})

    async def receive(self) -> SignalsEvent:
        if self.websocket is None and self._reader_task is None:
            raise RuntimeError("signals client is not connected")
        item = await self._receive_queue.get()
        if item is None:
            raise RuntimeError("signals client is closed")
        if isinstance(item, BaseException):
            raise item
        return item

    async def events(self) -> AsyncIterator[SignalsEvent]:
        if self._terminal_error is not None:
            raise self._terminal_error
        if self._closed:
            return
        queue: asyncio.Queue[Any] = asyncio.Queue()
        self._subscriber_queues.append(queue)
        try:
            while True:
                item = await queue.get()
                if item is None:
                    return
                if isinstance(item, BaseException):
                    raise item
                yield item
        finally:
            with contextlib.suppress(ValueError):
                self._subscriber_queues.remove(queue)

    async def _send(self, payload: Dict[str, Any]) -> None:
        if self.websocket is None:
            raise RuntimeError("signals client is not connected")
        await self.websocket.send(json.dumps(payload, separators=(",", ":"), default=_json_default))

    async def _read_loop(self) -> None:
        try:
            while self.websocket is not None:
                await self._publish(parse_event(await self.websocket.recv()))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._terminal_error = exc
            await self._publish(exc)
        finally:
            self._closed = True
            await self._publish(None)

    async def _publish(self, item: Any) -> None:
        await self._receive_queue.put(item)
        for queue in list(self._subscriber_queues):
            await queue.put(item)


class SignalsManager:
    """Owns one router basket subscription and forwards server-created intents."""

    def __init__(self, client: SignalsClient, state: Optional[SignalsManagerState] = None, config: Optional[SignalsManagerConfig] = None) -> None:
        self.client = client
        self.config = _normalize_manager_config(config or SignalsManagerConfig())
        self.subscription_id = 0
        self._assets: Dict[str, AssetSnapshot] = {}
        self._positions: Dict[str, Position] = {}
        self.intents: asyncio.Queue[Intent] = asyncio.Queue()
        self.protection_updates: asyncio.Queue[UpdateTPSLEvent] = asyncio.Queue()
        self.withdrawals: asyncio.Queue[WithdrawEvent] = asyncio.Queue()
        self.backtests: asyncio.Queue[BacktestEvent] = asyncio.Queue()
        self.messages: asyncio.Queue[InfoEvent] = asyncio.Queue()
        self.events_queue: asyncio.Queue[SignalsEvent] = asyncio.Queue()
        for asset in (state or SignalsManagerState()).assets:
            self._record_asset(asset)
        for position in (state or SignalsManagerState()).positions:
            self._record_position(position)

    async def run(self) -> None:
        await self.subscribe()
        try:
            async for event in self.client.events():
                await self.handle_event(event)
        finally:
            if self.subscription_id > 0:
                await self.client.unsubscribe(self.subscription_id)
                self.subscription_id = 0

    async def subscribe(self) -> None:
        await self.client.subscribe_basket(
            venue=self.config.venue,
            instruments=list(self.config.instruments),
            mode=self.config.mode,
            risk=self.config.risk,
            profit_withdraw_ratio=self.config.profit_withdraw_ratio,
            assets=self.assets(),
            positions=self.positions(),
        )

    async def update_asset(self, asset: AssetSnapshot) -> None:
        next_asset = self._record_asset(asset)
        if next_asset is not None and self.subscription_id > 0:
            await self.client.update_asset(self.subscription_id, next_asset)

    async def update_position(self, position: Position) -> None:
        next_position = self._record_position(position)
        if next_position is not None and self.subscription_id > 0:
            await self.client.update_position(self.subscription_id, next_position)

    async def add_instrument(self, instrument: str) -> None:
        instrument = _normalize_instrument(instrument)
        if not instrument:
            return
        self.config.instruments = _normalize_instrument_list([*self.config.instruments, instrument])
        if self.subscription_id > 0:
            await self.client.add_instrument(self.subscription_id, instrument)

    async def remove_instrument(self, instrument: str) -> None:
        instrument = _normalize_instrument(instrument)
        self.config.instruments = [current for current in self.config.instruments if current != instrument]
        if self.subscription_id > 0:
            await self.client.remove_instrument(self.subscription_id, instrument)

    async def update_config(self, *, profit_withdraw_ratio: float = 0.0) -> None:
        self.config.profit_withdraw_ratio = _clamp01(profit_withdraw_ratio)
        if self.subscription_id > 0:
            await self.client.update_config(self.subscription_id, profit_withdraw_ratio=self.config.profit_withdraw_ratio)

    async def schedule_withdrawal(self, *, currency: str, amount: float, venue: str = "", reason: str = "") -> None:
        if self.subscription_id <= 0:
            raise RuntimeError("signals manager basket is not subscribed")
        await self.client.schedule_withdrawal(self.subscription_id, currency=currency.upper(), amount=amount, venue=_normalize_venue(venue or self.config.venue), reason=reason)

    async def handle_event(self, event: SignalsEvent) -> None:
        if not self._accepts_event(event):
            return
        if isinstance(event, SubscribedEvent) and event.subscription_id > 0:
            self.subscription_id = event.subscription_id
            for asset in self.assets():
                await self.client.update_asset(self.subscription_id, asset)
            for position in self.positions():
                await self.client.update_position(self.subscription_id, position)
        elif isinstance(event, UnsubscribedEvent) and event.subscription_id == self.subscription_id:
            self.subscription_id = 0
        elif isinstance(event, CreateMarketOrderEvent):
            await self.intents.put(event)
        elif isinstance(event, UpdateTPSLEvent):
            self._apply_tpsl(event)
            await self.protection_updates.put(event)
        elif isinstance(event, WithdrawEvent):
            await self.withdrawals.put(event)
        elif isinstance(event, BacktestEvent):
            await self.backtests.put(event)
        elif isinstance(event, InfoEvent):
            await self.messages.put(event)
        await self.events_queue.put(event)

    def assets(self) -> List[AssetSnapshot]:
        return sorted(self._assets.values(), key=lambda asset: asset.currency)

    def positions(self) -> List[Position]:
        return sorted(self._positions.values(), key=lambda position: _position_key(position.venue, position.instrument))

    def state(self) -> SignalsManagerState:
        return SignalsManagerState(self.assets(), self.positions())

    def available_order_cash(self, currency: str) -> float:
        asset = self._assets.get(currency.upper())
        return max(asset.available if asset else 0.0, 0.0) * _clamp01(_positive_or(asset.max_usage if asset else 0.0, 1.0))

    def _record_asset(self, asset: AssetSnapshot) -> Optional[AssetSnapshot]:
        currency = asset.currency.strip().upper()
        if not currency:
            return None
        asset.venue = _normalize_venue(asset.venue or self.config.venue)
        asset.currency = currency
        asset.max_usage = _clamp01(_positive_or(asset.max_usage, 1.0))
        self._assets[currency] = asset
        return asset

    def _record_position(self, position: Position) -> Optional[Position]:
        instrument = _normalize_instrument(position.instrument)
        if not instrument:
            return None
        position.venue = _normalize_venue(position.venue or self.config.venue)
        position.instrument = instrument
        position.status = (position.status or ("open" if abs(position.size) > 1e-9 else "closed")).lower()
        if position.last_price <= 0:
            position.last_price = position.entry_price
        key = _position_key(position.venue, position.instrument)
        if position.status == "closed" or abs(position.size) <= 1e-9:
            self._positions.pop(key, None)
        else:
            self._positions[key] = position
        return position

    def _apply_tpsl(self, event: UpdateTPSLEvent) -> None:
        position = self._positions.get(_position_key(event.venue or self.config.venue, event.instrument))
        if position is None:
            return
        position.take_profit = event.take_profit or position.take_profit
        position.stop_loss = event.stop_loss or position.stop_loss
        position.take_profit_price = event.take_profit_price or position.take_profit_price
        position.stop_loss_price = event.stop_loss_price or position.stop_loss_price

    def _accepts_event(self, event: SignalsEvent) -> bool:
        subscription_id = getattr(event, "subscription_id", 0) or 0
        if self.subscription_id > 0 and subscription_id > 0:
            return subscription_id == self.subscription_id
        venue = getattr(event, "venue", "")
        instrument = getattr(event, "instrument", "")
        if isinstance(event, SubscribedEvent):
            return _normalize_venue(venue) == self.config.venue and (not instrument or _normalize_instrument(instrument) in self.config.instruments)
        if venue and instrument:
            return _normalize_venue(venue) == self.config.venue and _normalize_instrument(instrument) in self.config.instruments
        return True


def parse_event(raw: Union[str, bytes, Dict[str, Any]]) -> SignalsEvent:
    msg = json.loads(raw.decode() if isinstance(raw, bytes) else raw) if not isinstance(raw, dict) else raw
    event_type = msg.get("type")
    if event_type == "ready":
        return ReadyEvent("ready", msg.get("message", ""))
    if event_type == "subscribed":
        return SubscribedEvent("subscribed", int(msg.get("subscriptionId", 0)), msg.get("venue", ""), msg.get("instrument", ""))
    if event_type == "unsubscribed":
        return UnsubscribedEvent("unsubscribed", msg.get("subscriptionId"), msg.get("venue"), msg.get("instrument"), msg.get("code"), msg.get("message"))
    if event_type == "info":
        return InfoEvent("info", int(msg.get("subscriptionId", 0)), msg.get("venue", ""), msg.get("instrument", ""), msg.get("stage", ""), msg.get("message", ""), _parse_time(msg.get("timestamp")), bool(msg.get("replay", False)), _parse_time(msg.get("replayedAt")))
    if event_type == "backtest":
        payload = msg.get("backtest")
        return BacktestEvent("backtest", int(msg.get("subscriptionId", 0)), msg.get("venue", ""), msg.get("instrument", ""), dict(payload or {}) if isinstance(payload, dict) else {}, _parse_time(msg.get("timestamp")))
    if event_type == "signal":
        payload = dict(msg.get("signal") or {})
        payload.setdefault("venue", msg.get("venue", ""))
        payload.setdefault("instrument", msg.get("instrument", ""))
        payload.setdefault("timestamp", msg.get("timestamp"))
        signal = _signal_from_payload(payload)
        return SignalEvent("signal", int(msg.get("subscriptionId", 0)), msg.get("venue", signal.venue), msg.get("instrument", signal.instrument), signal, _parse_time(msg.get("timestamp")), bool(msg.get("replay", False)), _parse_time(msg.get("replayedAt")))
    if event_type == "create-market-order":
        return CreateMarketOrderEvent("create-market-order", int(msg.get("subscriptionId", 0)), msg.get("intentId"), msg.get("action"), msg.get("reason"), msg.get("venue"), msg.get("instrument", ""), msg.get("side", ""), msg.get("orderType"), float(msg.get("contractSize", 0.0) or 0.0), float(msg.get("leverage", 0.0) or 0.0), bool(msg.get("reduceOnly", False)), float(msg.get("takeProfitPrice", 0.0) or 0.0), float(msg.get("stopLossPrice", 0.0) or 0.0), float(msg.get("takeProfit", 0.0) or 0.0), float(msg.get("stopLoss", 0.0) or 0.0), _parse_time(msg.get("timestamp")))
    if event_type == "update-tpsl":
        return UpdateTPSLEvent("update-tpsl", int(msg.get("subscriptionId", 0)), msg.get("intentId"), msg.get("venue"), msg.get("instrument", ""), msg.get("side", ""), float(msg.get("takeProfitPrice", 0.0) or 0.0), float(msg.get("stopLossPrice", 0.0) or 0.0), float(msg.get("takeProfit", 0.0) or 0.0), float(msg.get("stopLoss", 0.0) or 0.0), _parse_time(msg.get("timestamp")))
    if event_type == "withdraw":
        return WithdrawEvent("withdraw", int(msg.get("subscriptionId", 0)), msg.get("intentId"), msg.get("venue"), msg.get("currency", ""), float(msg.get("amount", 0.0) or 0.0), _parse_time(msg.get("timestamp")))
    if event_type == "error":
        return ErrorEvent("error", msg.get("code"), msg.get("message"))
    raise ValueError(f"unsupported websocket event type {event_type!r}")


def _signal_from_payload(payload: Dict[str, Any]) -> Signal:
    return Signal(
        venue=payload.get("venue", ""),
        instrument=payload.get("instrument", ""),
        confidence=float(payload.get("confidence", 0.0) or 0.0),
        side=payload.get("side", "buy"),
        take_profit=float(payload.get("takeProfit", payload.get("take_profit", 0.0)) or 0.0),
        stop_loss=float(payload.get("stopLoss", payload.get("stop_loss", 0.0)) or 0.0),
        trailing_stop_activation=float(payload.get("trailingStopActivation", 0.0) or 0.0),
        trailing_stop_distance=float(payload.get("trailingStopDistance", 0.0) or 0.0),
        trailing_stop_min_profit=float(payload.get("trailingStopMinProfit", 0.0) or 0.0),
        timeframe=payload.get("timeframe"),
        score=float(payload.get("score", 0.0) or 0.0),
        manage_positions_only=bool(payload.get("managePositionsOnly", False)),
        timestamp=_parse_time(payload.get("timestamp")),
        price=float(payload.get("price", 0.0) or 0.0),
    )


def _asset_payload(asset: AssetSnapshot) -> Dict[str, Any]:
    return {
        "venue": asset.venue,
        "currency": asset.currency,
        "cash": asset.cash,
        "available": asset.available,
        "used": asset.used,
        "equity": asset.equity,
        "maxUsage": asset.max_usage,
        "updatedAt": asset.updated_at.isoformat(),
    }


def _position_payload(position: Position) -> Dict[str, Any]:
    return {
        "venue": position.venue,
        "instrument": position.instrument,
        "side": position.side,
        "status": position.status,
        "size": abs(position.size),
        "entryPrice": position.entry_price,
        "markPrice": position.last_price,
        "leverage": position.leverage,
        "takeProfitPrice": position.take_profit_price,
        "stopLossPrice": position.stop_loss_price,
    }


def _normalize_manager_config(config: SignalsManagerConfig) -> SignalsManagerConfig:
    config.venue = _normalize_venue(config.venue)
    config.instruments = _normalize_instrument_list(config.instruments)
    config.profit_withdraw_ratio = _clamp01(config.profit_withdraw_ratio)
    return config


def _normalize_venue(venue: str) -> str:
    return (venue or "okx").strip().lower()


def _normalize_instrument(instrument: str) -> str:
    return (instrument or "").strip().upper()


def _normalize_instrument_list(instruments: List[str]) -> List[str]:
    return sorted({normalized for normalized in (_normalize_instrument(item) for item in instruments) if normalized})


def _position_key(venue: str, instrument: str) -> str:
    return f"{_normalize_venue(venue)}:{_normalize_instrument(instrument)}"


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value if value == value else 0.0))


def _positive_or(*values: float) -> float:
    for value in values:
        if value > 0:
            return value
    return 0.0


def _parse_time(value: Any) -> Optional[datetime]:
    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


def _json_default(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


__all__ = [
    "AssetSnapshot",
    "BacktestEvent",
    "CreateMarketOrderEvent",
    "ErrorEvent",
    "InfoEvent",
    "Intent",
    "Position",
    "ReadyEvent",
    "Signal",
    "SignalComponent",
    "SignalEvent",
    "SignalEventSource",
    "SignalsClient",
    "SignalsEvent",
    "SignalsManager",
    "SignalsManagerConfig",
    "SignalsManagerState",
    "SignalsWebSocketToken",
    "Side",
    "SubscribedEvent",
    "UnsubscribedEvent",
    "UpdateTPSLEvent",
    "WithdrawEvent",
    "parse_event",
]
