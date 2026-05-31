"""Typed Python client for Grexie Signals.

The package exposes two primary objects:

* :class:`SignalsClient`, an async websocket client authenticated by a
  ``SignalsWebSocketToken``.
* :class:`PositionManager`, an in-memory, fee-aware position manager that
  follows the production Grexie Signals sizing model.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import math
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from typing import Any, AsyncIterator, Callable, Dict, Iterable, List, Literal, Optional, Protocol, Union

SignalsWebSocketToken = str
Side = Literal["buy", "sell"]


@dataclass
class SignalComponent:
    """One timeframe contribution to an aggregate signal."""

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
    """Public signal payload sent by the Grexie Signals websocket."""

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
    SignalEvent,
    CreateMarketOrderEvent,
    UpdateTPSLEvent,
    WithdrawEvent,
    ErrorEvent,
]


def parse_event(raw: Union[str, bytes, Dict[str, Any]]) -> SignalsEvent:
    """Parse one websocket JSON message into a typed event dataclass."""

    msg = json.loads(raw.decode() if isinstance(raw, bytes) else raw) if not isinstance(raw, dict) else raw
    event_type = msg.get("type")
    if event_type == "ready":
        return ReadyEvent("ready", msg.get("message", ""))
    if event_type == "subscribed":
        return SubscribedEvent("subscribed", int(msg.get("subscriptionId", 0)), msg.get("venue", ""), msg.get("instrument", ""))
    if event_type == "unsubscribed":
        return UnsubscribedEvent(
            "unsubscribed",
            msg.get("subscriptionId"),
            msg.get("venue"),
            msg.get("instrument"),
            msg.get("code"),
            msg.get("message"),
        )
    if event_type == "info":
        return InfoEvent(
            "info",
            int(msg.get("subscriptionId", 0)),
            msg.get("venue", ""),
            msg.get("instrument", ""),
            msg.get("stage", ""),
            msg.get("message", ""),
            _parse_time(msg.get("timestamp")),
            bool(msg.get("replay", False)),
            _parse_time(msg.get("replayedAt")),
        )
    if event_type == "signal":
        payload = dict(msg.get("signal") or {})
        payload.setdefault("venue", msg.get("venue", ""))
        payload.setdefault("instrument", msg.get("instrument", ""))
        payload.setdefault("timestamp", msg.get("timestamp"))
        signal = _signal_from_payload(payload)
        return SignalEvent(
            "signal",
            int(msg.get("subscriptionId", 0)),
            msg.get("venue", signal.venue),
            msg.get("instrument", signal.instrument),
            signal,
            _parse_time(msg.get("timestamp")),
            bool(msg.get("replay", False)),
            _parse_time(msg.get("replayedAt")),
        )
    if event_type == "create-market-order":
        return CreateMarketOrderEvent(
            "create-market-order",
            int(msg.get("subscriptionId", 0)),
            msg.get("intentId"),
            msg.get("action"),
            msg.get("venue"),
            msg.get("instrument", ""),
            msg.get("side", ""),
            msg.get("orderType"),
            float(msg.get("contractSize", 0.0) or 0.0),
            float(msg.get("leverage", 0.0) or 0.0),
            bool(msg.get("reduceOnly", False)),
            float(msg.get("takeProfitPrice", 0.0) or 0.0),
            float(msg.get("stopLossPrice", 0.0) or 0.0),
            float(msg.get("takeProfit", 0.0) or 0.0),
            float(msg.get("stopLoss", 0.0) or 0.0),
            _parse_time(msg.get("timestamp")),
        )
    if event_type == "update-tpsl":
        return UpdateTPSLEvent(
            "update-tpsl",
            int(msg.get("subscriptionId", 0)),
            msg.get("intentId"),
            msg.get("venue"),
            msg.get("instrument", ""),
            msg.get("side", ""),
            float(msg.get("takeProfitPrice", msg.get("take_profit_price", 0.0)) or 0.0),
            float(msg.get("stopLossPrice", msg.get("stop_loss_price", 0.0)) or 0.0),
            float(msg.get("takeProfit", msg.get("take_profit", 0.0)) or 0.0),
            float(msg.get("stopLoss", msg.get("stop_loss", 0.0)) or 0.0),
            _parse_time(msg.get("timestamp")),
        )
    if event_type == "withdraw":
        return WithdrawEvent(
            "withdraw",
            int(msg.get("subscriptionId", 0)),
            msg.get("intentId"),
            msg.get("venue"),
            msg.get("currency", ""),
            float(msg.get("amount", 0.0) or 0.0),
            _parse_time(msg.get("timestamp")),
        )
    if event_type == "error":
        return ErrorEvent("error", msg.get("code"), msg.get("message"))
    raise ValueError(f"unsupported websocket event type {event_type!r}")


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
        """Open the websocket connection."""

        import websockets

        headers = dict(self.headers)
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        self.websocket = await websockets.connect(self.url, extra_headers=headers)
        self._closed = False
        self._terminal_error = None
        self._reader_task = asyncio.create_task(self._read_loop())

    async def close(self) -> None:
        """Close the websocket connection."""

        if self._reader_task is not None:
            self._reader_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._reader_task
            self._reader_task = None
        if self.websocket is not None:
            await self.websocket.close()
            self.websocket = None

    async def subscribe(self, venue: str, instrument: str) -> None:
        """Subscribe to one venue/instrument pair."""

        await self._send({"type": "subscribe", "venue": venue, "instrument": instrument})

    async def subscribe_basket(
        self,
        *,
        venue: str,
        instruments: List[str],
        mode: str = "",
        risk: Optional[Dict[str, Any]] = None,
        profit_withdraw_ratio: float = 0.0,
        assets: Optional[List["AssetSnapshot"]] = None,
        positions: Optional[List["Position"]] = None,
    ) -> None:
        """Subscribe to one Bollinger-router basket."""

        payload: Dict[str, Any] = {
            "type": "subscribe",
            "venue": venue,
            "instruments": instruments,
        }
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

    async def update_asset(self, subscription_id: int, asset: "AssetSnapshot") -> None:
        """Publish an asset/currency snapshot for an active basket."""

        await self._send({"type": "update-asset", "subscriptionId": subscription_id, **_asset_payload(asset)})

    async def update_position(self, subscription_id: int, position: "Position") -> None:
        """Publish a venue position snapshot for an active basket."""

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
        """Unsubscribe by server subscription id."""

        await self._send({"type": "unsubscribe", "subscriptionId": subscription_id})

    async def unsubscribe_instrument(self, venue: str, instrument: str) -> None:
        """Unsubscribe by venue/instrument pair."""

        await self._send({"type": "unsubscribe", "venue": venue, "instrument": instrument})

    async def receive(self) -> SignalsEvent:
        """Receive the next typed event."""

        if self.websocket is None and self._reader_task is None:
            raise RuntimeError("signals client is not connected")
        if self._closed and self._receive_queue.empty():
            raise RuntimeError("signals client is closed")
        item = await self._receive_queue.get()
        if item is None:
            raise RuntimeError("signals client is closed")
        if isinstance(item, BaseException):
            raise item
        return item

    async def events(self) -> AsyncIterator[SignalsEvent]:
        """Yield events until the websocket closes."""

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
        await self.websocket.send(json.dumps(payload, separators=(",", ":")))

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


class SignalEventSource(Protocol):
    """Narrow async event stream consumed by PositionManager."""

    def events(self) -> AsyncIterator[SignalsEvent]:
        ...


@dataclass
class InstrumentConfig:
    maker_fee_rate: Optional[float] = None
    taker_fee_rate: Optional[float] = None
    min_leverage: Optional[float] = None
    max_leverage: Optional[float] = None
    trailing_stop_activation: Optional[float] = None
    trailing_stop_distance: Optional[float] = None
    trailing_stop_min_profit: Optional[float] = None


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


class AssetManager:
    """Tracks cash, available, used, and equity by settlement currency."""

    def __init__(self) -> None:
        self._assets: Dict[str, AssetSnapshot] = {}

    def update_asset(self, snapshot: AssetSnapshot) -> None:
        if snapshot.currency:
            self._assets[snapshot.currency] = snapshot

    def asset(self, currency: str) -> Optional[AssetSnapshot]:
        return self._assets.get(currency)

    def assets(self) -> List[AssetSnapshot]:
        return [self._assets[key] for key in sorted(self._assets)]


@dataclass
class InstrumentMetadata:
    venue: str
    instrument: str
    settlement_currency: str = "USDT"
    lot_size: float = 0.0
    min_size: float = 0.0
    tick_size: float = 0.0
    contract_value: float = 0.0
    contract_multiplier: float = 0.0
    max_leverage: float = 0.0


class InstrumentManager:
    """Tracks lot size, min size, tick size, settlement currency, and leverage caps."""

    def __init__(self) -> None:
        self._instruments: Dict[str, InstrumentMetadata] = {}

    def update_instrument(self, metadata: InstrumentMetadata) -> None:
        if metadata.venue and metadata.instrument:
            self._instruments[_position_key(metadata.venue, metadata.instrument)] = metadata

    def remove_instrument(self, venue: str, instrument: str) -> None:
        if venue and instrument:
            self._instruments.pop(_position_key(venue, instrument), None)

    def instrument(self, venue: str, instrument: str) -> InstrumentMetadata:
        return self._instruments.get(_position_key(venue, instrument), InstrumentMetadata(venue, instrument))

    def has_instrument(self, venue: str, instrument: str) -> bool:
        return _position_key(venue, instrument) in self._instruments

    def instruments(self) -> List[InstrumentMetadata]:
        return [self._instruments[key] for key in sorted(self._instruments)]


@dataclass
class PositionManagerConfig:
    max_margin_ratio: float = 1.0
    position_size: float = 0.0
    min_expected_edge: float = 0.0045
    min_order_delta: float = 0.20
    min_position_size_ratio: float = 0.01
    rebalance_interval: timedelta = timedelta(hours=6)
    flip_flop_window: timedelta = timedelta(minutes=30)
    signal_flip_min_confidence: float = 0.0
    maker_fee_rate: float = 0.0002
    taker_fee_rate: float = 0.0005
    min_leverage: float = 1.0
    max_leverage: float = 1.0
    available_margin_buffer: float = 0.10
    executable_margin_buffer: float = 0.001
    instruments: Dict[str, InstrumentConfig] = field(default_factory=dict)
    asset_manager: Optional[AssetManager] = None
    instrument_manager: Optional[InstrumentManager] = None
    initial_state: Optional["PositionManagerState"] = None
    persist: Optional[Callable[["PositionManagerState"], None]] = None


def production_position_manager_config(**overrides: Any) -> PositionManagerConfig:
    """Return server-compatible execution-policy defaults."""
    config = replace(PositionManagerConfig(), **overrides)
    if "max_margin_ratio" not in overrides and "position_size" in overrides and 0 < config.position_size <= 1:
        config.max_margin_ratio = config.position_size
    return config


@dataclass
class Position:
    venue: str
    instrument: str
    size: float = 0.0
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
    status: str = ""

    @property
    def side(self) -> Optional[Side]:
        if self.size < 0:
            return "sell"
        if self.size > 0:
            return "buy"
        return None

    @property
    def unrealized_pnl(self) -> float:
        return _move(self) * abs(self.size) * _positive_or(self.entry_price, 1.0)


@dataclass
class Order:
    venue: str
    instrument: str
    side: Side
    reason: str
    size_delta: float
    previous_size: float
    target_size: float
    price: float
    confidence: float
    expected_edge: float
    fee_rate: float
    estimated_fee: float
    leverage: float
    estimated_fee_value: float = 0.0
    margin: float = 0.0
    quantity: float = 0.0
    notional: float = 0.0
    settlement_currency: str = "USDT"
    min_size: float = 0.0
    lot_size: float = 0.0
    tick_size: float = 0.0
    score: float = 0.0
    take_profit: float = 0.0
    stop_loss: float = 0.0
    trailing_stop_activation: float = 0.0
    trailing_stop_distance: float = 0.0
    trailing_stop_min_profit: float = 0.0
    reduce_only: bool = False
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    subscription_id: Optional[int] = None
    replay: bool = False


@dataclass
class ClosedTrade:
    venue: str
    instrument: str
    side: Side
    size: float
    entry_price: float
    exit_price: float
    exit_move: float
    realized_gross: float
    fees: float
    realized_pnl: float
    mfe: float
    mae: float
    exit_reason: str
    closed_at: datetime


@dataclass
class PositionManagerState:
    positions: List[Position] = field(default_factory=list)
    closed_trades: List[ClosedTrade] = field(default_factory=list)


@dataclass
class InstrumentPositionStats:
    venue: str
    instrument: str
    settlement_currency: str
    side: Optional[Side]
    size: float
    quantity: float
    notional: float
    realized_pnl: float
    unrealized_pnl: float
    fees: float
    realized_pnl_percent: float
    unrealized_pnl_percent: float
    total_pnl_percent: float
    leverage: float


@dataclass
class CurrencyPositionStats:
    settlement_currency: str
    equity: float = 0.0
    available: float = 0.0
    used: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    fees: float = 0.0
    realized_pnl_percent: float = 0.0
    unrealized_pnl_percent: float = 0.0
    total_pnl_percent: float = 0.0


@dataclass
class PositionStats:
    equity: float = 0.0
    available: float = 0.0
    used: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    fees: float = 0.0
    realized_pnl_percent: float = 0.0
    unrealized_pnl_percent: float = 0.0
    total_pnl_percent: float = 0.0
    by_instrument: Dict[str, InstrumentPositionStats] = field(default_factory=dict)
    by_currency: Dict[str, CurrencyPositionStats] = field(default_factory=dict)


@dataclass
class _RebalanceCandidate:
    key: str
    position: Position
    delta: float
    weight: float
    context: Dict[str, Any]
    reason: str


@dataclass
class _ExecutableAllocation:
    quantity: float = 0.0
    margin: float = 0.0
    fee: float = 0.0


class PositionManager:
    """In-memory, fee-aware production-style position manager."""

    def __init__(self, client: Optional[SignalEventSource] = None, config: Optional[PositionManagerConfig] = None) -> None:
        self.client = client
        self.config = _normalize_config(config or production_position_manager_config())
        self.assets = self.config.asset_manager or AssetManager()
        self.instruments = self.config.instrument_manager or InstrumentManager()
        self._positions: Dict[str, Position] = {}
        self._closed: List[ClosedTrade] = []
        self._hydrate_state(self.config.initial_state)

    def update_config(self, config: PositionManagerConfig) -> None:
        """Replace manager policy without clearing runtime state."""

        next_config = replace(
            config,
            instruments=dict(config.instruments) if config.instruments else dict(self.config.instruments),
            asset_manager=config.asset_manager or self.assets,
            instrument_manager=config.instrument_manager or self.instruments,
            initial_state=None,
            persist=config.persist or self.config.persist,
        )
        self.config = _normalize_config(next_config)
        self.assets = self.config.asset_manager or self.assets
        self.instruments = self.config.instrument_manager or self.instruments

    async def run(self) -> AsyncIterator[Order]:
        """Consume attached client events and yield order recommendations."""

        if self.client is None:
            raise RuntimeError("position manager has no SignalsClient")
        async for event in self.client.events():
            for order in self.handle_event(event):
                yield order

    def add_position(self, position: Position) -> None:
        if position.leverage <= 0:
            position.leverage = self._min_leverage(_position_key(position.venue, position.instrument))
        self._positions[_position_key(position.venue, position.instrument)] = position
        self._persist()

    def update_position(self, position: Position) -> None:
        self.add_position(position)

    def replace_positions(self, positions: Iterable[Position]) -> None:
        self._positions = {}
        for position in positions:
            if not position.venue or not position.instrument or abs(position.size) <= 1e-9:
                continue
            copy = replace(position)
            if copy.leverage <= 0:
                copy.leverage = self._min_leverage(_position_key(copy.venue, copy.instrument))
            self._positions[_position_key(copy.venue, copy.instrument)] = copy
        self._persist()

    def close_position(self, venue: str, instrument: str) -> List[Order]:
        key = _position_key(venue, instrument)
        position = self._positions.get(key)
        if position is None or abs(position.size) <= 1e-9:
            return []
        order = self._order_for_delta(key, position, -position.size, 0.0, 0.0, "closing", position.confidence)
        if not self._order_meets_minimum(order):
            return []
        self._apply_delta(key, order.size_delta, position.last_price or position.entry_price, self._taker_fee_rate(key), "closing")
        self._persist()
        return [order]

    def update_price(self, venue: str, instrument: str, price: float, timestamp: Optional[datetime] = None) -> List[Order]:
        if price <= 0:
            raise ValueError("price must be positive")
        key = _position_key(venue, instrument)
        position = self._positions.get(key)
        if position is None or abs(position.size) <= 1e-9:
            return []
        position.last_price = price
        _update_excursion(position)
        reason = _exit_reason(position, price)
        if not reason:
            self._persist()
            return []
        fee_rate = self._maker_fee_rate(key) if reason == "take_profit" else self._taker_fee_rate(key)
        order = self._order_for_delta(key, position, -position.size, 0.0, 0.0, reason, position.confidence)
        order.fee_rate = fee_rate
        order.estimated_fee = _fee_value_for_notional(order.notional, fee_rate)
        order.estimated_fee_value = order.notional * fee_rate
        if not self._order_meets_minimum(order):
            self._persist()
            return []
        self._apply_delta(key, order.size_delta, price, fee_rate, reason, timestamp or datetime.now(timezone.utc))
        self._persist()
        return [order]

    def positions(self) -> List[Position]:
        return [replace(position) for position in sorted(self._positions.values(), key=lambda p: (p.venue, p.instrument))]

    def closed_trades(self) -> List[ClosedTrade]:
        return list(self._closed)

    def state(self) -> PositionManagerState:
        return PositionManagerState(self.positions(), [replace(trade) for trade in self._closed])

    def stats(self) -> PositionStats:
        stats = PositionStats()
        for asset in self.assets.assets():
            stats.equity += asset.equity
            stats.available += asset.available
            stats.used += asset.used
            stats.by_currency[asset.currency] = CurrencyPositionStats(asset.currency, asset.equity, asset.available, asset.used)
        for key, position in self._positions.items():
            metadata = self.instruments.instrument(position.venue, position.instrument)
            asset = self.assets.asset(metadata.settlement_currency)
            equity = _positive_or(asset.equity if asset else 0.0, (asset.cash + asset.used) if asset else 0.0, asset.cash if asset else 0.0, 1.0)
            price = _round_to_tick(position.last_price or position.entry_price, metadata.tick_size)
            contract_notional = _instrument_contract_notional(price, metadata)
            quantity = _round_down_to_step(abs(position.size), metadata.lot_size) if contract_notional > 0 else abs(position.size)
            notional = quantity * contract_notional
            realized = position.realized_pnl
            unrealized = self._position_unrealized_pnl(key, position)
            fees = position.fees
            stats.by_instrument[key] = InstrumentPositionStats(
                position.venue,
                position.instrument,
                metadata.settlement_currency,
                position.side,
                position.size,
                quantity,
                notional,
                realized,
                unrealized,
                fees,
                _ratio_or_zero(position.realized_pnl, equity),
                _ratio_or_zero(unrealized, equity),
                _ratio_or_zero(position.realized_pnl + unrealized, equity),
                position.leverage,
            )
            stats.realized_pnl += realized
            stats.unrealized_pnl += unrealized
            stats.fees += fees
            currency = stats.by_currency.setdefault(metadata.settlement_currency, CurrencyPositionStats(metadata.settlement_currency, equity))
            currency.realized_pnl += realized
            currency.unrealized_pnl += unrealized
            currency.fees += fees
            if currency.equity > 0:
                currency.realized_pnl_percent = currency.realized_pnl / currency.equity
                currency.unrealized_pnl_percent = currency.unrealized_pnl / currency.equity
                currency.total_pnl_percent = (currency.realized_pnl + currency.unrealized_pnl) / currency.equity
        if stats.equity <= 0:
            stats.equity = 1.0
        stats.realized_pnl_percent = stats.realized_pnl / stats.equity
        stats.unrealized_pnl_percent = stats.unrealized_pnl / stats.equity
        stats.total_pnl_percent = (stats.realized_pnl + stats.unrealized_pnl) / stats.equity
        return stats

    def handle_event(self, event: SignalsEvent) -> List[Order]:
        if not isinstance(event, SignalEvent):
            return []
        if event.replay:
            return []
        orders = self.handle_signal(event.signal)
        for order in orders:
            order.subscription_id = event.subscription_id
            order.replay = event.replay
        return orders

    def handle_signal(self, signal: Signal) -> List[Order]:
        if not signal.venue or not signal.instrument:
            return []
        if not self.instruments.has_instrument(signal.venue, signal.instrument):
            return []
        key = _position_key(signal.venue, signal.instrument)
        target_sign = _side_sign(signal.side)
        target_confidence = _clamp01(signal.confidence)
        if target_sign == 0 or target_confidence <= 0:
            return []
        edge = _fee_adjusted_expected_edge(signal, self._taker_fee_rate(key))
        if self.config.min_expected_edge > 0 and edge < self.config.min_expected_edge and not signal.manage_positions_only:
            return []
        now = signal.timestamp or datetime.now(timezone.utc)
        portfolio_budget = self._max_portfolio_margin_budget()
        min_delta = self._effective_min_order_delta()
        position = self._positions.get(key)
        if position is None or abs(position.size) <= 1e-9:
            if signal.manage_positions_only:
                return []
            if portfolio_budget < min_delta or not self._meets_minimum_position_size(portfolio_budget):
                return []
            if position is None:
                position = Position(signal.venue, signal.instrument, entry_price=signal.price, last_price=signal.price, opened_at=now)
                _reset_excursion(position)
                self._positions[key] = position
        else:
            is_flip = _sign(position.size) != 0 and _sign(position.size) != target_sign
            below_minimum = not self._meets_minimum_position_size(self._position_margin(key, position))
            if is_flip and self._should_suppress_flip_flop(position, signal, now):
                return []
            if not is_flip and not below_minimum and self.config.rebalance_interval and position.last_signal_at:
                if now < position.last_signal_at + self.config.rebalance_interval:
                    return []
        if signal.manage_positions_only and _sign(position.size) == 0:
            return []
        context_confidence = target_confidence
        override_side = target_sign
        if signal.manage_positions_only:
            if _sign(position.size) != target_sign:
                override_side = 0.0
            else:
                context_confidence = min(context_confidence, _clamp01(position.confidence))
        position.confidence = context_confidence
        position.last_signal_at = now
        if signal.price > 0:
            position.last_price = signal.price
            if position.entry_price <= 0:
                position.entry_price = signal.price
        if position.take_profit <= 0 or position.stop_loss <= 0 or position.side != signal.side:
            position.take_profit = signal.take_profit
            position.stop_loss = signal.stop_loss
        else:
            position.take_profit = _blend_risk(position.take_profit, signal.take_profit, 0.5)
            position.stop_loss = _blend_risk(position.stop_loss, signal.stop_loss, 0.5)
        trailing = self._trailing_config_for_signal(key, signal)
        if trailing["activation"] > 0 and trailing["distance"] > 0:
            position.trailing_stop_activation = trailing["activation"]
            position.trailing_stop_distance = trailing["distance"]
            position.trailing_stop_min_profit = trailing["min_profit"]
        position.leverage = self._select_leverage(key, context_confidence, edge, signal.score)
        orders = self._rebalance(
            {key: override_side},
            {
                key: {
                    "confidence": context_confidence,
                    "score": signal.score,
                    "edge": edge,
                    "take_profit": signal.take_profit,
                    "stop_loss": signal.stop_loss,
                    "trailing_stop_activation": trailing["activation"],
                    "trailing_stop_distance": trailing["distance"],
                    "trailing_stop_min_profit": trailing["min_profit"],
                    "manage_positions_only": signal.manage_positions_only,
                }
            },
        )
        self._persist()
        return orders

    def _should_suppress_flip_flop(self, position: Position, signal: Signal, now: datetime) -> bool:
        if signal.manage_positions_only:
            return False
        if position.last_signal_at is None or self.config.flip_flop_window <= timedelta(0):
            return False
        if now >= position.last_signal_at + self.config.flip_flop_window:
            return False
        return self.config.signal_flip_min_confidence <= 0 or signal.confidence + 1e-12 < self.config.signal_flip_min_confidence

    def _hydrate_state(self, state: Optional[PositionManagerState]) -> None:
        if state is None:
            return
        self._positions = {}
        for position in state.positions:
            if not position.venue or not position.instrument or abs(position.size) <= 1e-9:
                continue
            copy = replace(position)
            if copy.leverage <= 0:
                copy.leverage = self._min_leverage(_position_key(copy.venue, copy.instrument))
            self._positions[_position_key(copy.venue, copy.instrument)] = copy
        self._closed = [replace(trade) for trade in state.closed_trades]

    def _persist(self) -> None:
        if self.config.persist is not None:
            self.config.persist(self.state())

    def _rebalance(self, side_overrides: Dict[str, float], contexts: Dict[str, Dict[str, float]]) -> List[Order]:
        portfolio_budget = self._max_portfolio_margin_budget()
        if portfolio_budget <= 0 or not self._positions:
            return []
        weights: Dict[str, float] = {}
        sides: Dict[str, float] = {}
        for key, position in self._positions.items():
            has_override = key in side_overrides
            weight = _clamp01(position.confidence)
            if not has_override and weight <= 0:
                weight = _clamp01(self._position_margin(key, position) / portfolio_budget)
            side = side_overrides.get(key, _sign(position.size))
            weights[key] = weight
            sides[key] = side
        targets = self._allocate_target_sizes(sorted(list(self._positions)), weights, sides, contexts)
        reductions: List[_RebalanceCandidate] = []
        openings: List[_RebalanceCandidate] = []
        for key in sorted(list(self._positions)):
            position = self._positions[key]
            target_size = targets.get(key, 0.0)
            if abs(position.size) > 1e-9 and not self._meets_minimum_position_size(self._position_margin(key, position)):
                target_size = 0.0
            elif target_size != 0 and not self._meets_minimum_position_size(self._margin_for_quantity(key, position, target_size)):
                if abs(position.size) <= 1e-9:
                    position.confidence = weights.get(key, 0.0)
                    continue
                target_size = 0.0
            context = contexts.get(key, {})
            if context.get("manage_positions_only"):
                target_size = _manage_positions_only_target_size(position.size, target_size)
            delta = target_size - position.size
            if _is_flip_target(position.size, target_size):
                delta = -position.size
            if abs(delta) <= 1e-9:
                position.confidence = weights.get(key, 0.0)
                continue
            is_flip = abs(position.size) > 1e-9 and abs(target_size) > 1e-9 and _sign(position.size) != _sign(target_size)
            is_opening = abs(position.size) <= 1e-9 and abs(target_size) > 1e-9
            is_closing = abs(target_size) <= 1e-9 and abs(position.size) > 1e-9
            if not (is_flip or is_opening or is_closing) and self._margin_for_quantity(key, position, delta) < self._effective_min_order_delta():
                position.confidence = weights.get(key, 0.0)
                continue
            candidate = _RebalanceCandidate(
                key,
                replace(position),
                delta,
                weights.get(key, 0.0),
                context,
                _order_reason(position, target_size),
            )
            if _is_exposure_reduction(position.size, position.size + delta):
                reductions.append(candidate)
            else:
                openings.append(candidate)
        if reductions:
            return self._materialize_rebalance_orders(reductions, cap_openings=False)
        return self._materialize_rebalance_orders(openings, cap_openings=True)

    def _allocate_target_sizes(
        self,
        keys: List[str],
        weights: Dict[str, float],
        sides: Dict[str, float],
        contexts: Dict[str, Dict[str, float]],
    ) -> Dict[str, float]:
        targets: Dict[str, float] = {}
        portfolio_budget = self._max_portfolio_margin_budget()
        if portfolio_budget <= 0:
            return targets
        active = {key for key in keys if weights.get(key, 0.0) > 1e-9 and sides.get(key, 0.0) != 0}
        while active:
            total_weight = sum(weights.get(key, 0.0) for key in active)
            if total_weight <= 1e-9:
                break
            dropped = ""
            dropped_weight = math.inf
            for key in keys:
                if key not in active:
                    continue
                position = self._positions.get(key)
                if position is None:
                    continue
                desired_budget = portfolio_budget * weights.get(key, 0.0) / total_weight
                executable = self._executable_allocation_for_budget(key, position, desired_budget, contexts.get(key, {}))
                if executable.margin > 1e-9:
                    continue
                weight = weights.get(key, 0.0)
                if weight < dropped_weight or (abs(weight - dropped_weight) <= 1e-9 and (not dropped or key < dropped)):
                    dropped = key
                    dropped_weight = weight
            if not dropped:
                break
            active.remove(dropped)
        if not active:
            return targets
        total_weight = sum(weights.get(key, 0.0) for key in active)
        if total_weight <= 1e-9:
            return targets
        allocated = 0.0
        for key in keys:
            if key not in active:
                continue
            position = self._positions.get(key)
            if position is None:
                continue
            desired_budget = portfolio_budget * weights.get(key, 0.0) / total_weight
            executable = self._executable_allocation_for_budget(key, position, desired_budget, contexts.get(key, {}))
            if executable.margin <= 1e-9:
                continue
            if not self._meets_minimum_position_size(executable.margin):
                continue
            targets[key] = sides.get(key, 0.0) * executable.quantity
            allocated += executable.margin + executable.fee
        free = portfolio_budget - allocated
        if free <= 1e-9:
            return targets
        priority = sorted(keys, key=lambda key: (-weights.get(key, 0.0), key))
        for key in priority:
            if key not in active or free <= 1e-9:
                continue
            position = self._positions.get(key)
            if position is None:
                continue
            step = self._executable_lot_step_cost(key, position, contexts.get(key, {}))
            step_cost = step.margin + step.fee
            if step_cost <= 1e-9:
                executable = self._executable_allocation_for_budget(key, position, free, contexts.get(key, {}))
                if executable.quantity > 1e-9 and self._meets_minimum_position_size(executable.margin):
                    targets[key] = targets.get(key, 0.0) + sides.get(key, 0.0) * executable.quantity
                break
            steps = math.floor((free + 1e-9) / step_cost)
            if steps <= 0:
                continue
            next_size = targets.get(key, 0.0) + sides.get(key, 0.0) * steps * step.quantity
            next_margin = abs(next_size) * step.margin / step.quantity if step.quantity > 0 else 0.0
            if not self._meets_minimum_position_size(next_margin):
                continue
            targets[key] = next_size
            free -= steps * step_cost
        return targets

    def _materialize_rebalance_orders(self, candidates: List[_RebalanceCandidate], *, cap_openings: bool) -> List[Order]:
        orders: List[Order] = []
        opening_exposure_by_currency: Dict[str, float] = {}
        for candidate in candidates:
            delta = candidate.delta
            if candidate.context.get("manage_positions_only") and not _is_exposure_reduction(candidate.position.size, candidate.position.size + delta):
                if candidate.key in self._positions:
                    self._positions[candidate.key].confidence = candidate.weight
                continue
            if cap_openings and not _is_exposure_reduction(candidate.position.size, candidate.position.size + delta):
                metadata = self.instruments.instrument(candidate.position.venue, candidate.position.instrument)
                available = self._available_exposure_budget(metadata.settlement_currency) - opening_exposure_by_currency.get(metadata.settlement_currency, 0.0)
                if available <= 1e-9:
                    if candidate.key in self._positions:
                        self._positions[candidate.key].confidence = candidate.weight
                    continue
                delta = self._cap_opening_delta_to_budget(candidate.key, candidate.position, delta, candidate.context, available)
                if abs(delta) <= 1e-9:
                    if candidate.key in self._positions:
                        self._positions[candidate.key].confidence = candidate.weight
                    continue
            context = candidate.context
            order = self._order_for_delta(
                candidate.key,
                candidate.position,
                delta,
                context.get("edge", 0.0),
                context.get("score", 0.0),
                candidate.reason,
                context.get("confidence", candidate.position.confidence),
            )
            order.take_profit = context.get("take_profit", 0.0)
            order.stop_loss = context.get("stop_loss", 0.0)
            order.trailing_stop_activation = context.get("trailing_stop_activation", 0.0)
            order.trailing_stop_distance = context.get("trailing_stop_distance", 0.0)
            order.trailing_stop_min_profit = context.get("trailing_stop_min_profit", 0.0)
            if not self._order_meets_minimum(order):
                if candidate.key in self._positions:
                    self._positions[candidate.key].confidence = candidate.weight
                continue
            orders.append(order)
            if cap_openings and not _is_exposure_reduction(order.previous_size, order.target_size):
                opening_exposure_by_currency[order.settlement_currency] = opening_exposure_by_currency.get(order.settlement_currency, 0.0) + _order_budget_cost(order)
            self._apply_delta(
                candidate.key,
                order.size_delta,
                candidate.position.last_price or candidate.position.entry_price,
                self._taker_fee_rate(candidate.key),
                candidate.reason,
            )
            if candidate.key in self._positions:
                self._positions[candidate.key].confidence = candidate.weight
                if order.trailing_stop_activation > 0 and order.trailing_stop_distance > 0:
                    self._positions[candidate.key].trailing_stop_activation = order.trailing_stop_activation
                    self._positions[candidate.key].trailing_stop_distance = order.trailing_stop_distance
                    self._positions[candidate.key].trailing_stop_min_profit = order.trailing_stop_min_profit
        return orders

    def _available_exposure_budget(self, currency: str) -> float:
        portfolio_budget = self._available_portfolio_budget()
        asset = self.assets.asset(currency)
        if asset is None:
            return portfolio_budget
        if asset.available <= 0:
            return 0.0
        budget = max(0.0, asset.available) * min(max(asset.max_usage or 1.0, 0.0), 1.0)
        if self.config.available_margin_buffer > 0:
            budget *= 1 - self.config.available_margin_buffer
        return min(budget, portfolio_budget)

    def _available_portfolio_budget(self) -> float:
        max_budget = self._max_portfolio_margin_budget()
        used = sum(self._position_margin(key, position) for key, position in self._positions.items())
        return max(0.0, max_budget - used)

    def _max_portfolio_margin_budget(self) -> float:
        capital = self._portfolio_capital()
        if capital <= 0 or self.config.max_margin_ratio <= 0:
            return 0.0
        return capital * self.config.max_margin_ratio

    def _portfolio_capital(self) -> float:
        capital = sum(_positive_or(asset.equity, asset.cash + asset.used, asset.cash) for asset in self.assets.assets())
        return capital if capital > 0 else 1.0

    def _position_margin(self, key: str, position: Position) -> float:
        if abs(position.size) <= 1e-9:
            return 0.0
        return self._margin_for_quantity(key, position, position.size)

    def _margin_for_quantity(self, key: str, position: Position, quantity: float) -> float:
        if abs(quantity) <= 1e-9:
            return 0.0
        metadata = self.instruments.instrument(position.venue, position.instrument)
        price = _round_to_tick(position.last_price or position.entry_price, metadata.tick_size)
        contract_notional = _instrument_contract_notional(price, metadata)
        leverage = _positive_or(position.leverage, self._min_leverage(key), 1.0)
        if contract_notional <= 0 or leverage <= 0:
            return 0.0
        return abs(quantity) * contract_notional / leverage

    def _position_unrealized_pnl(self, key: str, position: Position) -> float:
        if abs(position.size) <= 1e-9 or position.entry_price <= 0 or position.last_price <= 0:
            return 0.0
        return self._realized_gross_for_quantity(key, position, abs(position.size), position.last_price)

    def _realized_gross_for_quantity(self, key: str, position: Position, quantity: float, exit_price: float) -> float:
        if quantity <= 1e-9 or position.entry_price <= 0 or exit_price <= 0:
            return 0.0
        metadata = self.instruments.instrument(position.venue, position.instrument)
        contract_value = _positive_or(metadata.contract_value, 1.0)
        contract_multiplier = _positive_or(metadata.contract_multiplier, 1.0)
        move = position.entry_price - exit_price if position.size < 0 else exit_price - position.entry_price
        return move * quantity * contract_value * contract_multiplier

    def _fee_for_quantity(self, key: str, position: Position, quantity: float, price: float, fee_rate: float) -> float:
        if quantity <= 1e-9 or price <= 0 or fee_rate <= 0:
            return 0.0
        metadata = self.instruments.instrument(position.venue, position.instrument)
        return quantity * _instrument_contract_notional(price, metadata) * fee_rate

    def _executable_allocation_for_budget(
        self,
        key: str,
        position: Position,
        budget: float,
        context: Dict[str, float],
    ) -> _ExecutableAllocation:
        if budget <= 1e-9:
            return _ExecutableAllocation()
        metadata = self.instruments.instrument(position.venue, position.instrument)
        price = _round_to_tick(position.last_price or position.entry_price, metadata.tick_size)
        leverage = self._select_leverage(key, context.get("confidence", position.confidence), context.get("edge", 0.0), context.get("score", 0.0))
        contract_notional = _instrument_contract_notional(price, metadata)
        if contract_notional <= 0 or leverage <= 0:
            return _ExecutableAllocation()
        fee_rate = self._taker_fee_rate(key)
        max_margin = budget
        if metadata.lot_size <= 0:
            fee_multiplier = 1 + leverage * fee_rate
            if fee_multiplier > 0:
                max_margin = budget / fee_multiplier
        quantity = _round_down_to_step(max_margin * leverage / contract_notional, metadata.lot_size)
        while quantity > 1e-9:
            if metadata.min_size > 0 and quantity < metadata.min_size:
                return _ExecutableAllocation()
            margin = quantity * contract_notional / leverage
            fee = quantity * contract_notional * fee_rate
            if margin + fee <= budget + 1e-9:
                return _ExecutableAllocation(quantity, margin, fee)
            if metadata.lot_size <= 0:
                return _ExecutableAllocation()
            quantity = _round_down_to_step(quantity - metadata.lot_size, metadata.lot_size)
        return _ExecutableAllocation()

    def _executable_lot_step_cost(self, key: str, position: Position, context: Dict[str, float]) -> _ExecutableAllocation:
        metadata = self.instruments.instrument(position.venue, position.instrument)
        if metadata.lot_size <= 0:
            return _ExecutableAllocation()
        price = _round_to_tick(position.last_price or position.entry_price, metadata.tick_size)
        leverage = self._select_leverage(key, context.get("confidence", position.confidence), context.get("edge", 0.0), context.get("score", 0.0))
        contract_notional = _instrument_contract_notional(price, metadata)
        if contract_notional <= 0 or leverage <= 0:
            return _ExecutableAllocation()
        return _ExecutableAllocation(
            metadata.lot_size,
            metadata.lot_size * contract_notional / leverage,
            metadata.lot_size * contract_notional * self._taker_fee_rate(key),
        )

    def _cap_opening_delta_to_budget(
        self,
        key: str,
        position: Position,
        delta: float,
        context: Dict[str, float],
        budget: float,
    ) -> float:
        if abs(delta) <= 1e-9 or budget <= 1e-9:
            return 0.0
        executable = self._executable_allocation_for_budget(key, position, budget, context)
        if executable.margin <= 1e-9:
            return 0.0
        if not self._meets_minimum_position_size(executable.margin):
            return 0.0
        if executable.quantity < abs(delta):
            return self._cap_executable_delta_with_buffered_cost(key, position, _sign(delta) * executable.quantity, context, budget)
        order = self._order_for_delta(
            key,
            position,
            delta,
            context.get("edge", 0.0),
            context.get("score", 0.0),
            "budget-check",
            context.get("confidence", position.confidence),
        )
        if _order_budget_cost(order) > budget + 1e-9:
            return self._cap_executable_delta_with_buffered_cost(key, position, _sign(delta) * executable.quantity, context, budget)
        return delta

    def _cap_executable_delta_with_buffered_cost(
        self,
        key: str,
        position: Position,
        delta: float,
        context: Dict[str, float],
        budget: float,
    ) -> float:
        if abs(delta) <= 1e-9 or budget <= 1e-9:
            return 0.0
        metadata = self.instruments.instrument(position.venue, position.instrument)
        quantity_step = metadata.lot_size if metadata.lot_size > 0 else 0.0
        candidate = abs(delta)
        while candidate > 1e-9:
            order = self._order_for_delta(key, position, _sign(delta) * candidate, context.get("edge", 0.0), context.get("score", 0.0), "budget-check", context.get("confidence", position.confidence))
            if _order_budget_cost(order) <= budget + 1e-9:
                return _sign(delta) * candidate
            if quantity_step <= 1e-9:
                return self._cap_continuous_opening_delta_to_budget(key, position, delta, context, budget)
            candidate -= quantity_step
        return 0.0

    def _cap_continuous_opening_delta_to_budget(
        self,
        key: str,
        position: Position,
        delta: float,
        context: Dict[str, float],
        budget: float,
    ) -> float:
        if abs(delta) <= 1e-9 or budget <= 1e-9:
            return 0.0
        low = 0.0
        high = abs(delta)
        for _ in range(64):
            mid = (low + high) / 2
            if mid <= 1e-9:
                break
            order = self._order_for_delta(key, position, _sign(delta) * mid, context.get("edge", 0.0), context.get("score", 0.0), "budget-check", context.get("confidence", position.confidence))
            if _order_budget_cost(order) <= budget + 1e-9:
                low = mid
            else:
                high = mid
        return 0.0 if low <= 1e-9 else _sign(delta) * low

    def _order_for_delta(self, key: str, position: Position, delta: float, edge: float, score: float, reason: str, confidence: float) -> Order:
        fee_rate = self._taker_fee_rate(key)
        leverage = self._select_leverage(key, confidence, edge, score)
        metadata = self.instruments.instrument(position.venue, position.instrument)
        price = _round_to_tick(position.last_price or position.entry_price, metadata.tick_size)
        requested_abs_delta = abs(delta)
        contract_notional = _instrument_contract_notional(price, metadata)
        closes_to_zero = abs(position.size) > 1e-9 and abs(position.size + delta) <= 1e-9
        quantity = _round_down_to_step(requested_abs_delta, metadata.lot_size) if contract_notional > 0 and not closes_to_zero else requested_abs_delta
        notional = quantity * contract_notional
        margin = notional / leverage if leverage > 0 else 0.0
        executable_delta = _sign(delta) * quantity
        return Order(
            venue=position.venue,
            instrument=position.instrument,
            side="sell" if delta < 0 else "buy",
            reason=reason,
            size_delta=executable_delta,
            previous_size=position.size,
            target_size=position.size + executable_delta,
            price=price,
            confidence=confidence,
            score=score,
            expected_edge=edge,
            fee_rate=fee_rate,
            estimated_fee=_fee_value_for_notional(notional, fee_rate),
            leverage=leverage,
            estimated_fee_value=notional * fee_rate,
            margin=margin,
            quantity=quantity,
            notional=notional,
            settlement_currency=metadata.settlement_currency,
            min_size=metadata.min_size,
            lot_size=metadata.lot_size,
            tick_size=metadata.tick_size,
            trailing_stop_activation=position.trailing_stop_activation,
            trailing_stop_distance=position.trailing_stop_distance,
            trailing_stop_min_profit=position.trailing_stop_min_profit,
            reduce_only=_is_exposure_reduction(position.size, position.size + executable_delta),
        )

    def _apply_delta(
        self,
        key: str,
        delta: float,
        price: float,
        fee_rate: float,
        reason: str = "",
        now: Optional[datetime] = None,
    ) -> None:
        position = self._positions.get(key)
        if position is None:
            return
        if position.size == 0 or _sign(position.size) == _sign(delta):
            next_abs = abs(position.size) + abs(delta)
            if price > 0:
                position.entry_price = (position.entry_price * abs(position.size) + price * abs(delta)) / next_abs if next_abs > 0 and abs(position.size) > 1e-9 and position.entry_price > 0 else price
                position.last_price = price
            fee = self._fee_for_quantity(key, position, abs(delta), price, fee_rate)
            position.fees += fee
            position.realized_pnl -= fee
            position.size += delta
            _reset_excursion(position)
            return
        if price > 0:
            position.last_price = price
            _update_excursion(position)
        closing = min(abs(position.size), abs(delta))
        gross = self._realized_gross_for_quantity(key, position, closing, price)
        fee = self._fee_for_quantity(key, position, closing, price, fee_rate)
        position.realized_gross += gross
        position.fees += fee
        position.realized_pnl += gross - fee
        closed = ClosedTrade(
            venue=position.venue,
            instrument=position.instrument,
            side=position.side or "buy",
            size=closing,
            entry_price=position.entry_price,
            exit_price=price,
            exit_move=_move(position),
            realized_gross=position.realized_gross,
            fees=position.fees,
            realized_pnl=position.realized_pnl,
            mfe=position.mfe,
            mae=position.mae,
            exit_reason=reason,
            closed_at=now or datetime.now(timezone.utc),
        )
        remaining = abs(delta) - closing
        if remaining <= 1e-9:
            position.size += delta
            if abs(position.size) <= 1e-9:
                del self._positions[key]
                self._closed.append(closed)
            return
        self._closed.append(closed)
        position.size = _sign(delta) * remaining
        position.entry_price = price
        position.last_price = price
        position.confidence = 0.0
        position.realized_gross = 0.0
        position.fees = self._fee_for_quantity(key, position, remaining, price, fee_rate)
        position.realized_pnl = -position.fees
        _reset_excursion(position)

    def _effective_min_order_delta(self) -> float:
        return max(self.config.min_order_delta, 0.0) * self._max_portfolio_margin_budget()

    def _minimum_position_size(self) -> float:
        if self.config.min_position_size_ratio <= 0:
            return 0.0
        return self.config.min_position_size_ratio * self._portfolio_capital()

    def _meets_minimum_position_size(self, size: float) -> bool:
        minimum = self._minimum_position_size()
        return minimum <= 0 or abs(size) + 1e-9 >= minimum

    def _select_leverage(self, key: str, confidence: float, edge: float, score: float = 0.0) -> float:
        min_leverage = self._min_leverage(key)
        max_leverage = max(self._max_leverage(key), min_leverage)
        if math.isclose(max_leverage, min_leverage):
            return min_leverage
        edge_score = _clamp01(edge / max(self.config.min_expected_edge * 3, 0.001))
        quality = _clamp01(_clamp01(confidence) * 0.65 + edge_score * 0.25 + min(abs(score), 1.0) * 0.10)
        return min_leverage + (max_leverage - min_leverage) * quality

    def _maker_fee_rate(self, key: str) -> float:
        override = self.config.instruments.get(key)
        return override.maker_fee_rate if override and override.maker_fee_rate is not None else self.config.maker_fee_rate

    def _taker_fee_rate(self, key: str) -> float:
        override = self.config.instruments.get(key)
        return override.taker_fee_rate if override and override.taker_fee_rate is not None else self.config.taker_fee_rate

    def _min_leverage(self, key: str) -> float:
        override = self.config.instruments.get(key)
        return override.min_leverage if override and override.min_leverage is not None else self.config.min_leverage

    def _max_leverage(self, key: str) -> float:
        override = self.config.instruments.get(key)
        configured = override.max_leverage if override and override.max_leverage is not None else self.config.max_leverage
        venue, instrument = key.split(":", 1)
        metadata_max = self.instruments.instrument(venue, instrument).max_leverage
        return min(configured, metadata_max) if metadata_max > 0 and configured > 0 else configured

    def _trailing_config_for_signal(self, key: str, signal: Signal) -> Dict[str, float]:
        override = self.config.instruments.get(key)
        activation = _positive_or(
            signal.trailing_stop_activation,
            override.trailing_stop_activation if override and override.trailing_stop_activation is not None else 0.0,
        )
        distance = _positive_or(
            signal.trailing_stop_distance,
            override.trailing_stop_distance if override and override.trailing_stop_distance is not None else 0.0,
        )
        min_profit = _positive_or(
            signal.trailing_stop_min_profit,
            override.trailing_stop_min_profit if override and override.trailing_stop_min_profit is not None else 0.0,
        )
        if activation <= 0 or distance <= 0:
            return {"activation": 0.0, "distance": 0.0, "min_profit": 0.0}
        fee_floor = 2 * self._taker_fee_rate(key)
        min_profit = max(min_profit, fee_floor)
        if activation < min_profit + 1e-9:
            activation = min_profit + min(distance, fee_floor)
        return {"activation": activation, "distance": distance, "min_profit": min_profit}

    def _order_meets_minimum(self, order: Order) -> bool:
        if order.quantity <= 0:
            return False
        if order.reason in {"closing", "flip"}:
            return True
        if order.min_size > 0 and order.quantity > 0 and order.quantity < order.min_size:
            return False
        return True


def _signal_from_payload(payload: Dict[str, Any]) -> Signal:
    return Signal(
        venue=payload.get("venue", ""),
        instrument=payload.get("instrument", ""),
        timeframe=payload.get("timeframe"),
        confidence=float(payload.get("confidence", 0.0)),
        side=payload.get("side", "buy"),
        take_profit=float(payload.get("takeProfit", payload.get("take_profit", 0.0))),
        stop_loss=float(payload.get("stopLoss", payload.get("stop_loss", 0.0))),
        trailing_stop_activation=float(payload.get("trailingStopActivation", payload.get("trailing_stop_activation", 0.0)) or 0.0),
        trailing_stop_distance=float(payload.get("trailingStopDistance", payload.get("trailing_stop_distance", 0.0)) or 0.0),
        trailing_stop_min_profit=float(payload.get("trailingStopMinProfit", payload.get("trailing_stop_min_profit", 0.0)) or 0.0),
        score=float(payload.get("score", 0.0)),
        components=[],
        model_variant=payload.get("modelVariant", payload.get("model_variant")),
        model_version=payload.get("modelVersion", payload.get("model_version")),
        prediction_mode=payload.get("predictionMode", payload.get("prediction_mode")),
        confidence_mapping=payload.get("confidenceMapping", payload.get("confidence_mapping")),
        up_probability=float(payload.get("upProbability", payload.get("up_probability", 0.0)) or 0.0),
        down_probability=float(payload.get("downProbability", payload.get("down_probability", 0.0)) or 0.0),
        directional_edge=float(payload.get("directionalEdge", payload.get("directional_edge", 0.0)) or 0.0),
        normalized_edge=float(payload.get("normalizedEdge", payload.get("normalized_edge", 0.0)) or 0.0),
        expected_value=float(payload.get("expectedValue", payload.get("expected_value", 0.0)) or 0.0),
        regime=payload.get("regime"),
        regime_confidence=float(payload.get("regimeConfidence", payload.get("regime_confidence", 0.0)) or 0.0),
        volatility_state=payload.get("volatilityState", payload.get("volatility_state")),
        squeeze_state=payload.get("squeezeState", payload.get("squeeze_state")),
        trend_state=payload.get("trendState", payload.get("trend_state")),
        atr_percent=float(payload.get("atrPercent", payload.get("atr_percent", 0.0)) or 0.0),
        signal_ttl=float(payload.get("signalTTL", payload.get("signal_ttl", 0.0)) or 0.0),
        generated_at=_parse_time(payload.get("generatedAt", payload.get("generated_at"))),
        artifact_id=payload.get("artifactID", payload.get("artifact_id")),
        artifact_version=payload.get("artifactVersion", payload.get("artifact_version")),
        rejected_reason=payload.get("rejectedReason", payload.get("rejected_reason")),
        manage_positions_only=bool(payload.get("managePositionsOnly", payload.get("manage_positions_only", False))),
        timestamp=_parse_time(payload.get("timestamp")),
        price=float(payload.get("price", 0.0) or 0.0),
    )


def _normalize_config(config: PositionManagerConfig) -> PositionManagerConfig:
    if config.max_margin_ratio <= 0:
        if 0 < config.position_size <= 1:
            config.max_margin_ratio = config.position_size
        else:
            config.max_margin_ratio = 1.0
    config.max_margin_ratio = min(max(config.max_margin_ratio, 0.0), 1.0)
    config.position_size = max(config.position_size, 0.0)
    config.min_expected_edge = max(config.min_expected_edge, 0.0)
    config.min_order_delta = min(max(config.min_order_delta, 0.0), 1.0)
    config.min_position_size_ratio = min(max(config.min_position_size_ratio, 0.0), 1.0)
    if config.flip_flop_window < timedelta(0):
        config.flip_flop_window = timedelta(0)
    config.signal_flip_min_confidence = min(max(config.signal_flip_min_confidence, 0.0), 1.0)
    config.maker_fee_rate = max(config.maker_fee_rate, 0.0)
    config.taker_fee_rate = max(config.taker_fee_rate, 0.0)
    config.min_leverage = max(config.min_leverage, 0.0)
    config.max_leverage = max(config.max_leverage, 0.0)
    config.available_margin_buffer = min(max(config.available_margin_buffer, 0.0), 0.95)
    config.executable_margin_buffer = min(max(config.executable_margin_buffer, 0.0), 0.05)
    for key, instrument in list(config.instruments.items()):
        config.instruments[key] = _normalize_instrument_config(instrument)
    return config


def _normalize_instrument_config(config: InstrumentConfig) -> InstrumentConfig:
    if config.maker_fee_rate is not None:
        config.maker_fee_rate = max(config.maker_fee_rate, 0.0)
    if config.taker_fee_rate is not None:
        config.taker_fee_rate = max(config.taker_fee_rate, 0.0)
    if config.min_leverage is not None:
        config.min_leverage = max(config.min_leverage, 0.0)
    if config.max_leverage is not None:
        config.max_leverage = max(config.max_leverage, 0.0)
    if config.trailing_stop_activation is not None:
        config.trailing_stop_activation = max(config.trailing_stop_activation, 0.0)
    if config.trailing_stop_distance is not None:
        config.trailing_stop_distance = max(config.trailing_stop_distance, 0.0)
    if config.trailing_stop_min_profit is not None:
        config.trailing_stop_min_profit = max(config.trailing_stop_min_profit, 0.0)
    return config


def _position_key(venue: str, instrument: str) -> str:
    return f"{venue}:{instrument}"


def _side_sign(side: Side) -> float:
    return 1.0 if side == "buy" else -1.0 if side == "sell" else 0.0


def _sign(value: float) -> float:
    return -1.0 if value < 0 else 1.0 if value > 0 else 0.0


def _clamp01(value: float) -> float:
    return min(max(value, 0.0), 1.0)


def _positive_or(*values: float) -> float:
    for value in values:
        if value > 0:
            return value
    return 0.0


def _round_down_to_step(value: float, step: float) -> float:
    if value <= 0 or step <= 0:
        return value
    return math.floor(value / step) * step


def _round_to_tick(value: float, tick: float) -> float:
    if value <= 0 or tick <= 0:
        return value
    return round(value / tick) * tick


def _confidence_from_size(position: Position, position_size: float) -> float:
    return _clamp01(abs(position.size) if position_size <= 0 else abs(position.size) / position_size)


def _expected_edge(signal: Signal) -> float:
    return _clamp01(signal.confidence) * max(signal.take_profit, 0.0) - (1 - _clamp01(signal.confidence)) * max(signal.stop_loss, 0.0)


def _fee_adjusted_expected_edge(signal: Signal, taker_fee_rate: float) -> float:
    return _expected_edge(signal) - 2 * taker_fee_rate


def _order_budget_cost(order: Order) -> float:
    return max(order.margin, 0.0) + max(order.estimated_fee, 0.0)


def _fee_value_for_notional(notional: float, fee_rate: float) -> float:
    if notional <= 0 or fee_rate <= 0:
        return 0.0
    return notional * fee_rate


def _instrument_contract_notional(price: float, metadata: InstrumentMetadata) -> float:
    if price <= 0:
        return 0.0
    return price * _positive_or(metadata.contract_value, 1.0) * _positive_or(metadata.contract_multiplier, 1.0)


def _ratio_or_zero(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > 0 else 0.0


def _fee_exposure_for_margin(margin: float, leverage: float, fee_rate: float) -> float:
    if margin <= 0 or leverage <= 0 or fee_rate <= 0:
        return 0.0
    return margin * leverage * fee_rate


def _move(position: Position) -> float:
    if position.entry_price <= 0 or position.last_price <= 0:
        return 0.0
    return (position.entry_price - position.last_price) / position.entry_price if position.size < 0 else (position.last_price - position.entry_price) / position.entry_price


def _take_profit_price(position: Position) -> float:
    if position.take_profit_price > 0:
        return position.take_profit_price
    if position.entry_price <= 0 or position.take_profit <= 0:
        return 0.0
    return position.entry_price * (1 - position.take_profit) if position.size < 0 else position.entry_price * (1 + position.take_profit)


def _stop_loss_price(position: Position) -> float:
    if position.stop_loss_price > 0:
        return position.stop_loss_price
    if position.entry_price <= 0 or position.stop_loss <= 0:
        return 0.0
    return position.entry_price * (1 + position.stop_loss) if position.size < 0 else position.entry_price * (1 - position.stop_loss)


def _take_profit_triggered(position: Position, price: float) -> bool:
    target = _take_profit_price(position)
    if target <= 0:
        return False
    return price <= target if position.size < 0 else price >= target


def _stop_loss_triggered(position: Position, price: float) -> bool:
    target = _stop_loss_price(position)
    if target <= 0:
        return False
    return price >= target if position.size < 0 else price <= target


def _trailing_stop_triggered(position: Position) -> bool:
    if position.trailing_stop_activation <= 0 or position.trailing_stop_distance <= 0:
        return False
    if position.mfe + 1e-9 < position.trailing_stop_activation:
        return False
    floor = max(position.mfe - position.trailing_stop_distance, position.trailing_stop_min_profit)
    return _move(position) <= floor + 1e-9


def _exit_reason(position: Position, price: float) -> str:
    if price <= 0:
        return ""
    if _take_profit_triggered(position, price):
        return "take_profit"
    if _stop_loss_triggered(position, price):
        return "stop_loss"
    if _trailing_stop_triggered(position):
        return "trailing_stop"
    return ""


def _reset_excursion(position: Position) -> None:
    move = _move(position)
    position.mfe = max(move, 0.0)
    position.mae = min(move, 0.0)


def _update_excursion(position: Position) -> None:
    move = _move(position)
    position.mfe = max(position.mfe, move)
    position.mae = min(position.mae, move)


def _order_reason(position: Position, target_size: float) -> str:
    if abs(position.size) <= 1e-9:
        return "opening"
    if abs(target_size) <= 1e-9:
        return "closing"
    if _sign(position.size) != _sign(target_size):
        return "flip"
    return "rebalance"


def _is_flip_target(previous_size: float, target_size: float) -> bool:
    return abs(previous_size) > 1e-9 and abs(target_size) > 1e-9 and _sign(previous_size) != _sign(target_size)


def _manage_positions_only_target_size(previous_size: float, target_size: float) -> float:
    if abs(previous_size) <= 1e-9:
        return 0.0
    if abs(target_size) <= 1e-9:
        return 0.0
    if _sign(previous_size) != _sign(target_size):
        return 0.0
    if abs(target_size) > abs(previous_size):
        return previous_size
    return target_size


def _is_exposure_reduction(previous_size: float, target_size: float) -> bool:
    if abs(previous_size) <= 1e-9:
        return False
    if abs(target_size) <= 1e-9:
        return True
    if _sign(previous_size) != _sign(target_size):
        return True
    return abs(target_size) < abs(previous_size) - 1e-9


def _blend_risk(current: float, incoming: float, gate: float) -> float:
    if current <= 0:
        return incoming
    if incoming <= 0:
        return current
    gate = _clamp01(gate)
    return current * (1 - gate) + incoming * gate


def _parse_time(value: Any) -> Optional[datetime]:
    if isinstance(value, datetime):
        return value
    if not value:
        return None
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


def _asset_payload(asset: AssetSnapshot) -> Dict[str, Any]:
    return {
        "venue": asset.venue,
        "currency": asset.currency,
        "cash": asset.cash,
        "available": asset.available,
        "used": asset.used,
        "equity": asset.equity,
        "maxUsage": asset.max_usage,
    }


def _position_payload(position: Position) -> Dict[str, Any]:
    return {
        "venue": position.venue,
        "instrument": position.instrument,
        "side": position.side or "",
        "status": position.status,
        "size": abs(position.size),
        "entryPrice": position.entry_price,
        "markPrice": position.last_price,
        "leverage": position.leverage,
        "takeProfitPrice": position.take_profit_price,
        "stopLossPrice": position.stop_loss_price,
    }


__all__ = [
    "SignalsWebSocketToken",
    "Side",
    "SignalComponent",
    "Signal",
    "ReadyEvent",
    "SubscribedEvent",
    "UnsubscribedEvent",
    "InfoEvent",
    "SignalEvent",
    "CreateMarketOrderEvent",
    "UpdateTPSLEvent",
    "WithdrawEvent",
    "ErrorEvent",
    "SignalsEvent",
    "SignalEventSource",
    "SignalsClient",
    "PositionManager",
    "PositionManagerConfig",
    "InstrumentConfig",
    "AssetSnapshot",
    "AssetManager",
    "InstrumentMetadata",
    "InstrumentManager",
    "Position",
    "Order",
    "ClosedTrade",
    "PositionManagerState",
    "PositionStats",
    "InstrumentPositionStats",
    "CurrencyPositionStats",
    "parse_event",
    "production_position_manager_config",
]
