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
from typing import Any, AsyncIterator, Dict, Iterable, List, Literal, Optional, Union

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
    timeframe: Optional[str] = None
    score: float = 0.0
    components: List[SignalComponent] = field(default_factory=list)
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
class ErrorEvent:
    type: Literal["error"]
    code: Optional[str] = None
    message: Optional[str] = None


SignalsEvent = Union[ReadyEvent, SubscribedEvent, UnsubscribedEvent, InfoEvent, SignalEvent, ErrorEvent]


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


@dataclass
class InstrumentConfig:
    maker_fee_rate: Optional[float] = None
    taker_fee_rate: Optional[float] = None
    min_leverage: Optional[float] = None
    max_leverage: Optional[float] = None


@dataclass
class AssetSnapshot:
    currency: str
    cash: float = 0.0
    available: float = 0.0
    used: float = 0.0
    equity: float = 0.0
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
    max_leverage: float = 0.0


class InstrumentManager:
    """Tracks lot size, min size, tick size, settlement currency, and leverage caps."""

    def __init__(self) -> None:
        self._instruments: Dict[str, InstrumentMetadata] = {}

    def update_instrument(self, metadata: InstrumentMetadata) -> None:
        if metadata.venue and metadata.instrument:
            self._instruments[_position_key(metadata.venue, metadata.instrument)] = metadata

    def instrument(self, venue: str, instrument: str) -> InstrumentMetadata:
        return self._instruments.get(_position_key(venue, instrument), InstrumentMetadata(venue, instrument))

    def has_instrument(self, venue: str, instrument: str) -> bool:
        return _position_key(venue, instrument) in self._instruments

    def instruments(self) -> List[InstrumentMetadata]:
        return [self._instruments[key] for key in sorted(self._instruments)]


@dataclass
class PositionManagerConfig:
    position_size: float = 1.0
    min_expected_edge: float = 0.0045
    min_order_delta: float = 0.20
    rebalance_interval: timedelta = timedelta(hours=6)
    maker_fee_rate: float = 0.0002
    taker_fee_rate: float = 0.0005
    min_leverage: float = 1.0
    max_leverage: float = 1.0
    instruments: Dict[str, InstrumentConfig] = field(default_factory=dict)
    asset_manager: Optional[AssetManager] = None
    instrument_manager: Optional[InstrumentManager] = None


def production_position_manager_config(**overrides: Any) -> PositionManagerConfig:
    """Return server-compatible execution-policy defaults."""

    return replace(PositionManagerConfig(), **overrides)


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
    leverage: float = 1.0
    realized_gross: float = 0.0
    fees: float = 0.0
    realized_pnl: float = 0.0
    opened_at: Optional[datetime] = None
    last_signal_at: Optional[datetime] = None

    @property
    def side(self) -> Optional[Side]:
        if self.size < 0:
            return "sell"
        if self.size > 0:
            return "buy"
        return None

    @property
    def unrealized_pnl(self) -> float:
        return _move(self) * abs(self.size)


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
    quantity: float = 0.0
    notional: float = 0.0
    settlement_currency: str = "USDT"
    min_size: float = 0.0
    lot_size: float = 0.0
    tick_size: float = 0.0
    score: float = 0.0
    take_profit: float = 0.0
    stop_loss: float = 0.0
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
    realized_gross: float
    fees: float
    realized_pnl: float
    closed_at: datetime


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


class PositionManager:
    """In-memory, fee-aware production-style position manager."""

    def __init__(self, client: Optional[SignalsClient] = None, config: Optional[PositionManagerConfig] = None) -> None:
        self.client = client
        self.config = _normalize_config(config or production_position_manager_config())
        self.assets = self.config.asset_manager or AssetManager()
        self.instruments = self.config.instrument_manager or InstrumentManager()
        self._positions: Dict[str, Position] = {}
        self._closed: List[ClosedTrade] = []

    async def run(self) -> AsyncIterator[Order]:
        """Consume attached client events and yield order recommendations."""

        if self.client is None:
            raise RuntimeError("position manager has no SignalsClient")
        async for event in self.client.events():
            for order in self.handle_event(event):
                yield order

    def add_position(self, position: Position) -> None:
        self._positions[_position_key(position.venue, position.instrument)] = position

    def update_position(self, position: Position) -> None:
        self.add_position(position)

    def close_position(self, venue: str, instrument: str) -> List[Order]:
        key = _position_key(venue, instrument)
        position = self._positions.get(key)
        if position is None or abs(position.size) <= 1e-9:
            return []
        order = self._order_for_delta(key, position, -position.size, 0.0, 0.0, "closing", position.confidence)
        if not self._order_meets_minimum(order):
            return []
        self._apply_delta(key, -position.size, position.last_price or position.entry_price, self._taker_fee_rate(key))
        return [order]

    def positions(self) -> List[Position]:
        return [replace(position) for position in sorted(self._positions.values(), key=lambda p: (p.venue, p.instrument))]

    def closed_trades(self) -> List[ClosedTrade]:
        return list(self._closed)

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
            equity = _positive_or(asset.equity if asset else 0.0, (asset.cash + asset.used) if asset else 0.0, 1.0)
            price = _round_to_tick(position.last_price or position.entry_price, metadata.tick_size)
            notional_raw = abs(position.size) * equity * _positive_or(position.leverage, self._min_leverage(key), 1.0)
            quantity = _round_down_to_step(notional_raw / price, metadata.lot_size) if price > 0 else 0.0
            notional = quantity * price
            realized = position.realized_pnl * equity
            unrealized = position.unrealized_pnl * equity
            fees = position.fees * equity
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
                position.realized_pnl,
                position.unrealized_pnl,
                position.realized_pnl + position.unrealized_pnl,
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
        if self.config.min_expected_edge > 0 and edge < self.config.min_expected_edge:
            return []
        now = signal.timestamp or datetime.now(timezone.utc)
        target_size = target_sign * self.config.position_size * target_confidence
        min_delta = self._effective_min_order_delta()
        position = self._positions.get(key)
        if position is None:
            if abs(target_size) < min_delta:
                return []
            position = Position(signal.venue, signal.instrument, entry_price=signal.price, last_price=signal.price, opened_at=now)
            self._positions[key] = position
        else:
            is_flip = _sign(position.size) != 0 and _sign(position.size) != target_sign
            if not is_flip and self.config.rebalance_interval and position.last_signal_at:
                if now < position.last_signal_at + self.config.rebalance_interval:
                    return []
            if not is_flip and min_delta > 0 and abs(target_size - position.size) < min_delta:
                return []
        position.confidence = target_confidence
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
        position.leverage = self._select_leverage(key, target_confidence, edge, signal.score)
        return self._rebalance(
            {key: target_sign},
            {key: {"confidence": target_confidence, "score": signal.score, "edge": edge, "take_profit": signal.take_profit, "stop_loss": signal.stop_loss}},
        )

    def _rebalance(self, side_overrides: Dict[str, float], contexts: Dict[str, Dict[str, float]]) -> List[Order]:
        weights: Dict[str, float] = {}
        sides: Dict[str, float] = {}
        total_weight = 0.0
        for key, position in self._positions.items():
            has_override = key in side_overrides
            weight = _clamp01(position.confidence) or (0.0 if has_override else _confidence_from_size(position, self.config.position_size))
            side = side_overrides.get(key, _sign(position.size))
            weights[key] = weight
            sides[key] = side
            if weight > 1e-9 and side != 0:
                total_weight += weight
        used_budget = min(self.config.position_size, total_weight)
        orders: List[Order] = []
        for key, position in list(self._positions.items()):
            target_size = sides.get(key, 0.0) * used_budget * weights.get(key, 0.0) / total_weight if total_weight > 0 else 0.0
            delta = target_size - position.size
            if abs(delta) <= 1e-9:
                position.confidence = weights.get(key, 0.0)
                continue
            is_flip = abs(position.size) > 1e-9 and abs(target_size) > 1e-9 and _sign(position.size) != _sign(target_size)
            is_opening = abs(position.size) <= 1e-9 and abs(target_size) > 1e-9
            is_closing = abs(target_size) <= 1e-9 and abs(position.size) > 1e-9
            if not (is_flip or is_opening or is_closing) and abs(delta) < self._effective_min_order_delta():
                continue
            context = contexts.get(key, {})
            order = self._order_for_delta(
                key,
                position,
                delta,
                context.get("edge", 0.0),
                context.get("score", 0.0),
                _order_reason(position, target_size),
                context.get("confidence", position.confidence),
            )
            order.take_profit = context.get("take_profit", 0.0)
            order.stop_loss = context.get("stop_loss", 0.0)
            if not self._order_meets_minimum(order):
                continue
            orders.append(order)
            self._apply_delta(key, delta, position.last_price or position.entry_price, self._taker_fee_rate(key))
            if key in self._positions:
                self._positions[key].confidence = weights.get(key, 0.0)
        return orders

    def _order_for_delta(self, key: str, position: Position, delta: float, edge: float, score: float, reason: str, confidence: float) -> Order:
        fee_rate = self._taker_fee_rate(key)
        leverage = self._select_leverage(key, confidence, edge, score)
        metadata = self.instruments.instrument(position.venue, position.instrument)
        asset = self.assets.asset(metadata.settlement_currency)
        equity = _positive_or(asset.equity if asset else 0.0, (asset.cash + asset.used) if asset else 0.0, 1.0)
        price = _round_to_tick(position.last_price or position.entry_price, metadata.tick_size)
        notional = abs(delta) * equity * leverage
        quantity = _round_down_to_step(notional / price, metadata.lot_size) if price > 0 else 0.0
        notional = quantity * price
        return Order(
            venue=position.venue,
            instrument=position.instrument,
            side="sell" if delta < 0 else "buy",
            reason=reason,
            size_delta=delta,
            previous_size=position.size,
            target_size=position.size + delta,
            price=price,
            confidence=confidence,
            score=score,
            expected_edge=edge,
            fee_rate=fee_rate,
            estimated_fee=abs(delta) * fee_rate,
            leverage=leverage,
            estimated_fee_value=notional * fee_rate,
            quantity=quantity,
            notional=notional,
            settlement_currency=metadata.settlement_currency,
            min_size=metadata.min_size,
            lot_size=metadata.lot_size,
            tick_size=metadata.tick_size,
        )

    def _apply_delta(self, key: str, delta: float, price: float, fee_rate: float) -> None:
        position = self._positions.get(key)
        if position is None:
            return
        if position.size == 0 or _sign(position.size) == _sign(delta):
            next_abs = abs(position.size) + abs(delta)
            if price > 0:
                position.entry_price = (position.entry_price * abs(position.size) + price * abs(delta)) / next_abs if next_abs > 0 and abs(position.size) > 1e-9 and position.entry_price > 0 else price
                position.last_price = price
            fee = abs(delta) * fee_rate
            position.fees += fee
            position.realized_pnl -= fee
            position.size += delta
            return
        if price > 0:
            position.last_price = price
        closing = min(abs(position.size), abs(delta))
        gross = _move(position) * closing
        fee = closing * fee_rate
        position.realized_gross += gross
        position.fees += fee
        position.realized_pnl += gross - fee
        closed = ClosedTrade(position.venue, position.instrument, position.side or "buy", closing, position.entry_price, price, position.realized_gross, position.fees, position.realized_pnl, datetime.now(timezone.utc))
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
        position.fees = remaining * fee_rate
        position.realized_pnl = -position.fees

    def _effective_min_order_delta(self) -> float:
        return max(self.config.min_order_delta, 0.0) * max(self.config.position_size, 0.0)

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

    def _order_meets_minimum(self, order: Order) -> bool:
        if order.reason in {"closing", "flip"}:
            return True
        if order.min_size > 0 and order.quantity > 0 and order.quantity < order.min_size:
            return False
        if order.min_size > 0 and order.quantity <= 0:
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
        score=float(payload.get("score", 0.0)),
        components=[],
        timestamp=_parse_time(payload.get("timestamp")),
        price=float(payload.get("price", 0.0) or 0.0),
    )


def _normalize_config(config: PositionManagerConfig) -> PositionManagerConfig:
    config.position_size = min(max(config.position_size, 0.0), 1.0)
    config.min_expected_edge = max(config.min_expected_edge, 0.0)
    config.min_order_delta = min(max(config.min_order_delta, 0.0), 1.0)
    config.maker_fee_rate = max(config.maker_fee_rate, 0.0)
    config.taker_fee_rate = max(config.taker_fee_rate, 0.0)
    config.min_leverage = max(config.min_leverage, 0.0)
    config.max_leverage = max(config.max_leverage, 0.0)
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


def _move(position: Position) -> float:
    if position.entry_price <= 0 or position.last_price <= 0:
        return 0.0
    return (position.entry_price - position.last_price) / position.entry_price if position.size < 0 else (position.last_price - position.entry_price) / position.entry_price


def _order_reason(position: Position, target_size: float) -> str:
    if abs(position.size) <= 1e-9:
        return "opening"
    if abs(target_size) <= 1e-9:
        return "closing"
    if _sign(position.size) != _sign(target_size):
        return "flip"
    return "rebalance"


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
    "ErrorEvent",
    "SignalsEvent",
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
    "PositionStats",
    "InstrumentPositionStats",
    "CurrencyPositionStats",
    "parse_event",
    "production_position_manager_config",
]
