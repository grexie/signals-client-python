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
class BasketUpdatedEvent:
    type: Literal["basket_updated"]
    subscription_id: int
    venue: Optional[str] = None
    basket_id: Optional[str] = None
    message: Optional[str] = None


@dataclass
class OrderRouterForwardedEvent:
    type: Literal["order_router_forwarded"]
    subscription_id: int
    venue: Optional[str] = None
    basket_id: Optional[str] = None
    message: Optional[str] = None


@dataclass
class InfoEvent:
    type: Literal["info"]
    subscription_id: int
    venue: str
    instrument: str
    level: Literal["info", "error", "warn", "debug"]
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
    margin: float = 0.0
    confidence: float = 0.0


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
    BasketUpdatedEvent,
    OrderRouterForwardedEvent,
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
    margin: float = 0.0
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
class RiskConfig:
    """Router risk settings sent when subscribing to a basket.

    Args:
        max_margin_ratio: Fraction of account cash the router may reserve for active positions.
        min_lot_haircut_ratio: Extra cash buffer applied to lot margin and fees before orders are allowed.
        max_concurrent_positions: Optional cap on simultaneous active positions; zero leaves it unset.
        max_drawdown: Optional drawdown guard; zero leaves it unset.
        switch_buffer: Router score buffer required before switching instruments.
        min_leverage: Minimum leverage the router may request; zero leaves it unset.
        max_leverage: Maximum leverage the router may request; zero leaves it unset.
        profit_withdraw_ratio: Fraction of profits eligible for withdrawal events.
    """

    max_margin_ratio: float = 1.0
    min_lot_haircut_ratio: float = 0.0
    max_concurrent_positions: int = 0
    max_drawdown: float = 0.0
    switch_buffer: float = 0.0
    min_leverage: float = 0.0
    max_leverage: float = 0.0
    profit_withdraw_ratio: float = 0.0


@dataclass
class RuntimeConfig:
    """Router risk patch sent after a basket has subscribed.

    Args mirror :class:`RiskConfig`; zero numeric values mean "no change" for
    risk fields except ``profit_withdraw_ratio``, which is sent as the current
    desired ratio.
    """

    max_margin_ratio: float = 0.0
    min_lot_haircut_ratio: float = 0.0
    max_concurrent_positions: int = 0
    max_drawdown: float = 0.0
    switch_buffer: float = 0.0
    min_leverage: float = 0.0
    max_leverage: float = 0.0
    profit_withdraw_ratio: float = 0.0


RiskInput = Union[RiskConfig, Dict[str, Any]]
RuntimeInput = Union[RuntimeConfig, Dict[str, Any]]


@dataclass
class SignalsManagerConfig:
    """Configuration for one server-managed router basket.

    Args:
        venue: Venue code such as ``"okx"``.
        instruments: Basket instruments to subscribe and keep in sync.
        mode: Optional server-side router mode.
        risk: Initial router risk configuration, as :class:`RiskConfig` or a camelCase dict.
        profit_withdraw_ratio: Top-level profit withdrawal ratio sent on subscribe.
    """

    venue: str = "okx"
    instruments: List[str] = field(default_factory=list)
    mode: str = ""
    risk: RiskInput = field(default_factory=RiskConfig)
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
        """Open the websocket and start the background reader task."""

        import websockets

        headers = dict(self.headers)
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        self.websocket = await websockets.connect(self.url, extra_headers=headers)
        self._closed = False
        self._terminal_error = None
        self._reader_task = asyncio.create_task(self._read_loop())

    async def close(self) -> None:
        """Close the websocket and stop the background reader task."""

        if self._reader_task is not None:
            self._reader_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._reader_task
            self._reader_task = None
        if self.websocket is not None:
            await self.websocket.close()
            self.websocket = None

    async def subscribe(self, venue: str, instrument: str) -> None:
        """Subscribe to a legacy single-instrument stream.

        Args:
            venue: Venue code.
            instrument: Instrument symbol.
        """

        await self._send({"type": "subscribe", "venue": venue, "instrument": instrument})

    async def subscribe_basket(
        self,
        *,
        venue: str,
        instruments: List[str],
        mode: str = "",
        risk: Optional[RiskInput] = None,
        profit_withdraw_ratio: float = 0.0,
        assets: Optional[List[AssetSnapshot]] = None,
        positions: Optional[List[Position]] = None,
    ) -> None:
        """Subscribe to a server-managed router basket.

        Args:
            venue: Venue code.
            instruments: Basket instruments.
            mode: Optional router mode.
            risk: Initial router risk settings.
            profit_withdraw_ratio: Top-level profit withdrawal ratio.
            assets: Optional current account snapshots to hydrate the router.
            positions: Optional current position snapshots to hydrate the router.
        """

        payload: Dict[str, Any] = {"type": "subscribe", "venue": venue, "instruments": instruments}
        if mode:
            payload["mode"] = mode
        payload["risk"] = _risk_payload(risk or RiskConfig())
        if profit_withdraw_ratio > 0:
            payload["profitWithdrawRatio"] = profit_withdraw_ratio
        if assets:
            payload["assets"] = [_asset_payload(asset) for asset in assets]
        if positions:
            payload["positions"] = [_position_payload(position) for position in positions]
        await self._send(payload)

    async def update_asset(self, subscription_id: int, asset: AssetSnapshot) -> None:
        """Publish an account asset snapshot for a live subscription."""

        await self._send({"type": "update-asset", "subscriptionId": subscription_id, **_asset_payload(asset)})

    async def update_position(self, subscription_id: int, position: Position) -> None:
        """Publish a venue position snapshot for a live subscription."""

        await self._send({"type": "update-position", "subscriptionId": subscription_id, **_position_payload(position)})

    async def add_instrument(self, subscription_id: int, instrument: str) -> None:
        """Add an instrument to an existing basket subscription."""

        await self._send({"type": "add-instrument", "subscriptionId": subscription_id, "instrument": instrument})

    async def remove_instrument(self, subscription_id: int, instrument: str) -> None:
        """Remove an instrument from an existing basket subscription."""

        await self._send({"type": "remove-instrument", "subscriptionId": subscription_id, "instrument": instrument})

    async def update_config(self, subscription_id: int, config: Optional[RuntimeInput] = None, **updates: Any) -> None:
        """Send a runtime router config patch.

        Args:
            subscription_id: Server subscription id.
            config: RuntimeConfig or dict containing camelCase/snake_case risk fields.
            **updates: Keyword form of runtime config fields when config is omitted.
        """

        runtime = _runtime_payload(config if config is not None else updates)
        await self._send({"type": "update-config", "subscriptionId": subscription_id, **runtime})

    async def schedule_withdrawal(self, subscription_id: int, *, currency: str, amount: float, venue: str = "", reason: str = "") -> None:
        """Schedule a withdrawal request for the router subscription."""

        await self._send({"type": "schedule-withdrawal", "subscriptionId": subscription_id, "venue": venue, "currency": currency, "amount": amount, "reason": reason})

    async def unsubscribe(self, subscription_id: int) -> None:
        """Unsubscribe by server subscription id."""

        await self._send({"type": "unsubscribe", "subscriptionId": subscription_id})

    async def unsubscribe_instrument(self, venue: str, instrument: str) -> None:
        """Unsubscribe a legacy single-instrument stream."""

        await self._send({"type": "unsubscribe", "venue": venue, "instrument": instrument})

    async def receive(self) -> SignalsEvent:
        """Wait for the next typed websocket event."""

        if self.websocket is None and self._reader_task is None:
            raise RuntimeError("signals client is not connected")
        item = await self._receive_queue.get()
        if item is None:
            raise RuntimeError("signals client is closed")
        if isinstance(item, BaseException):
            raise item
        return item

    async def events(self) -> AsyncIterator[SignalsEvent]:
        """Yield an independent stream of typed websocket events."""

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
                raw = await self.websocket.recv()
                if _ignored_websocket_message(raw):
                    continue
                await self._publish(parse_event(raw))
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
        """Subscribe, process websocket events until the client stream ends, then unsubscribe."""

        await self.subscribe()
        try:
            async for event in self.client.events():
                await self.handle_event(event)
        finally:
            if self.subscription_id > 0:
                await self.client.unsubscribe(self.subscription_id)
                self.subscription_id = 0

    async def subscribe(self) -> None:
        """Subscribe the configured basket and send current assets/positions."""

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
        """Record and, once subscribed, send an account asset snapshot."""

        next_asset = self._record_asset(asset)
        if next_asset is not None and self.subscription_id > 0:
            await self.client.update_asset(self.subscription_id, next_asset)

    async def update_position(self, position: Position) -> None:
        """Record and, once subscribed, send a venue position snapshot."""

        next_position = self._record_position(position)
        if next_position is not None and self.subscription_id > 0:
            await self.client.update_position(self.subscription_id, next_position)

    async def add_instrument(self, instrument: str) -> None:
        """Add an instrument locally and to the live subscription."""

        instrument = _normalize_instrument(instrument)
        if not instrument:
            return
        self.config.instruments = _normalize_instrument_list([*self.config.instruments, instrument])
        if self.subscription_id > 0:
            await self.client.add_instrument(self.subscription_id, instrument)

    async def remove_instrument(self, instrument: str) -> None:
        """Remove an instrument locally and from the live subscription."""

        instrument = _normalize_instrument(instrument)
        self.config.instruments = [current for current in self.config.instruments if current != instrument]
        if self.subscription_id > 0:
            await self.client.remove_instrument(self.subscription_id, instrument)

    async def update_config(self, config: Optional[RuntimeInput] = None, **updates: Any) -> None:
        """Apply and optionally send a runtime router config patch."""

        runtime = _normalize_runtime_config(config if config is not None else updates)
        self.config.risk = _apply_runtime_to_risk(self.config.risk, runtime)
        self.config.profit_withdraw_ratio = runtime.profit_withdraw_ratio
        if self.subscription_id > 0:
            await self.client.update_config(self.subscription_id, runtime)

    async def schedule_withdrawal(self, *, currency: str, amount: float, venue: str = "", reason: str = "") -> None:
        """Schedule a withdrawal through the live router subscription."""

        if self.subscription_id <= 0:
            raise RuntimeError("signals manager basket is not subscribed")
        await self.client.schedule_withdrawal(self.subscription_id, currency=currency.upper(), amount=amount, venue=_normalize_venue(venue or self.config.venue), reason=reason)

    async def handle_event(self, event: SignalsEvent) -> None:
        """Apply one typed websocket event to the manager."""

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
        """Return current asset snapshots sorted by currency."""

        return sorted(self._assets.values(), key=lambda asset: asset.currency)

    def positions(self) -> List[Position]:
        """Return current open position snapshots sorted by venue/instrument."""

        return sorted(self._positions.values(), key=lambda position: _position_key(position.venue, position.instrument))

    def state(self) -> SignalsManagerState:
        """Return durable state suitable for restart hydration."""

        return SignalsManagerState(self.assets(), self.positions())

    def available_order_cash(self, currency: str) -> float:
        """Return available cash after applying the asset max_usage cap."""

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
    """Parse one raw websocket message into a typed event."""

    msg = json.loads(raw.decode() if isinstance(raw, bytes) else raw) if not isinstance(raw, dict) else raw
    event_type = msg.get("type")
    if event_type == "ready":
        return ReadyEvent("ready", msg.get("message", ""))
    if event_type == "subscribed":
        return SubscribedEvent("subscribed", int(msg.get("subscriptionId", 0)), msg.get("venue", ""), msg.get("instrument", ""))
    if event_type == "unsubscribed":
        return UnsubscribedEvent("unsubscribed", msg.get("subscriptionId"), msg.get("venue"), msg.get("instrument"), msg.get("code"), msg.get("message"))
    if event_type == "basket_updated":
        return BasketUpdatedEvent("basket_updated", int(msg.get("subscriptionId", 0)), msg.get("venue"), msg.get("basketId"), msg.get("message"))
    if event_type == "order_router_forwarded":
        return OrderRouterForwardedEvent("order_router_forwarded", int(msg.get("subscriptionId", 0)), msg.get("venue"), msg.get("basketId"), msg.get("message"))
    if event_type == "info":
        return InfoEvent("info", int(msg.get("subscriptionId", 0)), msg.get("venue", ""), msg.get("instrument", ""), _normalize_info_level(msg.get("level")), msg.get("stage", ""), msg.get("message", ""), _parse_time(msg.get("timestamp")), bool(msg.get("replay", False)), _parse_time(msg.get("replayedAt")))
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
        return CreateMarketOrderEvent(
            "create-market-order",
            int(msg.get("subscriptionId", 0)),
            msg.get("intentId"),
            msg.get("action"),
            msg.get("reason"),
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
            float(msg.get("margin", 0.0) or 0.0),
            float(msg.get("confidence", 0.0) or 0.0),
        )
    if event_type == "update-tpsl":
        return UpdateTPSLEvent("update-tpsl", int(msg.get("subscriptionId", 0)), msg.get("intentId"), msg.get("venue"), msg.get("instrument", ""), msg.get("side", ""), float(msg.get("takeProfitPrice", 0.0) or 0.0), float(msg.get("stopLossPrice", 0.0) or 0.0), float(msg.get("takeProfit", 0.0) or 0.0), float(msg.get("stopLoss", 0.0) or 0.0), _parse_time(msg.get("timestamp")))
    if event_type == "withdraw":
        return WithdrawEvent("withdraw", int(msg.get("subscriptionId", 0)), msg.get("intentId"), msg.get("venue"), msg.get("currency", ""), float(msg.get("amount", 0.0) or 0.0), _parse_time(msg.get("timestamp")))
    if event_type == "error":
        return ErrorEvent("error", msg.get("code"), msg.get("message"))
    raise ValueError(f"unsupported websocket event type {event_type!r}")


def _ignored_websocket_message(raw: Union[str, bytes, Dict[str, Any]]) -> bool:
    try:
        msg = json.loads(raw.decode() if isinstance(raw, bytes) else raw) if not isinstance(raw, dict) else raw
    except Exception:
        return False
    return msg.get("type") == "basket_state"


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
        "margin": position.margin,
        "leverage": position.leverage,
        "takeProfitPrice": position.take_profit_price,
        "stopLossPrice": position.stop_loss_price,
    }


def _risk_payload(risk: RiskInput) -> Dict[str, Any]:
    normalized = _normalize_risk_config(risk)
    return {
        "maxMarginRatio": normalized.max_margin_ratio,
        "minLotHaircutRatio": normalized.min_lot_haircut_ratio,
        "maxConcurrentPositions": normalized.max_concurrent_positions,
        "maxDrawdown": normalized.max_drawdown,
        "switchBuffer": normalized.switch_buffer,
        "minLeverage": normalized.min_leverage,
        "maxLeverage": normalized.max_leverage,
        "profitWithdrawRatio": normalized.profit_withdraw_ratio,
    }


def _runtime_payload(config: RuntimeInput) -> Dict[str, Any]:
    normalized = _normalize_runtime_config(config)
    return {
        "maxMarginRatio": normalized.max_margin_ratio,
        "minLotHaircutRatio": normalized.min_lot_haircut_ratio,
        "maxConcurrentPositions": normalized.max_concurrent_positions,
        "maxDrawdown": normalized.max_drawdown,
        "switchBuffer": normalized.switch_buffer,
        "minLeverage": normalized.min_leverage,
        "maxLeverage": normalized.max_leverage,
        "profitWithdrawRatio": normalized.profit_withdraw_ratio,
    }


def _normalize_risk_config(risk: RiskInput) -> RiskConfig:
    if isinstance(risk, RiskConfig):
        value = risk
    else:
        value = RiskConfig(
            max_margin_ratio=float(risk.get("maxMarginRatio", risk.get("max_margin_ratio", 1.0)) or 0.0),
            min_lot_haircut_ratio=float(risk.get("minLotHaircutRatio", risk.get("min_lot_haircut_ratio", 0.0)) or 0.0),
            max_concurrent_positions=int(risk.get("maxConcurrentPositions", risk.get("max_concurrent_positions", 0)) or 0),
            max_drawdown=float(risk.get("maxDrawdown", risk.get("max_drawdown", 0.0)) or 0.0),
            switch_buffer=float(risk.get("switchBuffer", risk.get("switch_buffer", 0.0)) or 0.0),
            min_leverage=float(risk.get("minLeverage", risk.get("min_leverage", 0.0)) or 0.0),
            max_leverage=float(risk.get("maxLeverage", risk.get("max_leverage", 0.0)) or 0.0),
            profit_withdraw_ratio=float(risk.get("profitWithdrawRatio", risk.get("profit_withdraw_ratio", 0.0)) or 0.0),
        )
    max_leverage = max(0.0, _finite_or(value.max_leverage, 0.0))
    min_leverage = max(0.0, _finite_or(value.min_leverage, 0.0))
    if max_leverage > 0 and min_leverage > max_leverage:
        min_leverage = max_leverage
    return RiskConfig(
        max_margin_ratio=_clamp01(_positive_or(_finite_or(value.max_margin_ratio, 0.0), 1.0)),
        min_lot_haircut_ratio=max(0.0, _finite_or(value.min_lot_haircut_ratio, 0.0)),
        max_concurrent_positions=max(0, int(value.max_concurrent_positions)),
        max_drawdown=max(0.0, _finite_or(value.max_drawdown, 0.0)),
        switch_buffer=max(0.0, _finite_or(value.switch_buffer, 0.0)),
        min_leverage=min_leverage,
        max_leverage=max_leverage,
        profit_withdraw_ratio=_clamp01(value.profit_withdraw_ratio),
    )


def _normalize_runtime_config(config: RuntimeInput) -> RuntimeConfig:
    if isinstance(config, RuntimeConfig):
        value = config
    else:
        value = RuntimeConfig(
            max_margin_ratio=float(config.get("maxMarginRatio", config.get("max_margin_ratio", 0.0)) or 0.0),
            min_lot_haircut_ratio=float(config.get("minLotHaircutRatio", config.get("min_lot_haircut_ratio", 0.0)) or 0.0),
            max_concurrent_positions=int(config.get("maxConcurrentPositions", config.get("max_concurrent_positions", 0)) or 0),
            max_drawdown=float(config.get("maxDrawdown", config.get("max_drawdown", 0.0)) or 0.0),
            switch_buffer=float(config.get("switchBuffer", config.get("switch_buffer", 0.0)) or 0.0),
            min_leverage=float(config.get("minLeverage", config.get("min_leverage", 0.0)) or 0.0),
            max_leverage=float(config.get("maxLeverage", config.get("max_leverage", 0.0)) or 0.0),
            profit_withdraw_ratio=float(config.get("profitWithdrawRatio", config.get("profit_withdraw_ratio", 0.0)) or 0.0),
        )
    max_leverage = max(0.0, _finite_or(value.max_leverage, 0.0))
    min_leverage = max(0.0, _finite_or(value.min_leverage, 0.0))
    if max_leverage > 0 and min_leverage > max_leverage:
        min_leverage = max_leverage
    return RuntimeConfig(
        max_margin_ratio=_clamp01(value.max_margin_ratio),
        min_lot_haircut_ratio=max(0.0, _finite_or(value.min_lot_haircut_ratio, 0.0)),
        max_concurrent_positions=max(0, int(value.max_concurrent_positions)),
        max_drawdown=max(0.0, _finite_or(value.max_drawdown, 0.0)),
        switch_buffer=max(0.0, _finite_or(value.switch_buffer, 0.0)),
        min_leverage=min_leverage,
        max_leverage=max_leverage,
        profit_withdraw_ratio=_clamp01(value.profit_withdraw_ratio),
    )


def _apply_runtime_to_risk(risk: RiskInput, config: RuntimeConfig) -> RiskConfig:
    normalized = _normalize_risk_config(risk)
    return _normalize_risk_config(
        RiskConfig(
            max_margin_ratio=config.max_margin_ratio if config.max_margin_ratio > 0 else normalized.max_margin_ratio,
            min_lot_haircut_ratio=config.min_lot_haircut_ratio if config.min_lot_haircut_ratio > 0 else normalized.min_lot_haircut_ratio,
            max_concurrent_positions=config.max_concurrent_positions if config.max_concurrent_positions > 0 else normalized.max_concurrent_positions,
            max_drawdown=config.max_drawdown if config.max_drawdown > 0 else normalized.max_drawdown,
            switch_buffer=config.switch_buffer if config.switch_buffer > 0 else normalized.switch_buffer,
            min_leverage=config.min_leverage if config.min_leverage > 0 else normalized.min_leverage,
            max_leverage=config.max_leverage if config.max_leverage > 0 else normalized.max_leverage,
            profit_withdraw_ratio=config.profit_withdraw_ratio,
        )
    )


def _normalize_manager_config(config: SignalsManagerConfig) -> SignalsManagerConfig:
    config.venue = _normalize_venue(config.venue)
    config.instruments = _normalize_instrument_list(config.instruments)
    config.risk = _normalize_risk_config(config.risk)
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


def _finite_or(value: float, fallback: float) -> float:
    return value if value == value and value not in (float("inf"), float("-inf")) else fallback


def _normalize_info_level(value: Any) -> Literal["info", "error", "warn", "debug"]:
    level = str(value or "").strip().lower()
    if level == "error":
        return "error"
    if level == "warn":
        return "warn"
    if level == "debug":
        return "debug"
    return "info"


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
    "RiskConfig",
    "RuntimeConfig",
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
