import asyncio
import unittest

from grexie_signals_client import (
    AssetSnapshot,
    BacktestEvent,
    CreateMarketOrderEvent,
    ErrorEvent,
    InfoEvent,
    Position,
    ReadyEvent,
    SignalEvent,
    SignalsClient,
    SignalsManager,
    SignalsManagerConfig,
    SignalsManagerState,
    SubscribedEvent,
    UpdateTPSLEvent,
    WithdrawEvent,
    parse_event,
)


class AsyncClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_events_fan_out_to_independent_consumers(self):
        client = SignalsClient("")
        first = client.events()
        second = client.events()
        first_event = asyncio.create_task(first.__anext__())
        second_event = asyncio.create_task(second.__anext__())
        while len(client._subscriber_queues) < 2:
            await asyncio.sleep(0)

        await client._publish(ReadyEvent("ready", "ok"))

        self.assertEqual((await first_event).type, "ready")
        self.assertEqual((await second_event).type, "ready")
        await first.aclose()
        await second.aclose()

    async def test_signals_manager_subscribes_with_state_and_emits_intents(self):
        client = FakeClient(
            [
                SubscribedEvent("subscribed", 9, "okx", "BTC-USDT-SWAP"),
                CreateMarketOrderEvent("create-market-order", 9, "intent_1", "open_position", "entry", "okx", "BTC-USDT-SWAP", "buy", contract_size=3),
            ]
        )
        manager = SignalsManager(
            client,
            SignalsManagerState(
                assets=[AssetSnapshot("USDT", venue="okx", available=100, equity=100)],
                positions=[Position("BTC-USDT-SWAP", 2, venue="okx", entry_price=100, last_price=101)],
            ),
            SignalsManagerConfig(venue="okx", instruments=["BTC-USDT-SWAP"]),
        )

        await manager.run()

        self.assertEqual(client.sent[0]["type"], "subscribe")
        self.assertEqual(client.sent[0]["assets"][0].currency, "USDT")
        self.assertEqual(client.sent[0]["positions"][0].instrument, "BTC-USDT-SWAP")
        self.assertTrue(any(item["type"] == "update-asset" and item["subscriptionId"] == 9 for item in client.sent))
        self.assertTrue(any(item["type"] == "update-position" and item["subscriptionId"] == 9 for item in client.sent))
        intent = await manager.intents.get()
        self.assertEqual(intent.intent_id, "intent_1")
        self.assertEqual(intent.contract_size, 3)

    async def test_signals_manager_updates_snapshots_after_subscription(self):
        client = FakeClient([])
        manager = SignalsManager(client, config=SignalsManagerConfig(venue="okx", instruments=["ETH-USDT-SWAP"]))
        await manager.handle_event(SubscribedEvent("subscribed", 15, "okx", "ETH-USDT-SWAP"))

        await manager.update_asset(AssetSnapshot("usdt", available=50, max_usage=0.5))
        await manager.update_position(Position("ETH-USDT-SWAP", -4, entry_price=2000))

        self.assertEqual(manager.available_order_cash("USDT"), 25)
        self.assertEqual(manager.state().positions[0].status, "open")
        self.assertTrue(any(item["type"] == "update-asset" and item["currency"] == "USDT" for item in client.sent))
        self.assertTrue(any(item["type"] == "update-position" and item["side"] == "sell" and item["size"] == 4 for item in client.sent))


class ClientTests(unittest.TestCase):
    def test_parse_signal_replay_event(self):
        event = parse_event(
            '{"type":"signal","subscriptionId":3,"venue":"okx","instrument":"BTC-USDT-SWAP","timestamp":"2026-05-26T00:00:00Z","replay":true,"signal":{"confidence":0.8,"side":"buy","takeProfit":0.01,"stopLoss":0.004,"trailingStopActivation":0.02,"trailingStopDistance":0.01,"trailingStopMinProfit":0.001,"managePositionsOnly":true}}'
        )
        self.assertIsInstance(event, SignalEvent)
        self.assertEqual(event.signal.venue, "okx")
        self.assertEqual(event.signal.instrument, "BTC-USDT-SWAP")
        self.assertAlmostEqual(event.signal.trailing_stop_activation, 0.02)
        self.assertAlmostEqual(event.signal.trailing_stop_distance, 0.01)
        self.assertAlmostEqual(event.signal.trailing_stop_min_profit, 0.001)
        self.assertTrue(event.signal.manage_positions_only)
        self.assertTrue(event.replay)

    def test_parse_info_and_error_events(self):
        info = parse_event(
            '{"type":"info","subscriptionId":3,"venue":"okx","instrument":"DOGE-USDT-SWAP","stage":"ready","message":"ready","replay":true,"replayedAt":"2026-05-26T00:00:01Z"}'
        )
        self.assertIsInstance(info, InfoEvent)
        self.assertEqual(info.stage, "ready")
        self.assertTrue(info.replay)
        self.assertIsNotNone(info.replayed_at)

        backtest = parse_event(
            '{"type":"backtest","subscriptionId":3,"venue":"okx","instrument":"BASKET:1","timestamp":"2026-05-31T17:00:00Z","backtest":{"accepted":true,"candidate":{"total":0.12}}}'
        )
        self.assertIsInstance(backtest, BacktestEvent)
        self.assertTrue(backtest.backtest["accepted"])

        error = parse_event('{"type":"error","code":"forbidden","message":"no access"}')
        self.assertIsInstance(error, ErrorEvent)
        self.assertEqual(error.code, "forbidden")

    def test_parse_order_router_events(self):
        order = parse_event(
            '{"type":"create-market-order","subscriptionId":12,"intentId":"intent_1","reason":"preempted_by_better_route","venue":"okx","instrument":"BTC-USDT-SWAP","side":"buy","contractSize":3}'
        )
        self.assertIsInstance(order, CreateMarketOrderEvent)
        self.assertEqual(order.reason, "preempted_by_better_route")

        tpsl = parse_event(
            '{"type":"update-tpsl","subscriptionId":12,"intentId":"intent_2","venue":"okx","instrument":"BTC-USDT-SWAP","side":"buy","takeProfitPrice":72100,"stopLossPrice":70050,"takeProfit":0.03,"stopLoss":0.0007}'
        )
        self.assertIsInstance(tpsl, UpdateTPSLEvent)
        self.assertEqual(tpsl.take_profit_price, 72100)
        self.assertEqual(tpsl.stop_loss_price, 70050)

        withdraw = parse_event(
            '{"type":"withdraw","subscriptionId":12,"intentId":"withdraw_1","venue":"okx","currency":"USDT","amount":42}'
        )
        self.assertIsInstance(withdraw, WithdrawEvent)
        self.assertEqual(withdraw.currency, "USDT")
        self.assertEqual(withdraw.amount, 42)


class FakeClient:
    def __init__(self, events):
        self._events = list(events)
        self.sent = []

    async def events(self):
        for event in self._events:
            yield event

    async def subscribe_basket(self, **request):
        self.sent.append({"type": "subscribe", **request})

    async def unsubscribe(self, subscription_id):
        self.sent.append({"type": "unsubscribe", "subscriptionId": subscription_id})

    async def update_asset(self, subscription_id, asset):
        self.sent.append({"type": "update-asset", "subscriptionId": subscription_id, "currency": asset.currency})

    async def update_position(self, subscription_id, position):
        self.sent.append({"type": "update-position", "subscriptionId": subscription_id, "instrument": position.instrument, "side": position.side, "size": abs(position.size)})

    async def add_instrument(self, subscription_id, instrument):
        self.sent.append({"type": "add-instrument", "subscriptionId": subscription_id, "instrument": instrument})

    async def remove_instrument(self, subscription_id, instrument):
        self.sent.append({"type": "remove-instrument", "subscriptionId": subscription_id, "instrument": instrument})

    async def update_config(self, subscription_id, *, profit_withdraw_ratio=0):
        self.sent.append({"type": "update-config", "subscriptionId": subscription_id, "profitWithdrawRatio": profit_withdraw_ratio})

    async def schedule_withdrawal(self, subscription_id, *, currency, amount, venue="", reason=""):
        self.sent.append({"type": "schedule-withdrawal", "subscriptionId": subscription_id, "currency": currency, "amount": amount, "venue": venue, "reason": reason})


if __name__ == "__main__":
    unittest.main()
