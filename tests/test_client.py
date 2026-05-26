import asyncio
import unittest

from grexie_signals_client import (
    AssetManager,
    AssetSnapshot,
    InstrumentManager,
    InstrumentMetadata,
    PositionManager,
    Position,
    ReadyEvent,
    Signal,
    SignalsClient,
    SignalEvent,
    InfoEvent,
    ErrorEvent,
    parse_event,
    production_position_manager_config,
)


def order_budget_cost(order):
    return abs(order.size_delta) + max(order.estimated_fee, 0.0)


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


class ClientTests(unittest.TestCase):
    def test_parse_signal_replay_event(self):
        event = parse_event(
            '{"type":"signal","subscriptionId":3,"venue":"okx","instrument":"BTC-USDT-SWAP","timestamp":"2026-05-26T00:00:00Z","replay":true,"signal":{"confidence":0.8,"side":"buy","takeProfit":0.01,"stopLoss":0.004}}'
        )
        self.assertIsInstance(event, SignalEvent)
        self.assertEqual(event.signal.venue, "okx")
        self.assertEqual(event.signal.instrument, "BTC-USDT-SWAP")
        self.assertTrue(event.replay)

    def test_parse_info_and_error_events(self):
        info = parse_event(
            '{"type":"info","subscriptionId":3,"venue":"okx","instrument":"DOGE-USDT-SWAP","stage":"ready","message":"ready","replay":true,"replayedAt":"2026-05-26T00:00:01Z"}'
        )
        self.assertIsInstance(info, InfoEvent)
        self.assertEqual(info.stage, "ready")
        self.assertTrue(info.replay)
        self.assertIsNotNone(info.replayed_at)

        error = parse_event('{"type":"error","code":"forbidden","message":"no access"}')
        self.assertIsInstance(error, ErrorEvent)
        self.assertEqual(error.code, "forbidden")

    def test_position_manager_opens_and_flips(self):
        manager = PositionManager(
            config=production_position_manager_config(
                position_size=0.10,
                min_expected_edge=0.0,
                min_order_delta=0.20,
                max_leverage=5.0,
            )
        )
        manager.instruments.update_instrument(InstrumentMetadata("okx", "BTC-USDT-SWAP"))
        buy = manager.handle_signal(
            Signal("okx", "BTC-USDT-SWAP", 0.8, "buy", 0.02, 0.004, price=100.0, score=0.5)
        )
        self.assertEqual(len(buy), 1)
        self.assertEqual(buy[0].reason, "opening")
        self.assertAlmostEqual(order_budget_cost(buy[0]), 0.10)

        sell = manager.handle_signal(
            Signal("okx", "BTC-USDT-SWAP", 0.9, "sell", 0.02, 0.004, price=99.0, score=-0.6)
        )
        self.assertEqual(len(sell), 1)
        self.assertEqual(sell[0].side, "sell")
        self.assertEqual(sell[0].reason, "flip")
        self.assertAlmostEqual(sell[0].target_size, 0.0)
        self.assertAlmostEqual(sell[0].size_delta, -buy[0].target_size)

        open_short = manager.handle_signal(
            Signal("okx", "BTC-USDT-SWAP", 0.9, "sell", 0.02, 0.004, price=99.0, score=-0.6)
        )
        self.assertEqual(len(open_short), 1)
        self.assertEqual(open_short[0].side, "sell")
        self.assertEqual(open_short[0].reason, "opening")

    def test_confidence_is_allocation_weight(self):
        manager = PositionManager(
            config=production_position_manager_config(
                position_size=0.10,
                min_expected_edge=0.0,
                min_order_delta=0.20,
            )
        )
        manager.instruments.update_instrument(InstrumentMetadata("okx", "DOGE-USDT-SWAP"))
        accepted = manager.handle_signal(
            Signal("okx", "DOGE-USDT-SWAP", 0.15, "buy", 0.02, 0.004, price=0.2)
        )
        self.assertEqual(len(accepted), 1)
        self.assertAlmostEqual(order_budget_cost(accepted[0]), 0.10)

    def test_quantizes_emitted_target_size_to_executable_lots(self):
        assets = AssetManager()
        assets.update_asset(AssetSnapshot("USDT", equity=1000, available=1000))
        instruments = InstrumentManager()
        instruments.update_instrument(
            InstrumentMetadata("okx", "BTC-USDT-SWAP", "USDT", lot_size=1, min_size=1, tick_size=0.1)
        )
        manager = PositionManager(
            config=production_position_manager_config(
                position_size=0.50,
                min_expected_edge=0.0,
                min_order_delta=0.0,
                asset_manager=assets,
                instrument_manager=instruments,
            )
        )
        orders = manager.handle_signal(
            Signal("okx", "BTC-USDT-SWAP", 0.15, "buy", 0.02, 0.004, price=333)
        )
        self.assertEqual(len(orders), 1)
        self.assertEqual(orders[0].quantity, 1)
        self.assertAlmostEqual(orders[0].size_delta, 0.333)
        self.assertAlmostEqual(orders[0].target_size, 0.333)

    def test_ignores_unconfigured_signals(self):
        manager = PositionManager(
            config=production_position_manager_config(
                position_size=0.10,
                min_expected_edge=0.0,
                min_order_delta=0.0,
            )
        )
        signal = Signal("okx", "SOL-USDT-SWAP", 1.0, "buy", 0.02, 0.004, price=100)
        self.assertEqual(manager.handle_signal(signal), [])
        self.assertEqual(manager.positions(), [])

        manager.instruments.update_instrument(InstrumentMetadata("okx", "SOL-USDT-SWAP"))
        self.assertEqual(len(manager.handle_signal(signal)), 1)

    def test_ignores_replay_signal_events(self):
        manager = PositionManager(
            config=production_position_manager_config(
                position_size=0.10,
                min_expected_edge=0.0,
                min_order_delta=0.0,
            )
        )
        manager.instruments.update_instrument(InstrumentMetadata("okx", "BTC-USDT-SWAP"))
        event = SignalEvent(
            "signal",
            3,
            "okx",
            "BTC-USDT-SWAP",
            Signal("okx", "BTC-USDT-SWAP", 1.0, "buy", 0.02, 0.004, price=100),
            replay=True,
        )
        self.assertEqual(manager.handle_event(event), [])
        self.assertEqual(manager.positions(), [])
        event.replay = False
        self.assertEqual(len(manager.handle_event(event)), 1)

    def test_leverage_adapts_by_confidence_edge_and_score_within_caps(self):
        def leverage_for(instrument: str, confidence: float, take_profit: float, score: float) -> float:
            manager = PositionManager(
                config=production_position_manager_config(
                    position_size=1.0,
                    min_expected_edge=0.0,
                    min_order_delta=0.0,
                    min_leverage=1.0,
                    max_leverage=5.0,
                )
            )
            manager.instruments.update_instrument(InstrumentMetadata("okx", instrument))
            orders = manager.handle_signal(
                Signal("okx", instrument, confidence, "buy", take_profit, 0.0, score=score, price=100)
            )
            return orders[0].leverage

        low = leverage_for("LOW-USDT-SWAP", 0.2, 0.0, 0.0)
        scored = leverage_for("SCORE-USDT-SWAP", 0.2, 0.0, 1.0)
        high = leverage_for("HIGH-USDT-SWAP", 1.0, 0.02, 1.0)
        self.assertGreaterEqual(low, 1.0)
        self.assertLessEqual(high, 5.0)
        self.assertGreater(scored, low)
        self.assertAlmostEqual(high, 5.0)

    def test_asset_and_instrument_managers_create_concrete_orders(self):
        assets = AssetManager()
        assets.update_asset(AssetSnapshot("USDT", cash=1000, available=900, used=100, equity=1000))
        instruments = InstrumentManager()
        instruments.update_instrument(
            InstrumentMetadata("okx", "BTC-USDT-SWAP", "USDT", lot_size=0.001, min_size=0.002, tick_size=0.1, max_leverage=2)
        )
        manager = PositionManager(
            config=production_position_manager_config(
                position_size=0.10,
                min_expected_edge=0.0,
                min_order_delta=0.0,
                min_leverage=1.0,
                max_leverage=5.0,
                asset_manager=assets,
                instrument_manager=instruments,
            )
        )
        orders = manager.handle_signal(
            Signal("okx", "BTC-USDT-SWAP", 1.0, "buy", 0.02, 0.004, price=100.07)
        )
        self.assertEqual(len(orders), 1)
        self.assertAlmostEqual(orders[0].price, 100.1)
        self.assertEqual(orders[0].settlement_currency, "USDT")
        self.assertLessEqual(orders[0].leverage, 2.0)
        self.assertGreater(orders[0].quantity, 0)
        self.assertGreater(orders[0].notional, 0)
        self.assertGreater(orders[0].estimated_fee_value, 0)

    def test_rejects_below_instrument_min_size(self):
        assets = AssetManager()
        assets.update_asset(AssetSnapshot("USDT", equity=10))
        instruments = InstrumentManager()
        instruments.update_instrument(
            InstrumentMetadata("okx", "BTC-USDT-SWAP", "USDT", lot_size=0.001, min_size=1, tick_size=0.1)
        )
        manager = PositionManager(
            config=production_position_manager_config(
                position_size=0.01,
                min_expected_edge=0.0,
                min_order_delta=0.0,
                asset_manager=assets,
                instrument_manager=instruments,
            )
        )
        orders = manager.handle_signal(
            Signal("okx", "BTC-USDT-SWAP", 1.0, "buy", 0.02, 0.004, price=100)
        )
        self.assertEqual(orders, [])

    def test_phases_reductions_before_openings(self):
        assets = AssetManager()
        assets.update_asset(AssetSnapshot("USDT", cash=1000, available=1000, equity=1000))
        instruments = InstrumentManager()
        instruments.update_instrument(InstrumentMetadata("okx", "BTC-USDT-SWAP", "USDT"))
        instruments.update_instrument(InstrumentMetadata("okx", "ETH-USDT-SWAP", "USDT"))
        manager = PositionManager(
            config=production_position_manager_config(
                position_size=0.20,
                min_expected_edge=0.0,
                min_order_delta=0.0,
                asset_manager=assets,
                instrument_manager=instruments,
            )
        )
        manager.add_position(
            Position("okx", "BTC-USDT-SWAP", size=0.15, confidence=1.0, entry_price=100, last_price=100)
        )
        reductions = manager.handle_signal(
            Signal("okx", "ETH-USDT-SWAP", 1.0, "buy", 0.02, 0.004, price=100)
        )
        self.assertEqual(len(reductions), 1)
        self.assertEqual(reductions[0].instrument, "BTC-USDT-SWAP")
        self.assertEqual(reductions[0].side, "sell")
        expected_btc_target = 0.10 / (1 + reductions[0].leverage * reductions[0].fee_rate)
        self.assertAlmostEqual(reductions[0].target_size, expected_btc_target)

        openings = manager.handle_signal(
            Signal("okx", "ETH-USDT-SWAP", 1.0, "buy", 0.02, 0.004, price=100)
        )
        self.assertEqual(len(openings), 1)
        self.assertEqual(openings[0].instrument, "ETH-USDT-SWAP")
        self.assertEqual(openings[0].side, "buy")

    def test_caps_openings_to_available_exposure(self):
        assets = AssetManager()
        assets.update_asset(AssetSnapshot("USDT", cash=1000, available=50, equity=1000))
        instruments = InstrumentManager()
        instruments.update_instrument(InstrumentMetadata("okx", "BTC-USDT-SWAP", "USDT"))
        manager = PositionManager(
            config=production_position_manager_config(
                position_size=0.20,
                min_expected_edge=0.0,
                min_order_delta=0.0,
                asset_manager=assets,
                instrument_manager=instruments,
            )
        )
        orders = manager.handle_signal(
            Signal("okx", "BTC-USDT-SWAP", 1.0, "buy", 0.02, 0.004, price=100)
        )
        self.assertEqual(len(orders), 1)
        self.assertLessEqual(order_budget_cost(orders[0]), 0.05 + 1e-9)
        self.assertLess(orders[0].size_delta, 0.05)

    def test_stats_by_instrument_and_currency(self):
        assets = AssetManager()
        assets.update_asset(AssetSnapshot("USDT", cash=1000, available=800, used=200, equity=1000))
        instruments = InstrumentManager()
        instruments.update_instrument(
            InstrumentMetadata("okx", "ETH-USDT-SWAP", "USDT", lot_size=0.01, min_size=0.01, tick_size=0.01)
        )
        manager = PositionManager(
            config=production_position_manager_config(asset_manager=assets, instrument_manager=instruments)
        )
        manager.add_position(
            Position("okx", "ETH-USDT-SWAP", size=0.10, confidence=0.8, entry_price=100, last_price=110, leverage=2, realized_pnl=0.01, fees=0.001)
        )
        stats = manager.stats()
        self.assertEqual(stats.equity, 1000)
        self.assertEqual(stats.available, 800)
        self.assertEqual(stats.by_instrument["okx:ETH-USDT-SWAP"].settlement_currency, "USDT")
        self.assertGreater(stats.by_instrument["okx:ETH-USDT-SWAP"].quantity, 0)
        self.assertGreater(stats.total_pnl_percent, 0)


if __name__ == "__main__":
    unittest.main()
