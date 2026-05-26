import unittest

from grexie_signals_client import (
    AssetManager,
    AssetSnapshot,
    InstrumentManager,
    InstrumentMetadata,
    PositionManager,
    Position,
    Signal,
    SignalEvent,
    InfoEvent,
    ErrorEvent,
    parse_event,
    production_position_manager_config,
)


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
        buy = manager.handle_signal(
            Signal("okx", "BTC-USDT-SWAP", 0.8, "buy", 0.02, 0.004, price=100.0, score=0.5)
        )
        self.assertEqual(len(buy), 1)
        self.assertEqual(buy[0].reason, "opening")
        self.assertAlmostEqual(buy[0].target_size, 0.10)

        sell = manager.handle_signal(
            Signal("okx", "BTC-USDT-SWAP", 0.9, "sell", 0.02, 0.004, price=99.0, score=-0.6)
        )
        self.assertEqual(len(sell), 1)
        self.assertEqual(sell[0].side, "sell")
        self.assertEqual(sell[0].reason, "flip")
        self.assertLess(sell[0].size_delta, -0.19)

    def test_min_delta_scales_to_position_size(self):
        manager = PositionManager(
            config=production_position_manager_config(
                position_size=0.10,
                min_expected_edge=0.0,
                min_order_delta=0.20,
            )
        )
        rejected = manager.handle_signal(
            Signal("okx", "DOGE-USDT-SWAP", 0.15, "buy", 0.02, 0.004, price=0.2)
        )
        accepted = manager.handle_signal(
            Signal("okx", "DOGE-USDT-SWAP", 0.25, "buy", 0.02, 0.004, price=0.2)
        )
        self.assertEqual(rejected, [])
        self.assertEqual(len(accepted), 1)

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
