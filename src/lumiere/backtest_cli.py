from __future__ import annotations

import argparse
import asyncio
import json
from decimal import Decimal

from lumiere.backtest import BacktestConfig, Backtester, CostModel
from lumiere.config import SUPPORTED_INST_IDS
from lumiere.historical_data import HistoricalCandleRequest, OKXHistoricalDataClient
from lumiere.strategy import MovingAverageCrossoverConfig, MovingAverageCrossoverStrategy


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a Lumiere OKX historical backtest")
    parser.add_argument("--inst-id", action="append", choices=SUPPORTED_INST_IDS, dest="inst_ids")
    parser.add_argument("--bar", default="1m")
    parser.add_argument("--limit", type=int, default=300)
    parser.add_argument("--starting-equity-usdt", default="1000")
    parser.add_argument("--fast-window", type=int, default=5)
    parser.add_argument("--slow-window", type=int, default=20)
    parser.add_argument("--trade-size-btc", default="0.001")
    parser.add_argument("--trade-size-eth", default="0.01")
    parser.add_argument("--taker-fee-bps", default="10")
    parser.add_argument("--spread-bps", default="2")
    parser.add_argument("--slippage-bps", default="5")
    parser.add_argument("--market-impact-bps", default="0")
    parser.add_argument("--reject-every-n-orders", type=int, default=0)
    return parser


async def run_backtest(args: argparse.Namespace) -> list[dict]:
    inst_ids = tuple(args.inst_ids or SUPPORTED_INST_IDS)
    data_client = OKXHistoricalDataClient(flag="1")
    cost_model = CostModel(
        taker_fee_bps=Decimal(args.taker_fee_bps),
        spread_bps=Decimal(args.spread_bps),
        slippage_bps=Decimal(args.slippage_bps),
        market_impact_bps=Decimal(args.market_impact_bps),
        reject_every_n_orders=args.reject_every_n_orders,
    )
    reports = []
    for inst_id in inst_ids:
        candles = await data_client.fetch_candles(
            HistoricalCandleRequest(inst_id=inst_id, bar=args.bar, limit=args.limit)
        )
        configured_size = args.trade_size_eth if inst_id.startswith("ETH-") else args.trade_size_btc
        trade_size = Decimal(configured_size)
        strategy = MovingAverageCrossoverStrategy(
            MovingAverageCrossoverConfig(
                inst_id=inst_id,
                fast_window=args.fast_window,
                slow_window=args.slow_window,
                trade_size_btc=trade_size,
            )
        )
        report = Backtester(
            strategy,
            BacktestConfig(
                starting_equity_usdt=Decimal(args.starting_equity_usdt),
                cost_model=cost_model,
            ),
        ).run(candles)
        reports.append(report.to_dict())
    return reports


def main() -> None:
    args = build_parser().parse_args()
    reports = asyncio.run(run_backtest(args))
    print(json.dumps({"reports": reports}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
