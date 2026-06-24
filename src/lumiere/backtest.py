from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
from decimal import Decimal
from typing import Any

from lumiere.ledger import (
    BASIS_POINTS,
    EquityPoint,
    PnlMetrics,
    TradeFill,
    build_pnl_metrics,
    max_drawdown_usdt,
    risk_adjusted_ratios,
)
from lumiere.models import AccountSnapshot, DecisionAction, MarketCandle, Position
from lumiere.strategy import TradingStrategy


@dataclass(frozen=True, slots=True)
class CostModel:
    """Deterministic market-order cost assumptions for backtests."""

    taker_fee_bps: Decimal = Decimal("10")
    spread_bps: Decimal = Decimal("2")
    slippage_bps: Decimal = Decimal("5")
    market_impact_bps: Decimal = Decimal("0")
    reject_every_n_orders: int = 0

    def __post_init__(self) -> None:
        if self.taker_fee_bps < 0:
            raise ValueError("taker_fee_bps cannot be negative")
        if self.spread_bps < 0:
            raise ValueError("spread_bps cannot be negative")
        if self.slippage_bps < 0:
            raise ValueError("slippage_bps cannot be negative")
        if self.market_impact_bps < 0:
            raise ValueError("market_impact_bps cannot be negative")
        if self.reject_every_n_orders < 0:
            raise ValueError("reject_every_n_orders cannot be negative")

    @property
    def order_cost_bps(self) -> Decimal:
        return self.spread_bps / Decimal("2") + self.slippage_bps + self.market_impact_bps

    def execution_price(self, side: DecisionAction, mid_price: Decimal) -> Decimal:
        adjustment = self.order_cost_bps / BASIS_POINTS
        if side is DecisionAction.BUY:
            return mid_price * (Decimal("1") + adjustment)
        if side is DecisionAction.SELL:
            return mid_price * (Decimal("1") - adjustment)
        raise ValueError("execution_price requires buy or sell side")

    def fee_usdt(self, notional_usdt: Decimal) -> Decimal:
        return abs(notional_usdt) * self.taker_fee_bps / BASIS_POINTS

    def is_rejected(self, order_number: int) -> bool:
        return self.reject_every_n_orders > 0 and order_number % self.reject_every_n_orders == 0

    def assumptions(self) -> dict[str, str | int]:
        return {
            "taker_fee_bps": str(self.taker_fee_bps),
            "spread_bps": str(self.spread_bps),
            "slippage_bps": str(self.slippage_bps),
            "market_impact_bps": str(self.market_impact_bps),
            "reject_every_n_orders": self.reject_every_n_orders,
        }


@dataclass(frozen=True, slots=True)
class BacktestConfig:
    starting_equity_usdt: Decimal = Decimal("1000")
    cost_model: CostModel = CostModel()

    def __post_init__(self) -> None:
        if self.starting_equity_usdt <= 0:
            raise ValueError("starting_equity_usdt must be positive")


@dataclass(frozen=True, slots=True)
class BacktestReport:
    inst_id: str
    strategy_name: str
    parameters: dict[str, str | int]
    period_start: datetime
    period_end: datetime
    metrics: PnlMetrics
    buy_and_hold_pnl_usdt: Decimal
    no_trade_pnl_usdt: Decimal
    rejected_order_count: int
    assumptions: dict[str, str | int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "inst_id": self.inst_id,
            "strategy_name": self.strategy_name,
            "parameters": self.parameters,
            "period_start": self.period_start.isoformat(),
            "period_end": self.period_end.isoformat(),
            "metrics": self.metrics.to_dict(),
            "buy_and_hold_pnl_usdt": str(self.buy_and_hold_pnl_usdt),
            "no_trade_pnl_usdt": str(self.no_trade_pnl_usdt),
            "rejected_order_count": self.rejected_order_count,
            "assumptions": self.assumptions,
        }


class Backtester:
    def __init__(
        self,
        strategy: TradingStrategy,
        config: BacktestConfig | None = None,
    ) -> None:
        self.strategy = strategy
        self.config = config or BacktestConfig()

    def run(self, candles: list[MarketCandle] | tuple[MarketCandle, ...]) -> BacktestReport:
        ordered_candles = tuple(sorted(candles, key=lambda candle: candle.ts))
        if not ordered_candles:
            raise ValueError("at least one candle is required")

        cash = self.config.starting_equity_usdt
        position = Decimal("0")
        fills: list[TradeFill] = []
        equity_curve: list[EquityPoint] = []
        attempted_orders = 0
        rejected_orders = 0
        cost_model = self.config.cost_model
        inst_id = self.strategy.config.inst_id

        for index, candle in enumerate(ordered_candles):
            account = AccountSnapshot(
                equity_usdt=cash + position * candle.close,
                available_usdt=cash,
                positions=((Position(inst_id=inst_id, size_btc=position),) if position > 0 else ()),
                performance_gate_passed=True,
            )
            decision = self.strategy.decide(list(ordered_candles[: index + 1]), account)
            if decision.action in {DecisionAction.BUY, DecisionAction.SELL}:
                attempted_orders += 1
                if cost_model.is_rejected(attempted_orders):
                    rejected_orders += 1
                else:
                    fill = self._execute_decision(decision, candle, cash, position)
                    if fill is None:
                        rejected_orders += 1
                    else:
                        fills.append(fill)
                        if fill.side is DecisionAction.BUY:
                            cash -= fill.notional_usdt + fill.fee_cost_usdt()
                            position += fill.size_base
                        else:
                            cash += fill.notional_usdt - fill.fee_cost_usdt()
                            position -= min(position, fill.size_base)

            equity_curve.append(EquityPoint(candle.ts, cash + position * candle.close))

        metrics = build_pnl_metrics(
            fills,
            starting_equity_usdt=self.config.starting_equity_usdt,
            mark_prices={inst_id: ordered_candles[-1].close},
        )
        sharpe, sortino = risk_adjusted_ratios(tuple(equity_curve))
        metrics = replace(
            metrics,
            ending_equity_usdt=equity_curve[-1].equity_usdt,
            net_pnl_usdt=equity_curve[-1].equity_usdt - self.config.starting_equity_usdt,
            max_drawdown_usdt=max_drawdown_usdt(tuple(equity_curve)),
            sharpe=sharpe,
            sortino=sortino,
            equity_curve=tuple(equity_curve),
        )
        return BacktestReport(
            inst_id=inst_id,
            strategy_name=self.strategy.name,
            parameters=self.strategy.describe(),
            period_start=ordered_candles[0].ts,
            period_end=ordered_candles[-1].ts,
            metrics=metrics,
            buy_and_hold_pnl_usdt=_buy_and_hold_pnl(
                ordered_candles,
                starting_equity_usdt=self.config.starting_equity_usdt,
                cost_model=cost_model,
            ),
            no_trade_pnl_usdt=Decimal("0"),
            rejected_order_count=rejected_orders,
            assumptions=cost_model.assumptions(),
        )

    def _execute_decision(
        self,
        decision,
        candle: MarketCandle,
        cash: Decimal,
        position: Decimal,
    ) -> TradeFill | None:
        size = decision.size_btc
        if decision.action is DecisionAction.SELL:
            size = min(size, position)
        if size <= 0:
            return None
        price = self.config.cost_model.execution_price(decision.action, candle.close)
        fee = self.config.cost_model.fee_usdt(size * price)
        if decision.action is DecisionAction.BUY and cash < size * price + fee:
            return None
        return TradeFill(
            inst_id=decision.inst_id,
            side=decision.action,
            size_base=size,
            price=price,
            fee=fee,
            fee_ccy="USDT",
            ts=candle.ts,
        )


def _buy_and_hold_pnl(
    candles: tuple[MarketCandle, ...],
    *,
    starting_equity_usdt: Decimal,
    cost_model: CostModel,
) -> Decimal:
    if len(candles) < 2:
        return Decimal("0")
    buy_price = cost_model.execution_price(DecisionAction.BUY, candles[0].close)
    sell_price = cost_model.execution_price(DecisionAction.SELL, candles[-1].close)
    fee_rate = cost_model.taker_fee_bps / BASIS_POINTS
    size = starting_equity_usdt / (buy_price * (Decimal("1") + fee_rate))
    ending_equity = size * sell_price * (Decimal("1") - fee_rate)
    return ending_equity - starting_equity_usdt
