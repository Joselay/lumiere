from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
from decimal import Decimal
from typing import Any, Literal

from lumiere.ledger import (
    BASIS_POINTS,
    EquityPoint,
    ExposurePoint,
    PnlMetrics,
    TradeFill,
    build_pnl_metrics,
    max_drawdown_duration_bars,
    max_drawdown_usdt,
    risk_adjusted_ratios,
)
from lumiere.models import AccountSnapshot, DecisionAction, MarketCandle, Position, StrategyDecision
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


ExecutionTiming = Literal["next_open", "same_close"]


@dataclass(frozen=True, slots=True)
class BacktestConfig:
    starting_equity_usdt: Decimal = Decimal("1000")
    cost_model: CostModel = CostModel()
    stop_loss_bps: Decimal | None = None
    take_profit_bps: Decimal | None = None
    trailing_stop_bps: Decimal | None = None
    max_bars_in_trade: int | None = None
    execution_timing: ExecutionTiming = "next_open"
    ignore_unconfirmed_candles: bool = True
    include_same_close_comparison: bool = True

    def __post_init__(self) -> None:
        if self.starting_equity_usdt <= 0:
            raise ValueError("starting_equity_usdt must be positive")
        for value in (self.stop_loss_bps, self.take_profit_bps, self.trailing_stop_bps):
            if value is not None and value <= 0:
                raise ValueError("exit bps values must be positive when configured")
        if self.max_bars_in_trade is not None and self.max_bars_in_trade <= 0:
            raise ValueError("max_bars_in_trade must be positive when configured")
        if self.execution_timing not in {"next_open", "same_close"}:
            raise ValueError("execution_timing must be 'next_open' or 'same_close'")


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
    execution_timing: ExecutionTiming = "next_open"
    same_close_comparison: dict[str, str | int] | None = None

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
            "execution_timing": self.execution_timing,
            "same_close_comparison": self.same_close_comparison,
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
        ordered_candles = tuple(
            sorted(
                (
                    candle
                    for candle in candles
                    if candle.confirmed or not self.config.ignore_unconfirmed_candles
                ),
                key=lambda candle: candle.ts,
            )
        )
        if not ordered_candles:
            raise ValueError("at least one confirmed candle is required")

        report = self._run_once(ordered_candles)
        if (
            self.config.execution_timing == "next_open"
            and self.config.include_same_close_comparison
        ):
            comparison_config = replace(
                self.config,
                execution_timing="same_close",
                include_same_close_comparison=False,
            )
            comparison = Backtester(self.strategy, comparison_config)._run_once(ordered_candles)
            report = replace(
                report,
                same_close_comparison={
                    "execution_timing": comparison.execution_timing,
                    "net_pnl_usdt": str(comparison.metrics.net_pnl_usdt),
                    "ending_equity_usdt": str(comparison.metrics.ending_equity_usdt),
                    "max_drawdown_usdt": str(comparison.metrics.max_drawdown_usdt),
                    "trade_count": comparison.metrics.trade_count,
                    "rejected_order_count": comparison.rejected_order_count,
                },
            )
        return report

    def _run_once(self, ordered_candles: tuple[MarketCandle, ...]) -> BacktestReport:
        cash = self.config.starting_equity_usdt
        position = Decimal("0")
        fills: list[TradeFill] = []
        equity_curve: list[EquityPoint] = []
        exposure_curve: list[ExposurePoint] = []
        attempted_orders = 0
        entry_price: Decimal | None = None
        entry_index: int | None = None
        highest_since_entry: Decimal | None = None
        rejected_orders = 0
        pending_decision: StrategyDecision | None = None
        cost_model = self.config.cost_model
        inst_id = self.strategy.config.inst_id

        for index, candle in enumerate(ordered_candles):
            if pending_decision is not None:
                attempted_orders += 1
                fill = None
                if cost_model.is_rejected(attempted_orders):
                    rejected_orders += 1
                else:
                    fill = self._execute_decision(
                        pending_decision,
                        mid_price=candle.open,
                        ts=candle.ts,
                        cash=cash,
                        position=position,
                    )
                    if fill is None:
                        rejected_orders += 1
                if fill is not None:
                    fills.append(fill)
                    cash, position, entry_price, entry_index, highest_since_entry = (
                        self._apply_fill(
                            fill,
                            cash,
                            position,
                            entry_price,
                            entry_index,
                            highest_since_entry,
                            entry_index_for_new_position=index,
                            high_for_new_position=max(candle.open, candle.high),
                        )
                    )
                pending_decision = None

            account = AccountSnapshot(
                equity_usdt=cash + position * candle.close,
                available_usdt=cash,
                positions=((Position(inst_id=inst_id, size_btc=position),) if position > 0 else ()),
                performance_gate_passed=True,
            )
            exit_decision = self._exit_decision(
                candle,
                position,
                entry_price,
                entry_index,
                index,
                highest_since_entry,
            )
            decision = exit_decision or self.strategy.decide(
                list(ordered_candles[: index + 1]),
                account,
            )
            if decision.action in {DecisionAction.BUY, DecisionAction.SELL}:
                if self.config.execution_timing == "next_open" and exit_decision is None:
                    if index + 1 < len(ordered_candles):
                        pending_decision = decision
                    else:
                        # A signal on the final candle has no known next bar and is deliberately
                        # not filled; filling it would reintroduce a lookahead assumption.
                        rejected_orders += 1
                else:
                    attempted_orders += 1
                    if cost_model.is_rejected(attempted_orders):
                        rejected_orders += 1
                    else:
                        fill = self._execute_decision(
                            decision,
                            mid_price=self._decision_mid_price(decision, candle),
                            ts=candle.ts,
                            cash=cash,
                            position=position,
                        )
                        if fill is None:
                            rejected_orders += 1
                        else:
                            fills.append(fill)
                            cash, position, entry_price, entry_index, highest_since_entry = (
                                self._apply_fill(
                                    fill,
                                    cash,
                                    position,
                                    entry_price,
                                    entry_index,
                                    highest_since_entry,
                                    entry_index_for_new_position=index,
                                    high_for_new_position=candle.high,
                                )
                            )
            if position > 0:
                highest_since_entry = max(highest_since_entry or candle.high, candle.high)

            equity_curve.append(EquityPoint(candle.ts, cash + position * candle.close))
            exposure_curve.append(ExposurePoint(candle.ts, position * candle.close))

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
            max_drawdown_duration_bars=max_drawdown_duration_bars(tuple(equity_curve)),
            exposure_curve=tuple(exposure_curve),
        )
        fill_price = (
            "next_bar_open" if self.config.execution_timing == "next_open" else "same_bar_close"
        )
        unconfirmed_candles = "ignored" if self.config.ignore_unconfirmed_candles else "included"
        assumptions = cost_model.assumptions() | {
            "execution_timing": self.config.execution_timing,
            "signal_candle": "closed_confirmed",
            "fill_price": fill_price,
            "unconfirmed_candles": unconfirmed_candles,
            "intrabar_exit_sequence": "pessimistic_high_low_stop_before_profit",
        }
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
            assumptions=assumptions,
            execution_timing=self.config.execution_timing,
        )

    def _exit_decision(
        self,
        candle: MarketCandle,
        position: Decimal,
        entry_price: Decimal | None,
        entry_index: int | None,
        index: int,
        highest_since_entry: Decimal | None,
    ) -> StrategyDecision | None:
        if position <= 0 or entry_price is None:
            return None

        candidates: list[tuple[Decimal, str]] = []
        if self.config.stop_loss_bps is not None:
            stop_price = entry_price * (
                Decimal("1") - self.config.stop_loss_bps / Decimal("10000")
            )
            if candle.low <= stop_price:
                candidates.append((stop_price, "stop_loss"))
        if self.config.trailing_stop_bps is not None:
            effective_high = max(highest_since_entry or candle.high, candle.high)
            trailing_stop = effective_high * (
                Decimal("1") - self.config.trailing_stop_bps / Decimal("10000")
            )
            if candle.low <= trailing_stop:
                candidates.append((trailing_stop, "trailing_stop"))
        if self.config.take_profit_bps is not None:
            take_profit_price = entry_price * (
                Decimal("1") + self.config.take_profit_bps / Decimal("10000")
            )
            if candle.high >= take_profit_price:
                candidates.append((take_profit_price, "take_profit"))
        if candidates:
            # If the bar's high and low cross several exits, assume the least favorable long exit
            # happened first. This avoids optimistic close-only or take-profit-first bias.
            execution_price, reason = min(candidates, key=lambda item: item[0])
            return StrategyDecision(
                DecisionAction.SELL,
                self.strategy.config.inst_id,
                position,
                reason,
                {"execution_price": str(execution_price)},
            )
        if (
            self.config.max_bars_in_trade is not None
            and entry_index is not None
            and index - entry_index >= self.config.max_bars_in_trade
        ):
            return StrategyDecision(
                DecisionAction.SELL,
                self.strategy.config.inst_id,
                position,
                "time_in_trade_exit",
            )
        return None

    def _decision_mid_price(self, decision: StrategyDecision, candle: MarketCandle) -> Decimal:
        raw_execution_price = decision.inputs.get("execution_price")
        if raw_execution_price is not None:
            return Decimal(str(raw_execution_price))
        return candle.close

    def _execute_decision(
        self,
        decision: StrategyDecision,
        *,
        mid_price: Decimal,
        ts: datetime,
        cash: Decimal,
        position: Decimal,
    ) -> TradeFill | None:
        size = decision.size_btc
        if decision.action is DecisionAction.SELL:
            size = min(size, position)
        if size <= 0:
            return None
        price = self.config.cost_model.execution_price(decision.action, mid_price)
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
            ts=ts,
        )

    def _apply_fill(
        self,
        fill: TradeFill,
        cash: Decimal,
        position: Decimal,
        entry_price: Decimal | None,
        entry_index: int | None,
        highest_since_entry: Decimal | None,
        *,
        entry_index_for_new_position: int,
        high_for_new_position: Decimal,
    ) -> tuple[Decimal, Decimal, Decimal | None, int | None, Decimal | None]:
        if fill.side is DecisionAction.BUY:
            cash -= fill.notional_usdt + fill.fee_cost_usdt()
            position += fill.size_base
            entry_price = fill.price if entry_price is None else entry_price
            entry_index = entry_index_for_new_position if entry_index is None else entry_index
            highest_since_entry = max(
                highest_since_entry or high_for_new_position,
                high_for_new_position,
            )
        else:
            cash += fill.notional_usdt - fill.fee_cost_usdt()
            position -= min(position, fill.size_base)
            if position <= 0:
                position = Decimal("0")
                entry_price = None
                entry_index = None
                highest_since_entry = None
        return cash, position, entry_price, entry_index, highest_since_entry


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
