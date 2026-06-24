from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime
from decimal import Decimal
from typing import Any, Literal

from lumiere.execution import (
    ExecutionPolicy,
    ExecutionPolicyName,
    OrderBookTop,
    prepare_execution_plan,
)
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
from lumiere.risk import RiskConfig, RiskManager
from lumiere.strategy import TradingStrategy


@dataclass(frozen=True, slots=True)
class CostModel:
    """Deterministic market-order cost assumptions for backtests."""

    taker_fee_bps: Decimal = Decimal("10")
    maker_fee_bps: Decimal = Decimal("2")
    spread_bps: Decimal = Decimal("2")
    slippage_bps: Decimal = Decimal("5")
    market_impact_bps: Decimal = Decimal("0")
    reject_every_n_orders: int = 0
    execution_policy: ExecutionPolicyName = "market"
    marketable_limit_buffer_bps: Decimal = Decimal("1")
    post_only_offset_bps: Decimal = Decimal("0")
    maker_timeout_bars: int = 1
    maker_fill_fraction: Decimal = Decimal("1")

    def __post_init__(self) -> None:
        if self.taker_fee_bps < 0:
            raise ValueError("taker_fee_bps cannot be negative")
        if self.maker_fee_bps < 0:
            raise ValueError("maker_fee_bps cannot be negative")
        if self.spread_bps < 0:
            raise ValueError("spread_bps cannot be negative")
        if self.slippage_bps < 0:
            raise ValueError("slippage_bps cannot be negative")
        if self.market_impact_bps < 0:
            raise ValueError("market_impact_bps cannot be negative")
        if self.reject_every_n_orders < 0:
            raise ValueError("reject_every_n_orders cannot be negative")
        if self.execution_policy not in {"market", "marketable_limit", "post_only_maker"}:
            raise ValueError(
                "execution_policy must be market, marketable_limit, or post_only_maker"
            )
        if self.marketable_limit_buffer_bps < 0:
            raise ValueError("marketable_limit_buffer_bps cannot be negative")
        if self.post_only_offset_bps < 0:
            raise ValueError("post_only_offset_bps cannot be negative")
        if self.maker_timeout_bars <= 0:
            raise ValueError("maker_timeout_bars must be positive")
        if self.maker_fill_fraction <= 0 or self.maker_fill_fraction > 1:
            raise ValueError("maker_fill_fraction must be between 0 and 1")

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

    def fee_usdt(self, notional_usdt: Decimal, *, maker: bool = False) -> Decimal:
        fee_bps = self.maker_fee_bps if maker else self.taker_fee_bps
        return abs(notional_usdt) * fee_bps / BASIS_POINTS

    def execution_policy_config(self) -> ExecutionPolicy:
        return ExecutionPolicy(
            name=self.execution_policy,
            marketable_limit_buffer_bps=self.marketable_limit_buffer_bps,
            post_only_offset_bps=self.post_only_offset_bps,
        )

    def is_rejected(self, order_number: int) -> bool:
        return self.reject_every_n_orders > 0 and order_number % self.reject_every_n_orders == 0

    def assumptions(self) -> dict[str, str | int]:
        return {
            "taker_fee_bps": str(self.taker_fee_bps),
            "maker_fee_bps": str(self.maker_fee_bps),
            "spread_bps": str(self.spread_bps),
            "slippage_bps": str(self.slippage_bps),
            "market_impact_bps": str(self.market_impact_bps),
            "reject_every_n_orders": self.reject_every_n_orders,
            "execution_policy": self.execution_policy,
            "marketable_limit_buffer_bps": str(self.marketable_limit_buffer_bps),
            "post_only_offset_bps": str(self.post_only_offset_bps),
            "maker_timeout_bars": self.maker_timeout_bars,
            "maker_fill_fraction": str(self.maker_fill_fraction),
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
    risk_config: RiskConfig | None = None

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
class ExecutionQualityAccumulator:
    policy: str
    attempted_order_count: int = 0
    filled_order_count: int = 0
    non_fill_count: int = 0
    partial_fill_count: int = 0
    fill_delay_bars_total: int = 0
    realized_spread_capture_bps_total: Decimal = Decimal("0")
    adverse_selection_bps_total: Decimal = Decimal("0")
    missed_trade_opportunity_cost_usdt: Decimal = Decimal("0")

    def record_fill(
        self,
        *,
        requested_size: Decimal,
        filled_size: Decimal,
        fill_delay_bars: int,
        spread_capture_bps: Decimal,
        adverse_selection_bps: Decimal,
    ) -> ExecutionQualityAccumulator:
        return replace(
            self,
            filled_order_count=self.filled_order_count + 1,
            partial_fill_count=(
                self.partial_fill_count + (1 if filled_size < requested_size else 0)
            ),
            fill_delay_bars_total=self.fill_delay_bars_total + fill_delay_bars,
            realized_spread_capture_bps_total=(
                self.realized_spread_capture_bps_total + spread_capture_bps
            ),
            adverse_selection_bps_total=(
                self.adverse_selection_bps_total + adverse_selection_bps
            ),
        )

    def record_non_fill(self, opportunity_cost_usdt: Decimal) -> ExecutionQualityAccumulator:
        return replace(
            self,
            non_fill_count=self.non_fill_count + 1,
            missed_trade_opportunity_cost_usdt=(
                self.missed_trade_opportunity_cost_usdt + opportunity_cost_usdt
            ),
        )

    def to_dict(self) -> dict[str, str | int]:
        attempts = Decimal(self.attempted_order_count or 1)
        fills = Decimal(self.filled_order_count or 1)
        return {
            "policy": self.policy,
            "attempted_order_count": self.attempted_order_count,
            "filled_order_count": self.filled_order_count,
            "non_fill_count": self.non_fill_count,
            "partial_fill_count": self.partial_fill_count,
            "non_fill_rate": str(Decimal(self.non_fill_count) / attempts),
            "partial_fill_rate": str(Decimal(self.partial_fill_count) / attempts),
            "average_fill_delay_bars": str(Decimal(self.fill_delay_bars_total) / fills),
            "realized_spread_capture_bps": str(
                self.realized_spread_capture_bps_total / fills
            ),
            "adverse_selection_bps": str(self.adverse_selection_bps_total / fills),
            "missed_trade_opportunity_cost_usdt": str(
                self.missed_trade_opportunity_cost_usdt
            ),
        }


@dataclass(frozen=True, slots=True)
class SimulatedExecution:
    fill: TradeFill | None
    quality: ExecutionQualityAccumulator


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
    risk_rejection_count: int = 0
    risk_rejections: dict[str, int] = field(default_factory=dict)
    blocked_signal_opportunity_cost_usdt: Decimal = Decimal("0")
    execution_quality: dict[str, str | int] | None = None

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
            "risk_rejection_count": self.risk_rejection_count,
            "risk_rejections": self.risk_rejections,
            "blocked_signal_opportunity_cost_usdt": str(
                self.blocked_signal_opportunity_cost_usdt
            ),
            "execution_quality": self.execution_quality,
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
                    "risk_rejection_count": comparison.risk_rejection_count,
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
        risk_rejections: dict[str, int] = {}
        blocked_signal_opportunity_cost = Decimal("0")
        pending_decision: StrategyDecision | None = None
        risk_manager = RiskManager(self.config.risk_config) if self.config.risk_config else None
        cost_model = self.config.cost_model
        quality = ExecutionQualityAccumulator(policy=cost_model.execution_policy)
        inst_id = self.strategy.config.inst_id
        account_peak_equity: Decimal | None = None
        account_max_drawdown = Decimal("0")

        for index, candle in enumerate(ordered_candles):
            pending_still_open = False
            if pending_decision is not None:
                fill_delay_bars = _pending_fill_delay_bars(pending_decision, index)
                if fill_delay_bars == 1:
                    attempted_orders += 1
                    quality = replace(
                        quality,
                        attempted_order_count=quality.attempted_order_count + 1,
                    )
                fill = None
                maker_timeout_reached = fill_delay_bars >= cost_model.maker_timeout_bars
                if fill_delay_bars == 1 and cost_model.is_rejected(attempted_orders):
                    rejected_orders += 1
                    pending_decision = None
                else:
                    execution = self._execute_decision(
                        pending_decision,
                        mid_price=candle.open,
                        candle=candle,
                        ts=candle.ts,
                        cash=cash,
                        position=position,
                        quality=quality,
                        fill_delay_bars=fill_delay_bars,
                        record_non_fill=maker_timeout_reached,
                    )
                    fill = execution.fill
                    quality = execution.quality
                    if fill is None:
                        if (
                            cost_model.execution_policy == "post_only_maker"
                            and not maker_timeout_reached
                        ):
                            pending_still_open = True
                        else:
                            rejected_orders += 1
                            pending_decision = None
                if fill is not None:
                    fills.append(fill)
                    if risk_manager is not None:
                        risk_manager.record_trade(fill.ts, inst_id=fill.inst_id)
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

            current_equity = cash + position * candle.close
            if account_peak_equity is None or current_equity > account_peak_equity:
                account_peak_equity = current_equity
            account_max_drawdown = max(
                account_max_drawdown,
                account_peak_equity - current_equity,
            )
            account = AccountSnapshot(
                equity_usdt=current_equity,
                available_usdt=cash,
                positions=((Position(inst_id=inst_id, size_btc=position),) if position > 0 else ()),
                daily_trade_count=len(fills),
                max_drawdown_usdt=account_max_drawdown,
                spread_bps=cost_model.spread_bps,
                estimated_slippage_bps=cost_model.slippage_bps + cost_model.market_impact_bps,
                estimated_total_cost_bps=cost_model.order_cost_bps + cost_model.taker_fee_bps,
                performance_gate_passed=True,
                maker_non_fill_rate=_quality_rate(
                    quality.non_fill_count,
                    quality.attempted_order_count,
                ),
                maker_adverse_selection_bps=_quality_average(
                    quality.adverse_selection_bps_total,
                    quality.filled_order_count,
                ),
            )
            if pending_still_open:
                if position > 0:
                    highest_since_entry = max(highest_since_entry or candle.high, candle.high)
                equity_curve.append(EquityPoint(candle.ts, current_equity))
                exposure_curve.append(ExposurePoint(candle.ts, position * candle.close))
                continue

            exit_decision = self._exit_decision(
                candle,
                position,
                entry_price,
                entry_index,
                index,
                highest_since_entry,
            )
            decision = exit_decision or self.strategy.decide(
                _decision_history(self.strategy, ordered_candles, index),
                account,
            )
            if decision.action in {DecisionAction.BUY, DecisionAction.SELL}:
                execution_plan = prepare_execution_plan(
                    decision,
                    account,
                    risk_manager,
                    now=candle.ts,
                    mark_price=self._decision_mid_price(decision, candle),
                )
                if not execution_plan.allowed:
                    reason = execution_plan.reason
                    risk_rejections[reason] = risk_rejections.get(reason, 0) + 1
                    blocked_signal_opportunity_cost += (
                        execution_plan.blocked_signal_notional_usdt
                    )
                    rejected_orders += 1
                else:
                    executable_decision = self._decision_with_execution_inputs(
                        execution_plan.decision,
                        candle.close,
                        index,
                    )
                    if self.config.execution_timing == "next_open" and exit_decision is None:
                        if index + 1 < len(ordered_candles):
                            pending_decision = executable_decision
                        else:
                            # A signal on the final candle has no known next bar and is deliberately
                            # not filled; filling it would reintroduce a lookahead assumption.
                            rejected_orders += 1
                    else:
                        attempted_orders += 1
                        quality = replace(
                            quality,
                            attempted_order_count=quality.attempted_order_count + 1,
                        )
                        if cost_model.is_rejected(attempted_orders):
                            rejected_orders += 1
                        else:
                            execution = self._execute_decision(
                                executable_decision,
                                mid_price=self._decision_mid_price(executable_decision, candle),
                                candle=candle,
                                ts=candle.ts,
                                cash=cash,
                                position=position,
                                quality=quality,
                                fill_delay_bars=0,
                            )
                            fill = execution.fill
                            quality = execution.quality
                            if fill is None:
                                rejected_orders += 1
                            else:
                                fills.append(fill)
                                if risk_manager is not None:
                                    risk_manager.record_trade(fill.ts, inst_id=fill.inst_id)
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
            risk_rejection_count=sum(risk_rejections.values()),
            risk_rejections=risk_rejections,
            blocked_signal_opportunity_cost_usdt=blocked_signal_opportunity_cost,
            execution_quality=quality.to_dict(),
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

    def _decision_with_execution_inputs(
        self,
        decision: StrategyDecision,
        reference_mid_price: Decimal,
        submitted_index: int,
    ) -> StrategyDecision:
        return replace(
            decision,
            inputs=decision.inputs
            | {
                "execution_policy": self.config.cost_model.execution_policy,
                "execution_reference_mid_price": str(reference_mid_price),
                "execution_submitted_index": submitted_index,
            },
        )

    def _execute_decision(
        self,
        decision: StrategyDecision,
        *,
        mid_price: Decimal,
        candle: MarketCandle,
        ts: datetime,
        cash: Decimal,
        position: Decimal,
        quality: ExecutionQualityAccumulator,
        fill_delay_bars: int,
        record_non_fill: bool = True,
    ) -> SimulatedExecution:
        size = decision.size_btc
        if decision.action is DecisionAction.SELL:
            size = min(size, position)
        if size <= 0:
            return SimulatedExecution(None, quality)
        cost_model = self.config.cost_model
        reference_mid_price = Decimal(
            str(decision.inputs.get("execution_reference_mid_price", mid_price))
        )
        policy = cost_model.execution_policy
        maker = policy == "post_only_maker"
        if maker:
            price = _maker_limit_price(cost_model, decision.action, reference_mid_price)
            if not _maker_order_crossed(decision.action, price, candle):
                if not record_non_fill:
                    return SimulatedExecution(None, quality)
                opportunity_cost = _missed_trade_opportunity_cost(
                    decision.action,
                    size,
                    cost_model.execution_price(decision.action, mid_price),
                    candle.close,
                )
                return SimulatedExecution(None, quality.record_non_fill(opportunity_cost))
            size *= cost_model.maker_fill_fraction
        elif policy == "marketable_limit":
            book = _synthetic_order_book(cost_model, mid_price)
            price = _marketable_limit_price(cost_model, decision.action, book)
        else:
            price = cost_model.execution_price(decision.action, mid_price)
        fee = cost_model.fee_usdt(size * price, maker=maker)
        if decision.action is DecisionAction.BUY and cash < size * price + fee:
            return SimulatedExecution(None, quality)
        fill = TradeFill(
            inst_id=decision.inst_id,
            side=decision.action,
            size_base=size,
            price=price,
            fee=fee,
            fee_ccy="USDT",
            ts=ts,
        )
        spread_capture_bps = _spread_capture_bps(
            decision.action,
            maker_price=price,
            market_price=cost_model.execution_price(decision.action, mid_price),
        )
        adverse_selection_bps = _adverse_selection_bps(decision.action, price, candle.close)
        return SimulatedExecution(
            fill,
            quality.record_fill(
                requested_size=decision.size_btc,
                filled_size=size,
                fill_delay_bars=fill_delay_bars,
                spread_capture_bps=spread_capture_bps,
                adverse_selection_bps=adverse_selection_bps,
            ),
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


def _pending_fill_delay_bars(decision: StrategyDecision, index: int) -> int:
    submitted_index = int(decision.inputs.get("execution_submitted_index", index - 1))
    return max(index - submitted_index, 0)


def _quality_rate(count: int, attempts: int) -> Decimal | None:
    if attempts <= 0:
        return None
    return Decimal(count) / Decimal(attempts)


def _quality_average(total: Decimal, count: int) -> Decimal | None:
    if count <= 0:
        return None
    return total / Decimal(count)


def _decision_history(
    strategy: TradingStrategy,
    candles: tuple[MarketCandle, ...],
    index: int,
) -> list[MarketCandle]:
    required_bars = _strategy_required_history_bars(strategy)
    start = 0 if required_bars is None else max(0, index + 1 - required_bars)
    return list(candles[start : index + 1])


def _strategy_required_history_bars(strategy: TradingStrategy) -> int | None:
    config = getattr(strategy, "config", None)
    if config is None:
        return None
    slow_window = getattr(config, "slow_window", None)
    if slow_window is not None:
        return int(slow_window)
    rsi_period = getattr(config, "rsi_period", None)
    if rsi_period is not None:
        return int(rsi_period) + 1
    lookback = getattr(config, "lookback", None)
    atr_period = getattr(config, "atr_period", None)
    if lookback is not None or atr_period is not None:
        return max(int(lookback or 0) + 1, int(atr_period or 0) + 1)
    return None


def _synthetic_order_book(cost_model: CostModel, mid_price: Decimal) -> OrderBookTop:
    half_spread = cost_model.spread_bps / Decimal("2") / BASIS_POINTS
    return OrderBookTop(
        bid=mid_price * (Decimal("1") - half_spread),
        ask=mid_price * (Decimal("1") + half_spread),
    )


def _maker_limit_price(
    cost_model: CostModel,
    side: DecisionAction,
    reference_mid_price: Decimal,
) -> Decimal:
    book = _synthetic_order_book(cost_model, reference_mid_price)
    offset = cost_model.post_only_offset_bps / BASIS_POINTS
    if side is DecisionAction.BUY:
        return book.bid * (Decimal("1") - offset)
    if side is DecisionAction.SELL:
        return book.ask * (Decimal("1") + offset)
    raise ValueError("maker limit requires buy or sell side")


def _marketable_limit_price(
    cost_model: CostModel,
    side: DecisionAction,
    book: OrderBookTop,
) -> Decimal:
    buffer = cost_model.marketable_limit_buffer_bps / BASIS_POINTS
    if side is DecisionAction.BUY:
        return book.ask * (Decimal("1") + buffer)
    if side is DecisionAction.SELL:
        return book.bid * (Decimal("1") - buffer)
    raise ValueError("marketable limit requires buy or sell side")


def _maker_order_crossed(side: DecisionAction, limit_price: Decimal, candle: MarketCandle) -> bool:
    if side is DecisionAction.BUY:
        return candle.low <= limit_price
    if side is DecisionAction.SELL:
        return candle.high >= limit_price
    return False


def _spread_capture_bps(
    side: DecisionAction,
    *,
    maker_price: Decimal,
    market_price: Decimal,
) -> Decimal:
    if maker_price <= 0:
        return Decimal("0")
    if side is DecisionAction.BUY:
        return (market_price - maker_price) / maker_price * BASIS_POINTS
    if side is DecisionAction.SELL:
        return (maker_price - market_price) / maker_price * BASIS_POINTS
    return Decimal("0")


def _adverse_selection_bps(
    side: DecisionAction,
    fill_price: Decimal,
    mark_price: Decimal,
) -> Decimal:
    if fill_price <= 0:
        return Decimal("0")
    if side is DecisionAction.BUY:
        return max((fill_price - mark_price) / fill_price * BASIS_POINTS, Decimal("0"))
    if side is DecisionAction.SELL:
        return max((mark_price - fill_price) / fill_price * BASIS_POINTS, Decimal("0"))
    return Decimal("0")


def _missed_trade_opportunity_cost(
    side: DecisionAction,
    size: Decimal,
    market_price: Decimal,
    mark_price: Decimal,
) -> Decimal:
    if side is DecisionAction.BUY:
        return max((mark_price - market_price) * size, Decimal("0"))
    if side is DecisionAction.SELL:
        return max((market_price - mark_price) * size, Decimal("0"))
    return Decimal("0")


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
