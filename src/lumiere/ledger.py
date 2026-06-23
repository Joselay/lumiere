from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from math import sqrt
from statistics import mean, pstdev
from typing import Any

from lumiere.models import DecisionAction

BASIS_POINTS = Decimal("10000")


@dataclass(frozen=True, slots=True)
class TradeFill:
    inst_id: str
    side: DecisionAction
    size_base: Decimal
    price: Decimal
    fee: Decimal = Decimal("0")
    fee_ccy: str = "USDT"
    ts: datetime = datetime.min.replace(tzinfo=UTC)
    order_id: str = ""
    trade_id: str = ""

    def __post_init__(self) -> None:
        if self.side not in {DecisionAction.BUY, DecisionAction.SELL}:
            raise ValueError("TradeFill side must be buy or sell")
        if self.size_base <= 0:
            raise ValueError("TradeFill size_base must be positive")
        if self.price <= 0:
            raise ValueError("TradeFill price must be positive")
        if "-" not in self.inst_id:
            raise ValueError("TradeFill inst_id must be an OKX instrument id like BTC-USDT")

    @property
    def notional_usdt(self) -> Decimal:
        return self.size_base * self.price

    @property
    def base_ccy(self) -> str:
        return self.inst_id.split("-", maxsplit=1)[0]

    @property
    def quote_ccy(self) -> str:
        return self.inst_id.split("-")[-1]

    def fee_cost_usdt(self) -> Decimal:
        """Return a positive USDT cost for fees paid in quote or base currency."""

        fee_abs = abs(self.fee)
        fee_ccy = self.fee_ccy.upper()
        if fee_abs == 0:
            return Decimal("0")
        if fee_ccy in {"USDT", "USD", self.quote_ccy}:
            return fee_abs
        if fee_ccy == self.base_ccy:
            return fee_abs * self.price
        return Decimal("0")


@dataclass(frozen=True, slots=True)
class EquityPoint:
    ts: datetime
    equity_usdt: Decimal


@dataclass(frozen=True, slots=True)
class PnlMetrics:
    starting_equity_usdt: Decimal
    ending_equity_usdt: Decimal
    net_pnl_usdt: Decimal
    realized_pnl_usdt: Decimal
    unrealized_pnl_usdt: Decimal
    fees_usdt: Decimal
    trade_count: int
    closed_trade_count: int
    win_rate: Decimal
    profit_factor: Decimal | None
    max_drawdown_usdt: Decimal
    sharpe: float | None
    sortino: float | None
    equity_curve: tuple[EquityPoint, ...]

    def to_dict(self) -> dict[str, str | int | float | None | list[dict[str, str]]]:
        return {
            "starting_equity_usdt": str(self.starting_equity_usdt),
            "ending_equity_usdt": str(self.ending_equity_usdt),
            "net_pnl_usdt": str(self.net_pnl_usdt),
            "realized_pnl_usdt": str(self.realized_pnl_usdt),
            "unrealized_pnl_usdt": str(self.unrealized_pnl_usdt),
            "fees_usdt": str(self.fees_usdt),
            "trade_count": self.trade_count,
            "closed_trade_count": self.closed_trade_count,
            "win_rate": str(self.win_rate),
            "profit_factor": None if self.profit_factor is None else str(self.profit_factor),
            "max_drawdown_usdt": str(self.max_drawdown_usdt),
            "sharpe": self.sharpe,
            "sortino": self.sortino,
            "equity_curve": [
                {"ts": point.ts.isoformat(), "equity_usdt": str(point.equity_usdt)}
                for point in self.equity_curve
            ],
        }


def build_pnl_metrics(
    fills: list[TradeFill] | tuple[TradeFill, ...],
    *,
    starting_equity_usdt: Decimal,
    mark_prices: dict[str, Decimal] | None = None,
) -> PnlMetrics:
    """Build weighted-average realized/unrealized PnL metrics from fills.

    Fees are treated as costs. Buy fees increase cost basis; sell fees reduce closing PnL.
    The implementation supports multiple spot instruments quoted in USDT.
    """

    sorted_fills = sorted(fills, key=lambda fill: fill.ts)
    mark_prices = dict(mark_prices or {})
    cash = starting_equity_usdt
    size_by_inst: dict[str, Decimal] = {}
    cost_basis_by_inst: dict[str, Decimal] = {}
    last_price_by_inst: dict[str, Decimal] = {}
    realized_pnl = Decimal("0")
    fees = Decimal("0")
    closed_trade_count = 0
    winning_closes = 0
    gross_profit = Decimal("0")
    gross_loss = Decimal("0")
    curve: list[EquityPoint] = []

    for fill in sorted_fills:
        fee_cost = fill.fee_cost_usdt()
        fees += fee_cost
        last_price_by_inst[fill.inst_id] = fill.price
        size = size_by_inst.get(fill.inst_id, Decimal("0"))
        cost_basis = cost_basis_by_inst.get(fill.inst_id, Decimal("0"))

        if fill.side is DecisionAction.BUY:
            cash -= fill.notional_usdt + fee_cost
            size_by_inst[fill.inst_id] = size + fill.size_base
            cost_basis_by_inst[fill.inst_id] = cost_basis + fill.notional_usdt + fee_cost
        else:
            close_pnl, remaining_size, remaining_cost_basis = _close_position(
                current_size=size,
                current_cost_basis=cost_basis,
                sell_size=fill.size_base,
                proceeds_usdt=fill.notional_usdt,
                fee_cost_usdt=fee_cost,
            )
            cash += fill.notional_usdt - fee_cost
            size_by_inst[fill.inst_id] = remaining_size
            cost_basis_by_inst[fill.inst_id] = remaining_cost_basis
            realized_pnl += close_pnl
            closed_trade_count += 1
            if close_pnl > 0:
                winning_closes += 1
                gross_profit += close_pnl
            elif close_pnl < 0:
                gross_loss += abs(close_pnl)

        curve.append(EquityPoint(fill.ts, _equity(cash, size_by_inst, last_price_by_inst)))

    final_prices = {**last_price_by_inst, **mark_prices}
    ending_equity = _equity(cash, size_by_inst, final_prices)
    unrealized_pnl = _unrealized_pnl(size_by_inst, cost_basis_by_inst, final_prices)
    win_rate = (
        Decimal(winning_closes) / Decimal(closed_trade_count)
        if closed_trade_count
        else Decimal("0")
    )
    profit_factor = None
    if gross_loss > 0:
        profit_factor = gross_profit / gross_loss
    elif gross_profit > 0:
        profit_factor = Decimal("Infinity")

    max_drawdown = max_drawdown_usdt(tuple(curve))
    sharpe, sortino = risk_adjusted_ratios(tuple(curve))
    return PnlMetrics(
        starting_equity_usdt=starting_equity_usdt,
        ending_equity_usdt=ending_equity,
        net_pnl_usdt=ending_equity - starting_equity_usdt,
        realized_pnl_usdt=realized_pnl,
        unrealized_pnl_usdt=unrealized_pnl,
        fees_usdt=fees,
        trade_count=len(sorted_fills),
        closed_trade_count=closed_trade_count,
        win_rate=win_rate,
        profit_factor=profit_factor,
        max_drawdown_usdt=max_drawdown,
        sharpe=sharpe,
        sortino=sortino,
        equity_curve=tuple(curve),
    )


def realized_pnl_for_period(
    fills: list[TradeFill] | tuple[TradeFill, ...],
    *,
    period_start: datetime,
) -> Decimal:
    """Return realized PnL for closing fills at or after period_start.

    Earlier fills seed cost basis so a sell today can be measured against inventory bought before
    today when those fills are available from OKX fill history.
    """

    sorted_fills = sorted(fills, key=lambda fill: fill.ts)
    size_by_inst: dict[str, Decimal] = {}
    cost_basis_by_inst: dict[str, Decimal] = {}
    realized = Decimal("0")

    for fill in sorted_fills:
        fee_cost = fill.fee_cost_usdt()
        size = size_by_inst.get(fill.inst_id, Decimal("0"))
        cost_basis = cost_basis_by_inst.get(fill.inst_id, Decimal("0"))
        if fill.side is DecisionAction.BUY:
            size_by_inst[fill.inst_id] = size + fill.size_base
            cost_basis_by_inst[fill.inst_id] = cost_basis + fill.notional_usdt + fee_cost
            continue

        close_pnl, remaining_size, remaining_cost_basis = _close_position(
            current_size=size,
            current_cost_basis=cost_basis,
            sell_size=fill.size_base,
            proceeds_usdt=fill.notional_usdt,
            fee_cost_usdt=fee_cost,
        )
        size_by_inst[fill.inst_id] = remaining_size
        cost_basis_by_inst[fill.inst_id] = remaining_cost_basis
        if fill.ts >= period_start:
            realized += close_pnl

    return realized


def trade_fill_from_okx_row(row: dict[str, Any]) -> TradeFill:
    inst_id = str(row.get("instId") or "")
    side = DecisionAction(str(row.get("side") or "").lower())
    ts_ms = int(str(row.get("ts") or "0"))
    return TradeFill(
        inst_id=inst_id,
        side=side,
        size_base=Decimal(str(row.get("fillSz") or row.get("sz") or "0")),
        price=Decimal(str(row.get("fillPx") or row.get("px") or "0")),
        fee=Decimal(str(row.get("fee") or "0")),
        fee_ccy=str(row.get("feeCcy") or "USDT"),
        ts=datetime.fromtimestamp(ts_ms / 1000, tz=UTC),
        order_id=str(row.get("ordId") or ""),
        trade_id=str(row.get("tradeId") or row.get("execId") or ""),
    )


def max_drawdown_usdt(curve: tuple[EquityPoint, ...]) -> Decimal:
    peak: Decimal | None = None
    max_drawdown = Decimal("0")
    for point in curve:
        if peak is None or point.equity_usdt > peak:
            peak = point.equity_usdt
        drawdown = peak - point.equity_usdt
        if drawdown > max_drawdown:
            max_drawdown = drawdown
    return max_drawdown


def risk_adjusted_ratios(curve: tuple[EquityPoint, ...]) -> tuple[float | None, float | None]:
    if len(curve) < 3:
        return None, None
    returns: list[float] = []
    downside_returns: list[float] = []
    previous = curve[0].equity_usdt
    for point in curve[1:]:
        if previous == 0:
            previous = point.equity_usdt
            continue
        ret = float((point.equity_usdt - previous) / previous)
        returns.append(ret)
        if ret < 0:
            downside_returns.append(ret)
        previous = point.equity_usdt
    if len(returns) < 2:
        return None, None
    avg_return = mean(returns)
    volatility = pstdev(returns)
    sharpe = None if volatility == 0 else avg_return / volatility * sqrt(len(returns))
    downside_volatility = pstdev(downside_returns) if len(downside_returns) > 1 else 0
    sortino = (
        None if downside_volatility == 0 else avg_return / downside_volatility * sqrt(len(returns))
    )
    return sharpe, sortino


def _close_position(
    *,
    current_size: Decimal,
    current_cost_basis: Decimal,
    sell_size: Decimal,
    proceeds_usdt: Decimal,
    fee_cost_usdt: Decimal,
) -> tuple[Decimal, Decimal, Decimal]:
    if current_size <= 0:
        return -fee_cost_usdt, Decimal("0"), Decimal("0")

    costed_size = min(current_size, sell_size)
    uncosted_size = sell_size - costed_size
    avg_cost = current_cost_basis / current_size
    removed_cost_basis = avg_cost * costed_size
    # If history is incomplete and we see a sell larger than known inventory, do not fabricate
    # gains/losses for the uncosted portion; value it at proceeds before fees.
    uncosted_proceeds = Decimal("0")
    if uncosted_size > 0 and sell_size > 0:
        uncosted_proceeds = proceeds_usdt * (uncosted_size / sell_size)
    realized = proceeds_usdt - removed_cost_basis - uncosted_proceeds - fee_cost_usdt
    remaining_size = max(current_size - sell_size, Decimal("0"))
    remaining_cost_basis = (
        Decimal("0") if remaining_size == 0 else current_cost_basis - removed_cost_basis
    )
    return realized, remaining_size, remaining_cost_basis


def _equity(
    cash: Decimal,
    size_by_inst: dict[str, Decimal],
    price_by_inst: dict[str, Decimal],
) -> Decimal:
    equity = cash
    for inst_id, size in size_by_inst.items():
        equity += size * price_by_inst.get(inst_id, Decimal("0"))
    return equity


def _unrealized_pnl(
    size_by_inst: dict[str, Decimal],
    cost_basis_by_inst: dict[str, Decimal],
    price_by_inst: dict[str, Decimal],
) -> Decimal:
    unrealized = Decimal("0")
    for inst_id, size in size_by_inst.items():
        market_value = size * price_by_inst.get(inst_id, Decimal("0"))
        unrealized += market_value - cost_basis_by_inst.get(inst_id, Decimal("0"))
    return unrealized
