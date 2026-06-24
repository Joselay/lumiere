from __future__ import annotations

from collections.abc import Mapping, Sequence
from decimal import Decimal, InvalidOperation
from html import escape
from typing import Protocol

from lumiere.models import AccountSnapshot, OrderResult
from lumiere.risk import RiskConfig


class CommandLike(Protocol):
    command: str
    description: str


def format_command_help(commands: Sequence[CommandLike]) -> str:
    lines = ["🤖 <b>Lumiere commands</b>", ""]
    lines.extend(f"/{command.command} — {command.description}" for command in commands)
    return "\n".join(lines)


def format_start_message(commands: Sequence[CommandLike], symbols: Sequence[str]) -> str:
    symbol_text = ", ".join(escape(symbol) for symbol in symbols) if symbols else "not configured"
    return "\n".join(
        (
            "👋 <b>Lumiere is online</b>",
            "",
            "Mode: OKX demo",
            f"Symbols: {symbol_text}",
            "",
            format_command_help(commands),
        )
    )


def format_status(status, account: AccountSnapshot) -> str:  # noqa: ANN001 - engine status DTO
    running_icon = "🟢" if status.running and not status.paused else "🟡"
    if status.panic_stopped:
        running_icon = "🔴"
    return "\n".join(
        (
            f"{running_icon} <b>Status</b>",
            "",
            "<b>Engine</b>",
            f"Running: {_yes_no(status.running)}",
            f"Paused: {_yes_no(status.paused)}",
            f"Panic stop: {_yes_no(status.panic_stopped)}",
            f"Failures: {status.consecutive_failures}",
            "",
            "<b>Account</b>",
            f"Equity: {_fmt_usdt(account.equity_usdt)} USDT",
            f"Available: {_fmt_usdt(account.available_usdt)} USDT",
            f"Positions: {_positions_text(account)}",
            "",
            "<b>Latest</b>",
            f"Decision: {_fmt_na(status.last_decision)}",
            f"Risk: {_fmt_reason(status.last_risk_reason)}",
            f"Error: {_fmt_na(status.last_error)}",
        )
    )


def format_performance(
    account: AccountSnapshot,
    *,
    rejected_by_cost_count: int,
    attribution_report: Mapping[str, object] | None,
) -> str:
    attribution_metrics = _nested_mapping(attribution_report, "metrics")
    attribution_alerts = _list_value(attribution_report, "alerts")
    lines = [
        "📊 <b>Performance — today</b>",
        "",
        "<b>OKX account</b>",
        f"Realized PnL: {_fmt_usdt(account.daily_realized_pnl_usdt)} USDT",
        f"Max drawdown: {_fmt_usdt(account.max_drawdown_usdt)} USDT",
        f"Trades: {account.daily_trade_count}",
        "",
        "<b>Costs</b>",
        f"Spread: {_fmt_bps(account.spread_bps)} bps",
        f"Est. slippage: {_fmt_bps(account.estimated_slippage_bps)} bps",
        f"Realized slippage: {_fmt_bps(account.realized_slippage_bps)} bps",
        f"Total est. cost: {_fmt_bps(account.estimated_total_cost_bps)} bps",
        f"Rejected by cost: {rejected_by_cost_count}",
        "",
        "<b>Gate</b>",
        f"Status: {_gate_status(account)}",
        f"Reason: {_fmt_reason(account.performance_gate_reason)}",
        "",
        "<b>Attribution ledger</b>",
    ]
    if attribution_metrics:
        lines.extend(
            (
                f"Net PnL: {_fmt_metric_usdt(attribution_metrics, 'net_pnl_usdt')} USDT",
                f"Fees: {_fmt_metric_usdt(attribution_metrics, 'fees_usdt')} USDT",
                f"Profit factor: {_fmt_profit_factor(attribution_metrics.get('profit_factor'))}",
                f"Alerts: {_fmt_alerts(attribution_alerts)}",
            )
        )
        has_no_attributed_fills = int(attribution_metrics.get("trade_count", 0)) == 0
        if account.daily_realized_pnl_usdt != 0 and has_no_attributed_fills:
            lines.extend(
                (
                    "",
                    "⚪ Note: attribution has no fills yet, so OKX PnL is more complete.",
                )
            )
    else:
        lines.append("Disabled or empty")
    return "\n".join(lines)


def format_risk(config: RiskConfig, account: AccountSnapshot, *, failures: int) -> str:
    return "\n".join(
        (
            "🛡️ <b>Risk</b>",
            "",
            "<b>Limits</b>",
            f"Symbols: {', '.join(escape(inst_id) for inst_id in config.allowed_inst_ids)}",
            (
                f"Daily loss: {_fmt_usdt(account.daily_realized_pnl_usdt)} / "
                f"-{_fmt_usdt(config.max_daily_loss_usdt)} USDT"
            ),
            (
                f"Drawdown: {_fmt_usdt(account.max_drawdown_usdt)} / "
                f"{_fmt_limit_usdt(config.max_drawdown_usdt)} USDT"
            ),
            f"Daily trades: {account.daily_trade_count} / {_fmt_limit(config.max_daily_trades)}",
            f"Spread: {_fmt_bps(account.spread_bps)} / {_fmt_limit_bps(config.max_spread_bps)} bps",
            "",
            "<b>Performance gate</b>",
            f"Required: {_yes_no(config.performance_gate_required)}",
            f"Status: {_gate_status(account)}",
            f"Reason: {_fmt_reason(account.performance_gate_reason)}",
            "",
            "<b>Runtime</b>",
            f"Failures: {failures}",
            f"Cost rejections: {account.rejected_by_cost_count}",
        )
    )


def format_strategies(strategies: Sequence[Mapping[str, object]]) -> str:
    lines = ["🧠 <b>Strategies</b>"]
    for index, params in enumerate(strategies, start=1):
        lines.extend(("", f"<b>{index}. {_fmt_na(params.get('name', 'strategy'))}</b>"))
        for key, value in params.items():
            if key == "name":
                continue
            lines.append(f"{_label(key)}: {_fmt_na(value)}")
    return "\n".join(lines)


def format_order_submitted(result: OrderResult, reason: str) -> str:
    icon = "🟢" if result.side.value == "buy" else "🔴"
    return "\n".join(
        (
            f"{icon} <b>{result.side.value.upper()} {escape(result.inst_id)}</b>",
            f"Size: {result.size_btc}",
            f"Status: {result.status}",
            f"Reason: {_fmt_reason(reason)}",
            f"Order: {_short_id(result.order_id)}",
        )
    )


def format_risk_blocked(inst_id: str, action: str, risk_reason: str, signal_reason: str) -> str:
    return "\n".join(
        (
            "🟡 <b>Risk blocked trade</b>",
            f"Market: {escape(inst_id)}",
            f"Action: {escape(action)}",
            f"Risk: {_fmt_reason(risk_reason)}",
            f"Signal: {_fmt_reason(signal_reason)}",
        )
    )


def format_lifecycle(event: str) -> str:
    if event == "started":
        return "🟢 <b>Lumiere started</b>\nMode: OKX demo"
    if event == "stopped":
        return "🔴 <b>Lumiere stopped</b>"
    return f"ℹ️ <b>Lumiere</b>\n{event}"


def format_error(error: object) -> str:
    return f"🔴 <b>Trading error</b>\n{error}"


def format_pause() -> str:
    return "⏸️ <b>Trading paused</b>"


def format_resume() -> str:
    return "▶️ <b>Trading resumed</b>"


def format_panic(cancelled_orders: int) -> str:
    return f"🚨 <b>PANIC stop active</b>\nCancelled open orders: {cancelled_orders}"


def _fmt_usdt(value: object | None) -> str:
    return _fmt_decimal(value, places=4)


def _fmt_bps(value: object | None) -> str:
    return _fmt_decimal(value, places=2)


def _fmt_limit(value: object | None) -> str:
    return "n/a" if value is None else str(value)


def _fmt_limit_usdt(value: object | None) -> str:
    return "n/a" if value is None else _fmt_usdt(value)


def _fmt_limit_bps(value: object | None) -> str:
    return "n/a" if value is None else _fmt_bps(value)


def _fmt_metric_usdt(metrics: Mapping[str, object], key: str) -> str:
    return _fmt_usdt(metrics.get(key))


def _fmt_decimal(value: object | None, *, places: int) -> str:
    if value in {None, "", "None"}:
        return "n/a"
    try:
        decimal = Decimal(str(value))
    except (InvalidOperation, ValueError):
        return escape(str(value))
    quantum = Decimal("1").scaleb(-places)
    rounded = decimal.quantize(quantum)
    return f"{rounded.normalize():f}"


def _fmt_profit_factor(value: object | None) -> str:
    if value in {None, "", "None"}:
        return "n/a"
    if str(value) == "Infinity":
        return "∞"
    return _fmt_decimal(value, places=2)


def _fmt_alerts(alerts: Sequence[object]) -> str:
    visible = [str(alert) for alert in alerts if str(alert) != "performance_gate_failure"]
    if not visible:
        return "none"
    return ", ".join(_fmt_reason(alert) for alert in visible)


def _fmt_reason(value: object | None) -> str:
    if value in {None, "", "None"}:
        return "n/a"
    return escape(str(value).replace("_", " "))


def _fmt_na(value: object | None) -> str:
    if value in {None, "", "None"}:
        return "n/a"
    return escape(str(value))


def _yes_no(value: bool) -> str:
    return "yes" if value else "no"


def _gate_status(account: AccountSnapshot) -> str:
    if account.performance_gate_reason == "not_evaluated":
        return "⚪ Not evaluated"
    if account.performance_gate_passed:
        return "🟢 Passed"
    return "🔴 Blocked"


def _positions_text(account: AccountSnapshot) -> str:
    if not account.positions:
        return "none"
    return ", ".join(
        f"{escape(position.inst_id)} {position.size_btc} @ {_fmt_usdt(position.avg_px)}"
        for position in account.positions
    )


def _short_id(value: str) -> str:
    if len(value) <= 12:
        return escape(value)
    return escape(f"{value[:6]}…{value[-4:]}")


def _label(key: object) -> str:
    return escape(str(key).replace("_", " ").title())


def _nested_mapping(source: Mapping[str, object] | None, key: str) -> Mapping[str, object]:
    if source is None:
        return {}
    value = source.get(key)
    return value if isinstance(value, Mapping) else {}


def _list_value(source: Mapping[str, object] | None, key: str) -> Sequence[object]:
    if source is None:
        return ()
    value = source.get(key)
    return value if isinstance(value, Sequence) and not isinstance(value, str) else ()
