from __future__ import annotations

import argparse
import json
from datetime import timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

from lumiere.attribution import AttributionLedger

STAGE_THRESHOLDS = {
    "backtest": {
        "min_trades": 30,
        "min_profit_factor": Decimal("1.2"),
        "max_drawdown_pct": Decimal("0.05"),
    },
    "paper": {
        "min_trades": 30,
        "min_profit_factor": Decimal("1.2"),
        "max_drawdown_pct": Decimal("0.05"),
    },
    "small_demo": {
        "min_trades": 20,
        "min_profit_factor": Decimal("1.0"),
        "max_drawdown_pct": Decimal("0.05"),
    },
    "larger_demo": {
        "min_trades": 20,
        "min_profit_factor": Decimal("1.0"),
        "max_drawdown_pct": Decimal("0.05"),
    },
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build Lumiere staged promotion evidence packet")
    parser.add_argument("--stage", choices=tuple(STAGE_THRESHOLDS), required=True)
    parser.add_argument(
        "--optimizer-report",
        default="reports/strategy_optimization/optimizer_report.json",
    )
    parser.add_argument("--attribution-ledger", default="data/attribution.jsonl")
    parser.add_argument("--window-days", type=int, default=14)
    parser.add_argument("--output", default="reports/promotion_evidence.json")
    return parser


def build_evidence_packet(args: argparse.Namespace) -> dict[str, Any]:
    thresholds = STAGE_THRESHOLDS[args.stage]
    optimizer = _load_json(Path(args.optimizer_report))
    attribution_path = Path(args.attribution_ledger)
    attribution_report = None
    if attribution_path.exists():
        attribution_report = AttributionLedger(attribution_path).report(
            window=timedelta(days=args.window_days)
        ).to_dict()

    missing: list[str] = []
    if optimizer is None:
        missing.append("optimizer_report")
    if attribution_report is None and args.stage != "backtest":
        missing.append("attribution_ledger")

    accepted = [] if optimizer is None else optimizer.get("accepted_configs", [])
    checks = {
        "has_accepted_candidate": bool(accepted),
        "attribution_present": attribution_report is not None,
        "thresholds": {key: str(value) for key, value in thresholds.items()},
    }
    go = not missing and checks["has_accepted_candidate"]
    if attribution_report is not None:
        metrics = attribution_report["metrics"]
        trade_count = int(metrics["trade_count"])
        profit_factor = metrics["profit_factor"]
        profit_factor_value = (
            Decimal("Infinity")
            if profit_factor == "Infinity"
            else Decimal(str(profit_factor or "0"))
        )
        drawdown = Decimal(str(metrics["max_drawdown_usdt"]))
        starting = Decimal(str(metrics["starting_equity_usdt"]))
        checks["min_trades_passed"] = trade_count >= thresholds["min_trades"]
        checks["profit_factor_passed"] = profit_factor_value >= thresholds["min_profit_factor"]
        checks["drawdown_passed"] = drawdown <= starting * thresholds["max_drawdown_pct"]
        checks["alerts"] = attribution_report["alerts"]
        go = (
            go
            and checks["min_trades_passed"]
            and checks["profit_factor_passed"]
            and checks["drawdown_passed"]
            and not checks["alerts"]
        )

    return {
        "stage": args.stage,
        "go": bool(go),
        "missing_evidence": missing,
        "checks": checks,
        "accepted_configs": accepted,
        "attribution_report": attribution_report,
    }


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = build_parser().parse_args()
    payload = build_evidence_packet(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
