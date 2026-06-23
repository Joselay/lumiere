from __future__ import annotations

import okx
from okx import Account, MarketData, Trade
from okx.websocket.WsPrivateAsync import WsPrivateAsync
from okx.websocket.WsPublicAsync import WsPublicAsync


def test_project_is_pinned_to_latest_reviewed_python_okx_sdk() -> None:
    assert okx.__version__ == "0.4.1"
    assert Account.AccountAPI is not None
    assert MarketData.MarketAPI is not None
    assert Trade.TradeAPI is not None
    assert WsPrivateAsync is not None
    assert WsPublicAsync is not None
