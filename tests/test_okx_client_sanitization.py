from __future__ import annotations

import pytest

from lumiere.okx_client import OKXAPIError, _require_ok_response


def test_okx_error_redacts_api_key_and_ip_from_messages() -> None:
    response = {
        "code": "50110",
        "msg": (
            "Your IP 96.9.79.170 is not included in API key's "
            "7da67a75-5e17-4d05-bc93-b337f7f1bdef whitelist."
        ),
    }

    with pytest.raises(OKXAPIError) as exc_info:
        _require_ok_response(response)

    message = str(exc_info.value)
    assert "7da67a75-5e17-4d05-bc93-b337f7f1bdef" not in message
    assert "96.9.79.170" not in message
    assert "<redacted-id>" in message
    assert "<redacted-ip>" in message
