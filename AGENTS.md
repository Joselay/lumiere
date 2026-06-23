# AGENTS.md

For OKX integration inside this codebase, always use `python-okx==0.4.1`; never hand-roll raw OKX HTTP/signing calls. Docs/source: `~/.cache/checkouts/github.com/okxapi/python-okx`.

For OKX manual account checks or operations requested in chat (orders, balances, positions, P&L, placing/canceling orders, etc.), use the OKX CLI, not project SDK scripts. Use the local cached `okx/agent-skills` docs at `~/.cache/checkouts/github.com/okx/agent-skills` and follow the relevant skill instructions.
