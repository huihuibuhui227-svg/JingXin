"""《衡才智析》标准 MCP（Model Context Protocol）校招岗位与行为证据对照服务包。

支持两种接入形态：
1. 标准 JSON-RPC 2.0 `stdio` MCP Server（`python -m mcp_career.server`）；
2. Flask HTTP JSON-RPC 与便捷 REST 桥接端点（`app.py` `/api/mcp/career` 等）。

设计约束：严格遵循「不给分，只给证据与缺口」，不计算岗位匹配分，也不下推荐结论。
"""
from __future__ import annotations

from .server import CareerMcpServer, handle_jsonrpc

__all__ = ["CareerMcpServer", "handle_jsonrpc"]
