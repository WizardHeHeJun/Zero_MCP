"""桌面屏幕能力 MCP 子包（Python 侧内部封装层）。

包含：
- capability_probe：启动时探测硬件/库能力并缓存 CapabilityFlags。
- tools/perception：感知工具（uiautomation/mss/RapidOCR）。
- tools/control：操控工具（pyautogui/pyperclip/win32）。

本包仅供 src/mcp/desktop_mcp_server.py 使用，不对外暴露给编排层。
"""
