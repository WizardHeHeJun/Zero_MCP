"""屏幕感知/操控的核心数据模型（蓝图 §3，签名级设计的落地）。

本文件是 MCP server/client 与 Agent 层数据形状契约的**唯一真相**（蓝图 §7.3：
server 与 client 同为 Python，单侧 pydantic 校验即可）。坐标统一为物理像素，
DPI 换算在 MCP server 侧完成。

Task 1 实测修正（notes/poc-uia-coverage-result.md）：微信 4.x 等 mmui 自绘应用
UIA 内容树为空，`ScreenSnapshot.uia_hollow` 标记该状态，感知侧据此自动切
OCR 主通道，消费侧不应假设 `uia_elements` 非空。
"""

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel


class BBox(BaseModel):
    """边界框，物理像素（蓝图 §7.3：{x,y,width,height} 统一口径）。"""

    x: int
    y: int
    width: int
    height: int


class UIAElement(BaseModel):
    """UIA 树节点（L1 感知层产物）。"""

    element_id: str
    control_type: str  # Button / Edit / List
    name: str
    automation_id: str | None
    bbox: BBox
    is_enabled: bool
    is_visible: bool
    value: str | None
    source: Literal["uia"]


class TextBlock(BaseModel):
    """OCR 文本块（L2 感知层产物）。"""

    block_id: str
    text: str
    bbox: BBox
    confidence: float
    source: Literal["ocr_rapidocr", "ocr_windows_media"]


class VisualObject(BaseModel):
    """视觉检测对象（L3 感知层产物：模板匹配 / OmniParser）。"""

    object_id: str
    label: str
    bbox: BBox
    confidence: float
    source: Literal["opencv_template", "omniparser"]


class ScreenSnapshot(BaseModel):
    """一次屏幕感知的完整快照。

    本体不进编排 state——state 只存 `snapshot_id` 引用（orchestration-rules）。
    """

    snapshot_id: str  # 唯一 ID 进 state；对象本体存 Postgres/磁盘
    timestamp_ms: int
    screen_width: int
    screen_height: int
    active_window_title: str | None
    uia_elements: list[UIAElement]
    text_blocks: list[TextBlock]
    visual_objects: list[VisualObject]
    screenshot_path: str | None
    perception_mode: Literal["uia_only", "uia_ocr", "full"]
    capability_flags: dict[str, bool]
    is_untrusted: bool = True  # [v2 WARN-1] 屏幕内容一律不可信，读取侧需再过滤
    # Task 1 实测：目标窗口 UIA 内容树为空（如微信 4.x mmui 自绘）。
    # True 时 uia_elements 仅含窗口帧节点，文字/控件 grounding 依赖 L2/L3。
    uia_hollow: bool = False
    # Task 12：screenshot_path 图像坐标系原点的屏幕绝对坐标。全屏截图为 (0,0)；
    # PrintWindow 窗口捕获为窗口左上角。消费方（如 TOCTOU 局部裁剪）用它把
    # 屏幕绝对坐标换算为图像坐标。
    capture_origin: tuple[int, int] = (0, 0)


class ActionRisk(StrEnum):
    """动作危险级别（安全门三级白名单的判定输入，蓝图 §8）。"""

    READ_ONLY = "read_only"
    LOW_RISK = "low_risk"  # 可逆：移动鼠标、只读截图
    DESTRUCTIVE = "destructive"  # 高危：删除、提交表单、系统设置


class ActionSpec(BaseModel):
    """单步操控动作指令。

    Task 1 实测：UIA 空洞窗口无元素句柄可 invoke，`target_element_id` 允许为
    None、以 `coordinates`（来自 OCR/视觉 bbox）为主路径。
    """

    action_id: str
    action_type: str  # click / type / key / window_close
    target_element_id: str | None
    coordinates: tuple[int, int] | None
    text_payload: str | None
    risk_level: ActionRisk


class ActionResult(BaseModel):
    """动作执行结果。"""

    action_id: str
    success: bool
    error_message: str | None
    ui_changed: bool  # TOCTOU 检测用
