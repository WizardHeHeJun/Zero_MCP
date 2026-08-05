"""屏幕感知/操控的核心数据模型（蓝图 §3，签名级设计的落地）。

本文件是 MCP server/client 与 Agent 层数据形状契约的**唯一真相**（蓝图 §7.3：
server 与 client 同为 Python，单侧 pydantic 校验即可）。坐标统一为物理像素，
DPI 换算在 MCP server 侧完成。

Task 1 实测修正（notes/poc-uia-coverage-result.md）：微信 4.x 等 mmui 自绘应用
UIA 内容树为空，`ScreenSnapshot.uia_hollow` 标记该状态，感知侧据此自动切
OCR 主通道，消费侧不应假设 `uia_elements` 非空。

桌面加固（feat/desktop-hardening）契约增量——全部带默认值，旧 payload
反序列化零回归：

- `ScreenSnapshot.desktop_locked`：桌面会话锁定标记（OpenInputDesktop 探测），
  锁屏下像素/OCR 不可信；
- `ScreenSnapshot.window_captured`：本快照是 PrintWindow 窗口直取（True）还是
  全屏截图口径（False），像素消费方据此选坐标换算与遮挡假设；
- `ScreenSnapshot.degradations`：本次感知的机读降级枚举清单（已定枚举值见
  `ScreenSnapshot` 类 docstring）；
- `ActionSpec.expected_root_hwnd`：坐标点击的期望落点顶层窗口句柄，None=不核验
  （仅主窗元素点击设期望值，菜单点击勿设，语义见 `ActionSpec` 类 docstring）。
"""

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field


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

    降级枚举（`degradations` 的已定机读值，消费方按集合成员判断、不解析散文）：

    - ``desktop_locked``：桌面会话锁定（OpenInputDesktop 探测），像素/OCR 不可信；
    - ``lock_probe_failed``：锁定探测本身失败（Win32 调用异常），锁定状态未知；
    - ``window_capture_failed``：PrintWindow 窗口直取失败，已回退全屏截图口径；
    - ``screenshot_failed``：截图完全失败，本快照无像素产物；
    - ``mss_unavailable``：mss 截图后端不可用；
    - ``ocr_error``：OCR 执行抛错，text_blocks 不完整；
    - ``ocr_unavailable``：OCR 引擎不可用，text_blocks 为空；
    - ``ocr_crop_invalid``：OCR 裁剪区域无效（越界/零面积），该区域跳过。
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
    # Task 12/13：screenshot_path 图像坐标系原点的屏幕绝对坐标。mss 全虚拟屏
    # 截图为虚拟屏 origin（显示器排在主屏左/上方时为**负值**）；PrintWindow
    # 窗口捕获为窗口左上角。消费方（如 TOCTOU 局部裁剪）用它把屏幕绝对坐标
    # 换算为图像坐标：image_xy = screen_xy - capture_origin。
    capture_origin: tuple[int, int] = (0, 0)
    # feat/desktop-hardening：桌面会话锁定标记。感知侧经 OpenInputDesktop 探测
    # 当前输入桌面（打不开/非交互桌面即视为锁定）。True 表示采样发生在锁屏/
    # 安全桌面下——实测锁屏下 mss 采到的是锁屏前的**旧帧**、PrintWindow 常整幅
    # 黑屏，本快照像素与 OCR 结果均不可信，消费方不应据此做 grounding/落点判定。
    desktop_locked: bool = False
    # feat/desktop-hardening：本快照 screenshot_path 的成像口径。True=PrintWindow
    # 直取目标窗口渲染面（含被遮挡/后台部分，capture_origin=窗口左上角）；
    # False=全屏截图（或其裁剪，capture_origin=虚拟屏 origin）。TOCTOU/stall 等
    # 像素消费方据此知道图的口径，选择坐标换算与遮挡假设。
    window_captured: bool = False
    # feat/desktop-hardening：本次感知发生过的机读降级枚举清单（已定枚举值见
    # 类 docstring）；空列表=各通道健康、无降级。
    degradations: list[str] = Field(default_factory=list)


class ActionRisk(StrEnum):
    """动作危险级别（安全门三级白名单的判定输入，蓝图 §8）。"""

    READ_ONLY = "read_only"
    LOW_RISK = "low_risk"  # 可逆：移动鼠标、只读截图
    DESTRUCTIVE = "destructive"  # 高危：删除、提交表单、系统设置


class ActionSpec(BaseModel):
    """单步操控动作指令。

    Task 1 实测：UIA 空洞窗口无元素句柄可 invoke，`target_element_id` 允许为
    None、以 `coordinates`（来自 OCR/视觉 bbox）为主路径。

    feat/desktop-hardening：`expected_root_hwnd` 是坐标点击的期望落点顶层窗口
    句柄，None=不核验（零回归）。**仅主窗元素点击设期望值**——应用内弹出菜单
    （Win32 类名 ``#32768`` 一类）是独立顶层窗口，菜单点击勿设，否则误拒：
    落点核验用 GA_ROOTOWNER 能把「有 owner 的主弹窗」兜回主窗，但无 owner 的
    菜单窗口仍会 mismatch。
    """

    action_id: str
    action_type: str  # click / type / key / window_close
    target_element_id: str | None
    coordinates: tuple[int, int] | None
    text_payload: str | None
    risk_level: ActionRisk
    # feat/desktop-hardening：期望落点顶层窗口句柄；None=不核验。语义与「菜单
    # 点击勿设」的误拒陷阱见类 docstring。
    expected_root_hwnd: int | None = None


class ActionResult(BaseModel):
    """动作执行结果。"""

    action_id: str
    success: bool
    error_message: str | None
    ui_changed: bool  # TOCTOU 检测用
