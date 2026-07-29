"""真 NeuroKit2 路径 eval（Task I · algo-lead 对抗性核验）。

marker：pytest.importorskip("neurokit2")——缺库自动 skip，不阻断 CI。

核心目标：
  用真 NeuroKit2 合成信号跑完整通道逻辑，验证「真路径出合法 ModalityPrior」且检测
  μa 是否具有判别力（区别于 mock 单测：这里 neurokit2 是真的）。

⚠ **EDA 侧的判别力不在本文件验**（2026-07-28 起）：EdaChannel 的度量是紧张性水平相对
基线的偏移，而 `nk.eda_simulate` 的合成信号**紧张性基本平坦、只变 SCR 数**——在合成信号上
比较不同 `scr_number` 无法证伪也无法证成该度量。这正是旧度量长期「合成全绿、真数据反号」的
成因（`notes/2026-07-28-wesad-eda-metric-invalidation.md`）。EDA 的判别力改由 WESAD 真被试
全会话回放验收：`evals/wesad_eda_v2_acceptance_gates.py`。本文件对 EDA 只保留**真路径合法性**
（含一条人工注入紧张性上升的方向性核验）。

环境约束：
  本环境（conda affective-expression）未装 cvxopt。EdaChannel 已不做 phasic 分解、不依赖
  neurokit2，故不受影响；HRV 侧不涉及 cvxEDA。

HRV 合法性测试（不要求度量有效性，验证真路径不崩 + 符号方向不反）：
  合成 ECG 暖机建立基线后，**同一实例内**改变心率看 μa 方向。⚠ 2026-07-29 起 HrvChannel
  同样是基线相对度量，跨实例比 μa 已无意义（各实例基线不同），旧的「每心率新建通道单发
  sense」形态已按语义重整。判别力仍只在 WESAD 真被试上验：
  `evals/wesad_hrv_v2_acceptance_gates.py`。
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

# 缺 neurokit2 整个文件 skip（不阻断 CI）
nk = pytest.importorskip("neurokit2")

from src.mcp.zero.channels.physio_channel import EdaChannel, HrvChannel  # noqa: E402
from src.mcp.zero.external_priors import MIN_PRECISION  # noqa: E402

# cvxEDA 分支（rate>4Hz）需 cvxopt；缺则相关 eval skip
_HAS_CVXOPT = importlib.util.find_spec("cvxopt") is not None

# ---------------------------------------------------------------------------
# 辅助：开 flag + 构造通道
# ---------------------------------------------------------------------------

_PHYSIO_FLAG_ENV = "ZERO_PHYSIO_CHANNEL_ENABLED"


class _StepClock:
    """可注入的确定性时钟：手动推进，让冷启动的**秒数**判定无需真等待。"""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture(autouse=True)
def _enable_physio(monkeypatch: pytest.MonkeyPatch) -> None:
    """所有本文件测试自动开 ZERO_PHYSIO_CHANNEL_ENABLED（默认关，零回归）。"""
    monkeypatch.setenv(_PHYSIO_FLAG_ENV, "true")


# ---------------------------------------------------------------------------
# EDA 真路径合法性（判别力见 evals/wesad_eda_v2_acceptance_gates.py）
# ---------------------------------------------------------------------------


async def _warm_and_sense(
    ch: EdaChannel, clock: _StepClock, signal: np.ndarray, rate: int
) -> object:
    """喂一窗并推进时钟，返回本窗读数。"""
    result = await ch.sense(signal={"eda": signal, "sampling_rate": rate})
    clock.advance(200.0)
    return result


class TestEdaChannelRealPathValidity:
    """真 nk 合成信号经 EdaChannel → 先验字段全部合法；冷启动语义在真路径上同样成立。"""

    async def test_cold_start_then_valid_prior(self) -> None:
        """前若干窗返 None（无基线证据），攒够后产出字段合法的先验。"""
        rate = 4
        clock = _StepClock()
        ch = EdaChannel(sampling_rate=rate, clock=clock)
        sig = nk.eda_simulate(duration=60, sampling_rate=rate, scr_number=5, random_state=0)

        assert await _warm_and_sense(ch, clock, sig, rate) is None, "首窗应因无基线返回 None"
        for _ in range(2):
            await _warm_and_sense(ch, clock, sig, rate)

        result = await ch.sense(signal={"eda": sig, "sampling_rate": rate})
        assert result is not None
        assert result.modality == "eda/sc"
        assert result.precision[0] == pytest.approx(MIN_PRECISION)
        assert result.mu[0] == pytest.approx(0.0)
        assert -1.0 <= result.mu[1] <= 1.0
        assert result.precision[1] > 0.0

    async def test_tonic_rise_reads_higher_than_flat(self) -> None:
        """在真合成信号上叠加**紧张性上升**（+1 μS）→ 读数显著高于同信号未叠加时。

        判别性：这条锁的是「度量跟随紧张性水平」这一根本方向。若哪天实现又退回相位/
        z-score 类度量（旧度量的失效模式），叠加的直流分量会被消掉，本例即失败。
        μS 量级与 `_SCL_DELTA_REF_US=1.0` 对齐，故 +1 μS 期望饱和到 +1.0。
        """
        rate = 4
        base = nk.eda_simulate(duration=60, sampling_rate=rate, scr_number=5, random_state=0)

        clock = _StepClock()
        ch = EdaChannel(sampling_rate=rate, clock=clock)
        for _ in range(3):
            await _warm_and_sense(ch, clock, base, rate)

        flat = await ch.sense(signal={"eda": base, "sampling_rate": rate})
        clock.advance(200.0)
        risen = await ch.sense(signal={"eda": base + 1.0, "sampling_rate": rate})

        assert flat is not None and risen is not None
        assert risen.mu[1] > flat.mu[1], "紧张性上升未被读到（度量方向错）"
        assert risen.mu[1] == pytest.approx(1.0), "+1 μS 应达 delta_ref 饱和"


# ---------------------------------------------------------------------------
# HRV 合法性核验（不要求判别力，验真路径不崩）
# ---------------------------------------------------------------------------


def _ecg(heart_rate: int, rate: int = 256, duration: int = 30, seed: int = 42) -> np.ndarray:
    """合成一段 ECG（duration 秒保证足够 R-R 间期数——RMSSD 需多个心跳）。"""
    return np.asarray(
        nk.ecg_simulate(
            duration=duration, sampling_rate=rate, heart_rate=heart_rate, random_state=seed
        ),
        dtype=np.float64,
    )


async def _warm_hrv(ch: HrvChannel, clock: _StepClock, sig: np.ndarray, rate: int) -> None:
    """按注入时钟喂若干窗，把通道推过冷启动双门（≥2 窗且跨度 ≥ horizon×coverage=270s）。

    时钟步长 100s 只为跨过覆盖率门，与真实采集节拍无关（生产里步长 ≈ 窗长）。
    """
    for _ in range(3):
        await ch.sense(signal={"ecg_or_ppg": sig, "sampling_rate": rate})
        clock.advance(100.0)


class TestHrvChannelRealValidity:
    """HrvChannel 真 NeuroKit2 路径：合成 ECG → 出合法 ModalityPrior，方向不反。

    ⚠ **2026-07-29 语义重整**：改造前这几条是「每个心率新建一个通道、单发 sense、**跨实例**
    比 μa」。基线相减后跨实例比较**语义上已失效**——各实例的基线由它自己看过的历史决定，
    两个实例的 μa 是相对不同零点的量，比大小没有意义（且单发必返回 None）。后继形态是
    **同一实例内**的方向性守卫：先暖机建立基线，再改变心率看 μa 怎么动。这比原来的
    「跨实例 ±0.1 容忍」**更强**——可以要求**严格**大于/小于，无需容忍阈值。

    ⚠ **这里是 `nk.ecg_simulate` 合成信号**，只能证「真路径不崩 + 方向不反」，
    **不得**用于标定任何常量、也**不构成度量有效性证据**：`heart_rate` 参数不等于 arousal，
    本类依赖的只是合成信号自带的「高心率 → 短 R-R → 低 RMSSD」这一机械关系。
    度量有效性由 WESAD 真被试验收：`evals/wesad_hrv_v2_acceptance_gates.py`。
    （EDA v1 正是在合成信号上长期全绿、接真数据后才暴露反号。）
    """

    @pytest.mark.parametrize(
        "heart_rate,label",
        [
            (60, "安静 60bpm"),
            (90, "中等 90bpm"),
            (120, "激活 120bpm"),
        ],
    )
    async def test_hrv_real_ecg_produces_valid_prior(self, heart_rate: int, label: str) -> None:
        """真 ECG（不同心率）→ 暖机后 → 合法 ModalityPrior，且**绝对水平无关**。

        暖机与读数用同一段信号 ⇒ Δ=0 ⇒ μa **恒为 0.0**，无论该心率下 RMSSD 的绝对值是多少。
        这一条同时是跨被试可比性在真路径上的直接体现（v1 的固定 ref 下三档心率会给出三个
        差异极大的读数：实测 0.579 / 0.852 / 0.928）。
        """
        rate = 256
        clock = _StepClock()
        ch = HrvChannel(sampling_rate=rate, clock=clock)
        ecg_sig = _ecg(heart_rate, rate=rate)

        assert await ch.sense(signal={"ecg_or_ppg": ecg_sig, "sampling_rate": rate}) is None, (
            f"{label}：首窗应因无基线返回 None"
        )
        clock.advance(100.0)
        await _warm_hrv(ch, clock, ecg_sig, rate)

        result = await ch.sense(signal={"ecg_or_ppg": ecg_sig, "sampling_rate": rate})

        print(
            f"\n[HRV 合法性 | {label}]\n"
            f"  modality = {result.modality if result else None}\n"
            f"  μa = {result.mu[1] if result else None}\n"
            f"  Πv = {result.precision[0] if result else None}"
        )

        assert result is not None, f"{label}：暖机后 HrvChannel 未产出 ModalityPrior（路径崩溃）"
        assert result.modality == "hrv/rmssd", f"modality 应为 hrv/rmssd，实际 {result.modality}"
        assert result.precision[0] == pytest.approx(MIN_PRECISION), "Πv 应 == MIN_PRECISION"
        assert result.mu[0] == pytest.approx(0.0), "μv 应 == 0.0（valence 盲）"
        assert -1.0 <= result.mu[1] <= 1.0, f"μa={result.mu[1]} 超出 [-1,1]"
        assert result.precision[1] > 0.0, "Πa 应 > 0"
        assert result.mu[1] == pytest.approx(0.0, abs=1e-9), (
            f"{label}：信号未变（Δ=0）却读出 μa={result.mu[1]}——度量不再是「相对自身基线」，"
            "绝对水平又漏进了读数"
        )

    async def test_hrv_direction_within_one_instance(self) -> None:
        """**同一实例内**方向性：基线建于 60bpm，喂 120bpm → μa **严格上升**；反向亦然。

        依据：高心率 → 短 R-R 间期 → RMSSD 降低；本度量 Δ = 基线 − 当前 ⇒ μa 升。
        这是符号方向的真路径守卫（另两重：`_process` 就近注释 + 单测
        `TestHrvSignDirection` + evals 的 G-Sign 门）。
        判别力已实证：把产品码 `baseline - rmssd_ms` 反号为 `rmssd_ms - baseline`，本例在
        第一条方向断言处即红（实测 60bpm 基线 → 120bpm 读出 **−0.174** 而非 +0.174；
        pytest 在首条断言失败即终止，故第二个方向在同一次运行里观测不到，但两侧读数严格
        互为相反数，反向断言必同红）。

        ⚠ 不是度量有效性证据（见类 docstring）：`heart_rate` ≠ arousal。
        """
        rate = 256
        calm, active = _ecg(60, rate=rate), _ecg(120, rate=rate)

        # ① 静息基线 → 高心率：μa 必须严格上升（且越过中性 0）
        clock_up = _StepClock()
        ch_up = HrvChannel(sampling_rate=rate, clock=clock_up)
        await _warm_hrv(ch_up, clock_up, calm, rate)
        calm_read = await ch_up.sense(signal={"ecg_or_ppg": calm, "sampling_rate": rate})
        clock_up.advance(100.0)
        active_read = await ch_up.sense(signal={"ecg_or_ppg": active, "sampling_rate": rate})

        # ② 高心率基线 → 静息：μa 必须严格下降（反向对照，排除「读数恒正」这类退化解释）
        clock_down = _StepClock()
        ch_down = HrvChannel(sampling_rate=rate, clock=clock_down)
        await _warm_hrv(ch_down, clock_down, active, rate)
        active_base = await ch_down.sense(signal={"ecg_or_ppg": active, "sampling_rate": rate})
        clock_down.advance(100.0)
        calm_after = await ch_down.sense(signal={"ecg_or_ppg": calm, "sampling_rate": rate})

        assert calm_read is not None and active_read is not None
        assert active_base is not None and calm_after is not None
        print(
            f"\n[HRV 同实例方向性] 60bpm 基线：{calm_read.mu[1]:+.6f} → 120bpm "
            f"{active_read.mu[1]:+.6f}\n"
            f"                    120bpm 基线：{active_base.mu[1]:+.6f} → 60bpm "
            f"{calm_after.mu[1]:+.6f}"
        )

        assert active_read.mu[1] > calm_read.mu[1], (
            f"心率 60→120（RMSSD 降）μa 未上升：{calm_read.mu[1]:.6f} → {active_read.mu[1]:.6f}"
            "——Δ 的减法方向很可能写反了（EDA v1 系统性反号病的复刻形态）"
        )
        assert active_read.mu[1] > 0.0, "相对静息基线的高心率窗应读出**正**唤醒"
        assert calm_after.mu[1] < active_base.mu[1], (
            f"心率 120→60（RMSSD 升）μa 未下降：{active_base.mu[1]:.6f} → {calm_after.mu[1]:.6f}"
            "——低唤醒半边可能又被单侧地板压平了"
        )
        assert calm_after.mu[1] < 0.0, "相对高心率基线的静息窗应读出**负**唤醒（地板已删）"


# ---------------------------------------------------------------------------
# EDA 通道禁用时真信号也返回 None（确保 flag 守卫不因真信号绕过）
# ---------------------------------------------------------------------------


class TestEdaChannelDisabledWithRealSignal:
    """flag 关闭时即使传入真合成信号也返回 None。"""

    async def test_disabled_flag_blocks_real_signal(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """ZERO_PHYSIO_CHANNEL_ENABLED=false → sense 不调用 neurokit2，直接返回 None。"""
        monkeypatch.setenv(_PHYSIO_FLAG_ENV, "false")
        rate = 16
        sig = nk.eda_simulate(duration=10, sampling_rate=rate, scr_number=5, random_state=9)
        ch = EdaChannel(sampling_rate=rate)
        result = await ch.sense(signal={"eda": sig, "sampling_rate": rate})
        assert result is None
