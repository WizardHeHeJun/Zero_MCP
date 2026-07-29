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

HRV 合法性测试（不要求判别力，验证真路径不崩）：
  不同心率合成 ECG，HrvChannel → 至少一条产出合法 ModalityPrior。
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


def _hrv_ch(rate: int = 256) -> HrvChannel:
    return HrvChannel(sampling_rate=rate)


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


class TestHrvChannelRealValidity:
    """HrvChannel 真 NeuroKit2 路径：合成 ECG → 出合法 ModalityPrior，不崩。

    ECG 合成唤醒语义较弱（heart_rate 与 arousal 关联不如 EDA 直接），
    故合法性断言（physio 前缀、Πv==MIN、μa∈[-1,1]、μv==0）是主要契约；
    W5 补弱单调断言：高心率 μa ≥ 低心率 μa - 0.1，容忍 nk 版本 offset，
    但若反转逻辑整体方向坏了（如 RMSSD 取反丢失），差距会远超 0.1 而被捕获。
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
        """真 ECG（不同心率）→ HrvChannel → 合法 ModalityPrior。"""
        rate = 256
        duration = 30  # 秒，保证足够 R-R 间期数（RMSSD 需多个心跳）
        ecg_sig = nk.ecg_simulate(
            duration=duration,
            sampling_rate=rate,
            heart_rate=heart_rate,
            random_state=42,
        )
        ch = _hrv_ch(rate=rate)
        result = await ch.sense(signal={"ecg_or_ppg": ecg_sig, "sampling_rate": rate})

        print(
            f"\n[HRV 合法性 | {label}]\n"
            f"  modality = {result.modality if result else None}\n"
            f"  μa = {result.mu[1] if result else None}\n"
            f"  Πv = {result.precision[0] if result else None}"
        )

        assert result is not None, f"{label}：HrvChannel 未产出 ModalityPrior（路径崩溃）"
        assert result.modality == "hrv/rmssd", f"modality 应为 hrv/rmssd，实际 {result.modality}"
        assert result.precision[0] == pytest.approx(MIN_PRECISION), "Πv 应 == MIN_PRECISION"
        assert result.mu[0] == pytest.approx(0.0), "μv 应 == 0.0（valence 盲）"
        assert -1.0 <= result.mu[1] <= 1.0, f"μa={result.mu[1]} 超出 [-1,1]"
        assert result.precision[1] > 0.0, "Πa 应 > 0"

    async def test_hrv_monotone_direction_weak_assertion(self) -> None:
        """W5：高心率 μa ≥ 低心率 μa - 0.1（宽容单调断言，锁定 RMSSD 反转方向）。

        实测（random_state=42）：60→0.579、90→0.852、120→0.928，差值远大于 0.1。
        容忍阈值 0.1 允许 nk 版本间的 RMSSD 数值 offset，但若反转逻辑方向整体坏了
        （如取反符号丢失导致高心率反而映射低 μa），差距会超过 0.1 被捕获。
        """
        rate = 256
        duration = 30
        heart_rates = [60, 90, 120]
        mu_a_by_hr: dict[int, float] = {}

        for hr in heart_rates:
            ecg_sig = nk.ecg_simulate(
                duration=duration, sampling_rate=rate, heart_rate=hr, random_state=42
            )
            ch = _hrv_ch(rate=rate)
            result = await ch.sense(signal={"ecg_or_ppg": ecg_sig, "sampling_rate": rate})
            assert result is not None, f"hr={hr}bpm 未产出先验"
            mu_a_by_hr[hr] = result.mu[1]
            print(f"  HrvChannel(hr={hr}bpm, random_state=42) → μa = {mu_a_by_hr[hr]:.6f}")

        mu_a_60, mu_a_90, mu_a_120 = mu_a_by_hr[60], mu_a_by_hr[90], mu_a_by_hr[120]

        # W5 宽容单调断言：容忍 ±0.1 的 nk 版本 offset，锁定整体方向
        tolerance = 0.1
        assert mu_a_90 >= mu_a_60 - tolerance, (
            f"[W5 方向断言] hr=90 μa={mu_a_90:.4f} 应 ≥ hr=60 μa={mu_a_60:.4f} - {tolerance}；"
            "差值超出容忍范围，RMSSD 反转逻辑可能方向坏了"
        )
        assert mu_a_120 >= mu_a_90 - tolerance, (
            f"[W5 方向断言] hr=120 μa={mu_a_120:.4f} 应 ≥ hr=90 μa={mu_a_90:.4f} - {tolerance}；"
            "差值超出容忍范围，RMSSD 反转逻辑可能方向坏了"
        )
        assert mu_a_120 >= mu_a_60 - tolerance, (
            f"[W5 方向断言] hr=120 μa={mu_a_120:.4f} 应 ≥ hr=60 μa={mu_a_60:.4f} - {tolerance}；"
            "差值超出容忍范围，RMSSD 反转逻辑可能方向坏了"
        )

    async def test_hrv_different_heart_rates_print_mu_a(self) -> None:
        """打印不同心率对应 μa，供人工复查 HRV 反转逻辑（高心率≈低 RMSSD≈高 arousal）。

        W5：在打印基础上补弱单调断言——最高档 μa ≥ 最低档 μa - 0.1，
        确保反转逻辑方向整体正确；容忍 nk 版本间的 RMSSD 数值 offset。
        """
        rate = 256
        heart_rates = [60, 75, 90, 110]
        mu_a_values: list[float] = []

        for hr in heart_rates:
            ecg_sig = nk.ecg_simulate(
                duration=30, sampling_rate=rate, heart_rate=hr, random_state=0
            )
            ch = _hrv_ch(rate=rate)
            result = await ch.sense(signal={"ecg_or_ppg": ecg_sig, "sampling_rate": rate})
            assert result is not None, f"hr={hr}bpm 未产出先验"
            mu_a = result.mu[1]
            mu_a_values.append(mu_a)
            print(f"  HrvChannel(hr={hr}bpm) → μa = {mu_a:.6f}")

        # W5 宽容单调断言：最高心率(110) μa ≥ 最低心率(60) μa - 0.1
        # 实测：60→0.305、75→0.790、90→0.877、110→0.911，差值 ~0.6，远大于容忍阈值
        mu_a_lowest_hr = mu_a_values[0]  # hr=60
        mu_a_highest_hr = mu_a_values[-1]  # hr=110
        tolerance = 0.1
        assert mu_a_highest_hr >= mu_a_lowest_hr - tolerance, (
            f"[W5 方向断言] hr=110 μa={mu_a_highest_hr:.4f} 应 ≥ hr=60 μa={mu_a_lowest_hr:.4f} "
            f"- {tolerance}；差值超出容忍范围，RMSSD 反转逻辑（高心率→低 RMSSD→高 μa）可能坏了"
        )


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
