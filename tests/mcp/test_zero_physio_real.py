"""真 NeuroKit2 判别性 eval（Task I · algo-lead 对抗性核验）。

marker：pytest.importorskip("neurokit2")——缺库自动 skip，不阻断 CI。

核心目标：
  用真 NeuroKit2 合成信号经 EdaChannel / HrvChannel 跑完整通道逻辑，
  验证「真路径出合法 ModalityPrior」且检测 μa 是否具有判别力。

algo-lead 发现的潜在 bug（待本 eval 实证）：
  EdaChannel._process 用 `phasic_df["EDA_Phasic"].mean()` 作 SCR 幅度——
  经 nk.standardize 后 phasic 相位成分零均值，.mean() ≈ 0 可能使 μa 恒≈-1（退化）。

判别性测试（EDA）：
  高唤醒（scr_number=9）vs 低/平缓（scr_number=0）两路信号，
  断言 mu_a_high > mu_a_low（有判别力）。
  若高≈低 或 μa 钉死（都≈-1）→ 该断言失败 → 回报 algo-lead。

环境约束：
  本环境（conda affective-expression）未装 cvxopt，故 EDA 真路径测试固定使用
  sampling_rate=4（≤4Hz → highpass 分支，不需要 cvxopt）。
  cvxEDA 分支（rate>4）在 mock 单测（test_zero_physio_channel.py）已覆盖。

HRV 合法性测试（不要求判别力，验证真路径不崩）：
  不同心率合成 ECG，HrvChannel → 至少一条产出合法 ModalityPrior。
"""

from __future__ import annotations

import pytest

# 缺 neurokit2 整个文件 skip（不阻断 CI）
nk = pytest.importorskip("neurokit2")

from src.mcp.zero.channels.physio_channel import EdaChannel, HrvChannel  # noqa: E402
from src.mcp.zero.external_priors import MIN_PRECISION  # noqa: E402

# ---------------------------------------------------------------------------
# 辅助：开 flag + 构造通道
# ---------------------------------------------------------------------------

_PHYSIO_FLAG_ENV = "ZERO_PHYSIO_CHANNEL_ENABLED"


def _eda_ch(rate: int = 4) -> EdaChannel:
    """构造已开 flag 的 EdaChannel（不依赖 monkeypatch，直接 os.environ 在 autouse 里改）。

    默认 rate=4（≤4Hz → highpass 分支）——本环境缺 cvxopt，rate>4 的 cvxEDA 不可用。
    cvxEDA 分支在 mock 单测（test_zero_physio_channel.py E14）已覆盖。
    """
    return EdaChannel(sampling_rate=rate)


def _hrv_ch(rate: int = 256) -> HrvChannel:
    return HrvChannel(sampling_rate=rate)


@pytest.fixture(autouse=True)
def _enable_physio(monkeypatch: pytest.MonkeyPatch) -> None:
    """所有本文件测试自动开 ZERO_PHYSIO_CHANNEL_ENABLED。"""
    monkeypatch.setenv(_PHYSIO_FLAG_ENV, "true")


# ---------------------------------------------------------------------------
# EDA 判别性核验（algo-lead 对抗性 eval）
# ---------------------------------------------------------------------------


class TestEdaChannelDiscriminability:
    """真 NeuroKit2 EDA 判别性：高 SCR 数 vs 低 SCR 数 → μa 有差异。

    algo-lead 怀疑 bug：phasic.mean() ≈ 0（standardize 后零均值）→ μa 恒 ≈ -1（退化）。
    本 eval 实证真实情况，失败则证实 bug，回报 algo-lead（不改 src）。

    采样率固定 rate=4（≤4Hz → highpass 分支）：本环境缺 cvxopt，cvxEDA 不可用。
    highpass 分支的 EDA_Phasic 同样经过 standardize，零均值问题同样适用。
    """

    async def test_high_vs_low_scr_mu_a_discriminability(self) -> None:
        """高 SCR 数信号的 μa 严格大于低/平缓信号 μa（判别力核验，rate=4 highpass 分支）。

        若 mu_a_high ≈ mu_a_low 或两者均 ≈ -1，断言失败，回报：
        「μa 退化，归一化取 .mean() 有 bug，建议改量级度量
         (.abs().mean() / .std() / eda_peaks 幅度)」给 algo-lead。
        """
        sampling_rate = 4  # ≤4Hz → highpass，不需 cvxopt
        duration = 60  # 秒，低采样率需要更长时长让 SCR 发展

        # 高唤醒：scr_number=9（密集 SCR 响应）
        sig_high = nk.eda_simulate(
            duration=duration,
            sampling_rate=sampling_rate,
            scr_number=9,
            random_state=42,
        )
        # 低/平缓：scr_number=0（无 SCR，基线漂移）
        sig_low = nk.eda_simulate(
            duration=duration,
            sampling_rate=sampling_rate,
            scr_number=0,
            random_state=42,
        )

        ch = _eda_ch(rate=sampling_rate)

        result_high = await ch.sense(signal={"eda": sig_high, "sampling_rate": sampling_rate})
        result_low = await ch.sense(signal={"eda": sig_low, "sampling_rate": sampling_rate})

        # 两路均应产出合法先验
        assert result_high is not None, "高唤醒信号未产出 ModalityPrior，通道路径异常"
        assert result_low is not None, "低唤醒信号未产出 ModalityPrior，通道路径异常"

        mu_a_high = result_high.mu[1]
        mu_a_low = result_low.mu[1]

        # 打印供人读（eval 关键可见输出）
        print(
            f"\n[EDA 判别性 eval | rate={sampling_rate}Hz]\n"
            f"  高唤醒(scr=9) μa = {mu_a_high:.6f}\n"
            f"  低唤醒(scr=0) μa = {mu_a_low:.6f}\n"
            f"  差值 Δμa      = {mu_a_high - mu_a_low:.6f}"
        )

        # 合法性断言（恒成立，独立于判别力）
        assert result_high.precision[0] == pytest.approx(MIN_PRECISION), (
            "高唤醒先验 Πv 应 == MIN_PRECISION"
        )
        assert result_high.mu[0] == pytest.approx(0.0), "高唤醒先验 μv 应 == 0.0（valence 盲）"
        assert -1.0 <= mu_a_high <= 1.0, f"高唤醒 μa={mu_a_high} 超出 [-1,1]"
        assert -1.0 <= mu_a_low <= 1.0, f"低唤醒 μa={mu_a_low} 超出 [-1,1]"

        # 判别力回归守卫：SCR 量级度量（现用 .abs().mean() + ref 校准）须使高唤醒 μa 严格
        # 大于低唤醒。若退化（高≈低或均钉 -1），说明度量疑回退到零均值量（历史 bug 曾用 .mean()）。
        assert mu_a_high > mu_a_low, (
            f"μa 判别力退化：高唤醒 μa={mu_a_high:.6f} 未严格大于低唤醒 μa={mu_a_low:.6f}"
            f"（差值={mu_a_high - mu_a_low:.6f}）。EdaChannel SCR 量级度量疑回归，回报 algo-lead。"
        )


class TestEdaChannelPriorValidity:
    """EdaChannel 真路径合法性（不要求判别力，只验基本契约，rate=4 highpass 分支）。"""

    async def test_eda_real_signal_produces_valid_prior(self) -> None:
        """真信号经 EdaChannel → ModalityPrior 字段全部合法（rate=4，highpass）。"""
        rate = 4  # ≤4Hz → highpass，不需 cvxopt
        sig = nk.eda_simulate(duration=60, sampling_rate=rate, scr_number=5, random_state=0)
        ch = _eda_ch(rate=rate)
        result = await ch.sense(signal={"eda": sig, "sampling_rate": rate})

        assert result is not None
        assert result.modality == "eda/sc"
        assert result.precision[0] == pytest.approx(MIN_PRECISION)
        assert result.mu[0] == pytest.approx(0.0)
        assert -1.0 <= result.mu[1] <= 1.0
        # precision[1]（Πa）应 > 0
        assert result.precision[1] > 0.0

    async def test_eda_prints_actual_mu_a(self) -> None:
        """打印真实 μa 值供人工复查（不断言值，仅确保路径不崩；rate=4 highpass）。"""
        rate = 4  # ≤4Hz → highpass，不需 cvxopt
        for scr_n, label in [(0, "scr=0"), (4, "scr=4"), (9, "scr=9")]:
            sig = nk.eda_simulate(duration=60, sampling_rate=rate, scr_number=scr_n, random_state=1)
            ch = _eda_ch(rate=rate)
            result = await ch.sense(signal={"eda": sig, "sampling_rate": rate})
            mu_a = result.mu[1] if result else None
            print(f"  EdaChannel({label}) → μa = {mu_a}")
            assert result is not None, f"scr_n={scr_n} 未产出先验"


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
