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

import importlib.util

import pytest

# 缺 neurokit2 整个文件 skip（不阻断 CI）
nk = pytest.importorskip("neurokit2")

from src.mcp.zero.channels.physio_channel import EdaChannel, HrvChannel  # noqa: E402
from src.mcp.zero.external_priors import MIN_PRECISION  # noqa: E402

# cvxEDA 分支（rate>4Hz）需 cvxopt；缺则相关 eval skip（highpass 分支已在 rate=4 覆盖）
_HAS_CVXOPT = importlib.util.find_spec("cvxopt") is not None

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
    """所有本文件测试自动开 ZERO_PHYSIO_CHANNEL_ENABLED，并把 EDA 度量钉到 **v1**。

    背景：蓝图任务 8 已把 `EdaChannel` 默认度量翻为 v2（`scl_baseline_delta_v2`）。
    本文件的 EDA 用例（SCR 幅度判别性 / standardize 增益不变性 / percentile 暖机与跨被试）
    **全部是 v1 特有行为**——v2 无 phasic 分解、无 percentile、语义完全不同。
    故显式钉 v1；不钉会让这些断言测到一个根本不存在对应行为的实现。

    v2 的真数据验证在 `evals/wesad_eda_v2_acceptance_gates.py`（对接真 EdaChannel 的验收门）。
    """
    monkeypatch.setenv(_PHYSIO_FLAG_ENV, "true")
    monkeypatch.setenv("ZERO_EDA_AROUSAL_METRIC", "scr_amplitude_v1")


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


@pytest.mark.skipif(not _HAS_CVXOPT, reason="cvxopt 未装，cvxEDA(rate>4) 路径不可用，跳过")
class TestEdaChannelCvxEDADiscriminability:
    """真 cvxEDA 路径（rate>4Hz）**分级**判别性核验（装 cvxopt 后才可真测）。

    背景：cvxEDA 反卷积出的 phasic 量级比 highpass 大 ~55×，若沿用 highpass 的 ref=0.6，
    μa 会恒饱和到 +1（scr=4/6/9 都读成 +1，丢失分级）。本 eval 断言**非饱和且单调**——
    旧版（cvxEDA 复用 0.6）会因 scr=4 与 scr=9 都饱和到 +1 而失败，锁定 cvxEDA 专属 ref 校准。
    """

    async def test_cvxeda_graded_monotonic(self) -> None:
        """rate=8（cvxEDA）下 scr=0<4<9 的 μa **严格单调且中档不饱和**。"""
        rate = 8  # >4Hz → cvxEDA 分支
        dur = 30
        results: dict[int, float] = {}
        for scr in (0, 4, 9):
            sig = nk.eda_simulate(duration=dur, sampling_rate=rate, scr_number=scr, random_state=42)
            ch = EdaChannel(sampling_rate=rate)
            r = await ch.sense(signal={"eda": sig, "sampling_rate": rate})
            assert r is not None, f"scr={scr}: cvxEDA 路径未产出 ModalityPrior"
            results[scr] = r.mu[1]

        print(
            f"\n[cvxEDA 分级判别 | rate=8]\n"
            f"  scr=0 μa={results[0]:+.4f}\n"
            f"  scr=4 μa={results[4]:+.4f}\n"
            f"  scr=9 μa={results[9]:+.4f}"
        )

        # 单调（判别力）
        assert results[9] > results[4] > results[0], (
            f"cvxEDA μa 非单调：{results}——疑 ref 未按 cvxEDA 校准（饱和丢分级），回报审查"
        )
        # 中档不饱和：若 cvxEDA 复用 highpass 的 0.6，scr=4 会钉 +1（与 scr=9 无从区分）
        assert results[4] < 0.9, (
            f"cvxEDA scr=4 μa={results[4]:.4f} 已饱和到接近 +1，丢失分级分辨力"
            "（cvxEDA 复用了 highpass 的 ref=0.6？应用 _SCR_REF_AMPLITUDE_CVXEDA）"
        )
        # 合法性
        for scr, mu_a in results.items():
            assert -1.0 <= mu_a <= 1.0, f"scr={scr} μa={mu_a} 越界"


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
# standardize 不变式守卫（路径回归防护）
# ---------------------------------------------------------------------------


class TestEdaChannelStandardizeInvariance:
    """standardize 不变式守卫：全局增益 k 与基线偏移 c 不影响 SCR 幅度度量。

    根因：`_process` 在 `nk.eda_phasic` 前做 `nk.standardize(eda)`（z-score）。
    恒等式 standardize(k·x)==standardize(x)、standardize(x+c)==standardize(x) →
    全局增益 k 与基线偏移 c 在进 eda_phasic 前被完全消除。
    现场核验：amp(base*k)==amp(base)，Δ≈1e-16，k=2/5/10 全等。

    这锁定「路径必经 standardize」的回归属性——若日后把 standardize 换成非尺度不变
    预处理（如 min-max、除以固定常数），此断言立即失败。这**不是冗余测试**，勿删：
    它同时纠正框架自述——percentile 适配的是 SCR 事件密度分布，非原始幅度。

    使用 normalization="linear"（无状态、逐次确定），隔离 standardize 行为，
    不受 percentile 历史/冷启动路径干扰。
    """

    async def test_gain_invariance_on_scr_amplitude(self) -> None:
        """全局增益 k 不影响 μa：amp(base*k)==amp(base)，Δ<1e-10，k∈[1,2,5,10]。

        现场核验：amp(base*k)==amp(base)，Δ≈1e-16（机器精度），k=2/5/10 全等。
        断言阈值放宽到 1e-10 容忍 float 累积误差，但远低于任何有意义的幅度差异。
        若断言失败，说明 _process 预处理不再经过 standardize（或引入了非尺度不变步骤），
        须回查 _process 并更新 percentile 选型依据（幅度轴鲁棒性恢复后 I1b 可重新评估）。
        """
        rate = 4
        duration = 60
        scr_number = 5
        base_sig = nk.eda_simulate(
            duration=duration,
            sampling_rate=rate,
            scr_number=scr_number,
            random_state=42,
        )

        # k=1.0 为基准
        ch_base = EdaChannel(sampling_rate=rate, normalization="linear")
        result_base = await ch_base.sense(signal={"eda": base_sig * 1.0, "sampling_rate": rate})
        assert result_base is not None, "基准信号（k=1）未产出 ModalityPrior"
        mu_a_base = result_base.mu[1]

        for k in [2.0, 5.0, 10.0]:
            ch_k = EdaChannel(sampling_rate=rate, normalization="linear")
            result_k = await ch_k.sense(signal={"eda": base_sig * k, "sampling_rate": rate})
            assert result_k is not None, f"k={k} 信号未产出 ModalityPrior"
            mu_a_k = result_k.mu[1]
            delta = abs(mu_a_k - mu_a_base)
            assert delta < 1e-10, (
                f"[standardize 增益不变式] k={k} 时 μa={mu_a_k:.15f}，"
                f"基准 μa={mu_a_base:.15f}，Δ={delta:.2e}（期望 <1e-10）。"
                "说明 _process 不再经过 nk.standardize 或引入了非尺度不变预处理，"
                "须回查 _process；percentile 的 SCR 事件密度适配假设也需重新评估。"
            )

    async def test_baseline_offset_invariance_on_scr_amplitude(self) -> None:
        """基线偏移 c 不影响 μa：amp(base+c)==amp(base)，Δ<1e-10，c∈[0,1,5,10]。

        z-score standardize 满足 standardize(x+c)==standardize(x)（平移不变），
        故基线漂移（传感器零点偏置/姿势漂移等）在 standardize 后被完全消除。
        现场核验：amp(base+c)==amp(base)，Δ≈1e-16，c=1/5/10 全等。
        断言阈值同增益测试（1e-10），理由同上。
        """
        rate = 4
        duration = 60
        scr_number = 5
        base_sig = nk.eda_simulate(
            duration=duration,
            sampling_rate=rate,
            scr_number=scr_number,
            random_state=42,
        )

        # c=0.0 为基准
        ch_base = EdaChannel(sampling_rate=rate, normalization="linear")
        result_base = await ch_base.sense(signal={"eda": base_sig + 0.0, "sampling_rate": rate})
        assert result_base is not None, "基准信号（c=0）未产出 ModalityPrior"
        mu_a_base = result_base.mu[1]

        for c in [1.0, 5.0, 10.0]:
            ch_c = EdaChannel(sampling_rate=rate, normalization="linear")
            result_c = await ch_c.sense(signal={"eda": base_sig + c, "sampling_rate": rate})
            assert result_c is not None, f"c={c} 信号未产出 ModalityPrior"
            mu_a_c = result_c.mu[1]
            delta = abs(mu_a_c - mu_a_base)
            assert delta < 1e-10, (
                f"[standardize 偏移不变式] c={c} 时 μa={mu_a_c:.15f}，"
                f"基准 μa={mu_a_base:.15f}，Δ={delta:.2e}（期望 <1e-10）。"
                "说明 _process 不再经过 nk.standardize 或引入了非平移不变预处理，"
                "须回查 _process；基线漂移鲁棒性假设需重新评估。"
            )


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


# ---------------------------------------------------------------------------
# percentile 归一化真行为 eval（真 NeuroKit2，R1-R2）
# ---------------------------------------------------------------------------


def _make_eda_signal_for_scr(scr_number: int, rate: int = 4, duration: int = 60) -> dict:
    """合成指定 SCR 数量的 EDA 信号 dict（真 nk.eda_simulate）。"""
    sig = nk.eda_simulate(
        duration=duration,
        sampling_rate=rate,
        scr_number=scr_number,
        random_state=42,
    )
    return {"eda": sig, "sampling_rate": rate}


async def _warmup_channel(
    ch: EdaChannel,
    scr_numbers: list[int],
    rate: int = 4,
    duration: int = 60,
) -> None:
    """向通道顺序喂入一批信号以完成暖机（超过 cold_start 样本数）。

    scr_numbers 中的值循环用于合成信号；每次使用不同 random_state 让信号有变化。
    """
    for i, scr in enumerate(scr_numbers):
        sig = nk.eda_simulate(
            duration=duration,
            sampling_rate=rate,
            scr_number=scr,
            random_state=i,
        )
        await ch.sense(signal={"eda": sig, "sampling_rate": rate})


class TestEdaChannelPercentileWarmupDiscriminability:
    """percentile 归一化：暖机后高 SCR 帧 μa 应显著大于低 SCR 帧（R1）。

    验证流程：
    1. 用混合 scr 档（0/3/6/9 轮转 >= cold_start 次）喂满暖机窗，历史已覆盖完整幅度区间。
    2. 再喂 scr=9（高唤醒）与 scr=1（低唤醒）两帧。
    3. 断言 μa(高) > μa(低)（自适应路径暖机后有判别力）。

    此测试用 rate=4（highpass 分支），不依赖 cvxopt。
    """

    async def test_r1_warmed_up_percentile_discriminability(self) -> None:
        """R1：暖机后 percentile 模式对 scr=9 vs scr=1 有判别力。

        暖机序列覆盖 scr∈{0,3,6,9}，确保历史跨越完整幅度区间；
        暖机后喂 scr=9 与 scr=1，断言自适应路径的 μa 有显著差异。
        """
        rate = 4
        cold_start = 20  # 刻意<window：不触恒退陷阱、缩短暖机；与产品默认40无关（验密度轴机制）
        # 暖机序列：轮转 scr={0,3,6,9}，共 cold_start+4 次，确保历史覆盖完整幅度区间
        warmup_scrs = [scr for i in range(cold_start + 4) for scr in [0, 3, 6, 9]][: cold_start + 4]

        # 两个独立通道用**相同**暖机序列（_warmup_channel 按 index 定 random_state → 确定性，
        # 两者历史完全一致），再分别喂高/低帧——避免单通道顺序依赖，且不复制私有历史/硬编码
        # maxlen（W1：改为公平地各自暖机，而非直接写内部 deque）。
        ch_high = EdaChannel(
            sampling_rate=rate,
            normalization="percentile",
            percentile_cold_start=cold_start,
            percentile_window=60,
        )
        ch_low = EdaChannel(
            sampling_rate=rate,
            normalization="percentile",
            percentile_cold_start=cold_start,
            percentile_window=60,
        )
        await _warmup_channel(ch_high, warmup_scrs, rate=rate)
        await _warmup_channel(ch_low, warmup_scrs, rate=rate)
        hist_len = len(ch_high._amplitude_history["highpass"])
        assert hist_len >= cold_start, f"暖机后历史长度={hist_len}，应 >= {cold_start}"

        # 喂高唤醒帧（scr=9）和低唤醒帧（scr=1），固定 random_state 保证可重复
        sig_high = nk.eda_simulate(duration=60, sampling_rate=rate, scr_number=9, random_state=99)
        sig_low = nk.eda_simulate(duration=60, sampling_rate=rate, scr_number=1, random_state=99)
        result_high = await ch_high.sense(signal={"eda": sig_high, "sampling_rate": rate})
        result_low = await ch_low.sense(signal={"eda": sig_low, "sampling_rate": rate})

        assert result_high is not None, "scr=9 帧未产出 ModalityPrior"
        assert result_low is not None, "scr=1 帧未产出 ModalityPrior"

        mu_a_high = result_high.mu[1]
        mu_a_low = result_low.mu[1]

        print(
            f"\n[R1 percentile 暖机判别 | rate={rate}Hz highpass]\n"
            f"  暖机样本数 = {hist_len}\n"
            f"  scr=9 μa  = {mu_a_high:.6f}\n"
            f"  scr=1 μa  = {mu_a_low:.6f}\n"
            f"  差值 Δμa  = {mu_a_high - mu_a_low:.6f}"
        )

        # 合法性
        assert -1.0 <= mu_a_high <= 1.0, f"scr=9 μa={mu_a_high} 超出 [-1,1]"
        assert -1.0 <= mu_a_low <= 1.0, f"scr=1 μa={mu_a_low} 超出 [-1,1]"
        assert result_high.mu[0] == pytest.approx(0.0), "μv 应 == 0.0（valence 盲）"
        assert result_high.precision[0] == pytest.approx(MIN_PRECISION), "Πv 应 == MIN_PRECISION"

        # 判别力守卫：要求显著 margin（实测 Δ≈1.64；退化/尺度不变 impl 的 Δ≈0 会跌破）
        assert mu_a_high > mu_a_low + 0.5, (
            f"[R1] percentile 暖机后判别力不足：scr=9 μa={mu_a_high:.6f} 未显著大于 "
            f"scr=1 μa={mu_a_low:.6f}（Δ={mu_a_high - mu_a_low:.6f}≤0.5）。"
            "自适应路径暖机后应显著区分高低 SCR（退化/尺度不变会跌破此 margin），回报 algo-lead。"
        )


@pytest.mark.skipif(not _HAS_CVXOPT, reason="cvxopt 未装，cvxEDA(rate>4) 路径不可用，跳过")
class TestEdaChannelPercentileCvxEDAWarmup:
    """percentile 归一化 cvxEDA 分支（rate>4Hz）暖机后判别力（R1 续，需 cvxopt）。"""

    async def test_r1_cvxeda_warmed_up_discriminability(self) -> None:
        """R1-cvx：cvxEDA 路径暖机后 scr=9 vs scr=1 μa 有判别力。"""
        rate = 8  # >4Hz → cvxEDA 分支
        cold_start = 20  # 刻意<window：不触恒退陷阱、缩短暖机；与产品默认40无关（验密度轴机制）
        warmup_scrs = [scr for i in range(cold_start + 4) for scr in [0, 3, 6, 9]][: cold_start + 4]

        # 两通道相同暖机序列（确定性→历史一致），分别喂高/低帧（W1：不复制私有历史）
        ch_high = EdaChannel(
            sampling_rate=rate,
            normalization="percentile",
            percentile_cold_start=cold_start,
            percentile_window=60,
        )
        ch_low = EdaChannel(
            sampling_rate=rate,
            normalization="percentile",
            percentile_cold_start=cold_start,
            percentile_window=60,
        )
        await _warmup_channel(ch_high, warmup_scrs, rate=rate, duration=30)
        await _warmup_channel(ch_low, warmup_scrs, rate=rate, duration=30)
        hist_len = len(ch_high._amplitude_history["cvxEDA"])
        assert hist_len >= cold_start, f"cvxEDA 暖机后历史={hist_len}，应 >= {cold_start}"

        sig_high = nk.eda_simulate(duration=30, sampling_rate=rate, scr_number=9, random_state=99)
        sig_low = nk.eda_simulate(duration=30, sampling_rate=rate, scr_number=1, random_state=99)
        result_high = await ch_high.sense(signal={"eda": sig_high, "sampling_rate": rate})
        result_low = await ch_low.sense(signal={"eda": sig_low, "sampling_rate": rate})

        assert result_high is not None and result_low is not None
        mu_a_high = result_high.mu[1]
        mu_a_low = result_low.mu[1]

        print(
            f"\n[R1-cvx cvxEDA 暖机判别 | rate={rate}Hz]\n"
            f"  scr=9 μa = {mu_a_high:.6f}\n"
            f"  scr=1 μa = {mu_a_low:.6f}\n"
            f"  差值 Δμa = {mu_a_high - mu_a_low:.6f}"
        )

        assert mu_a_high > mu_a_low + 0.5, (
            f"[R1-cvx] cvxEDA percentile 暖机后判别力不足：scr=9 μa={mu_a_high:.6f} "
            f"未显著大于 scr=1 μa={mu_a_low:.6f}（Δ={mu_a_high - mu_a_low:.6f}≤0.5）"
        )


class TestEdaChannelPercentileCrossSubjectRobustness:
    """percentile 归一化跨被试鲁棒性验证（R2）。

    核心命题（修正）：跨被试轴 = **SCR 事件密度**（scr_number），非幅度增益。
    因 per-call standardize（z-score）已消除全局幅度差与基线偏移（现场核验
    amp(base·k)==amp(base)，Δ≈1e-16），合成信号下唯一存活的跨被试轴是密度差异。
    percentile 自适应把不同密度分布的被试拉到可比标度；linear 在两被试间密度差距
    明显更大。

    实验设计：
    - 被试 A（高密度响应者）：暖机序列 scr 档偏高（{4,6,9} 轮转），历史密度大。
    - 被试 B（低密度响应者）：暖机序列 scr 档偏低（{0,1,2} 轮转），历史密度小。
    - 各自喂各自"高档"密度帧（A:scr=9 / B:scr=3），对各被试而言都是高唤醒。
    - percentile 下两者「高档帧」μa 应接近（自适应按各自密度分布归一化）。
    - linear 下两者同帧密度差距应明显大于 percentile（证明 adaptive 改善跨被试可比性）。

    局限分类（I1a / I1b）：
    - I1a（密度轴跨被试自适应）：R1+R2 已在合成信号闭合。
    - I1b（全局幅度增益轴鲁棒性）：合成信号无法构造有意义对照——standardize 使输出
      精确相等（见 TestEdaChannelStandardizeInvariance），**正式 defer 至真被试数据**
      （见重校协议 notes/2026-07-21-eda-percentile-recalibration-protocol.md）。

    使用 rate=4（highpass 分支），不依赖 cvxopt。
    """

    async def test_r2_cross_subject_robustness(self) -> None:
        """R2：percentile 自适应缩小跨被试高档帧密度差；linear 密度差明显更大。

        A（高密度）与 B（低密度）各自喂自己的暖机序列后，分别喂各自的"高档帧"，
        断言：
          |μa_A_pct - μa_B_pct| < |μa_A_lin - μa_B_lin| - margin
        即 percentile 下的跨被试差异比 linear 至少小 margin（默认 0.1）。

        打印实测数值供人工复查；若两者差异相近则说明未真自适应，回报 algo-lead。

        局限（I1a/I1b 分类）：
        - I1a（密度轴跨被试自适应）：R1+R2 已在合成信号闭合——A/B 的密度分布差异
          被 percentile 压缩，linear 差距更大，断言成立。
        - I1b（全局幅度增益轴鲁棒性）：合成信号无法构造有意义对照（standardize 使
          amp(base·k)==amp(base)，幅度轴输出精确相等，见 TestEdaChannelStandardizeInvariance），
          **正式 defer 至真被试数据**；真数据须含 standardize 前原始信号 + 不同传感器
          增益/波形形态对照组（见重校协议）。
        """
        rate = 4
        cold_start = 20  # 刻意<window：不触恒退陷阱、缩短暖机；与产品默认40无关（验密度轴机制）
        duration = 60

        # ---- 被试 A（高密度响应者）：暖机序列 scr∈{4,6,9} ----
        warmup_a = [scr for i in range(cold_start + 4) for scr in [4, 6, 9]][: cold_start + 4]
        # ---- 被试 B（低密度响应者）：暖机序列 scr∈{0,1,2} ----
        warmup_b = [scr for i in range(cold_start + 4) for scr in [0, 1, 2]][: cold_start + 4]

        # percentile 通道：A 与 B 各自暖机（按各自密度分布建立历史）
        ch_a_pct = EdaChannel(
            sampling_rate=rate,
            normalization="percentile",
            percentile_cold_start=cold_start,
            percentile_window=60,
        )
        ch_b_pct = EdaChannel(
            sampling_rate=rate,
            normalization="percentile",
            percentile_cold_start=cold_start,
            percentile_window=60,
        )
        await _warmup_channel(ch_a_pct, warmup_a, rate=rate, duration=duration)
        await _warmup_channel(ch_b_pct, warmup_b, rate=rate, duration=duration)

        # 测试帧：A 喂 scr=9（高密度档），B 喂 scr=3（高密度档相对其历史范围）
        sig_a_high = nk.eda_simulate(
            duration=duration, sampling_rate=rate, scr_number=9, random_state=100
        )
        sig_b_high = nk.eda_simulate(
            duration=duration, sampling_rate=rate, scr_number=3, random_state=100
        )

        result_a_pct = await ch_a_pct.sense(signal={"eda": sig_a_high, "sampling_rate": rate})
        result_b_pct = await ch_b_pct.sense(signal={"eda": sig_b_high, "sampling_rate": rate})

        assert result_a_pct is not None and result_b_pct is not None

        mu_a_A_pct = result_a_pct.mu[1]
        mu_a_B_pct = result_b_pct.mu[1]
        diff_pct = abs(mu_a_A_pct - mu_a_B_pct)

        # linear 对照：用与 percentile 测试帧完全相同的信号跑 linear
        ch_a_lin = EdaChannel(sampling_rate=rate, normalization="linear")
        ch_b_lin = EdaChannel(sampling_rate=rate, normalization="linear")
        result_a_lin = await ch_a_lin.sense(signal={"eda": sig_a_high, "sampling_rate": rate})
        result_b_lin = await ch_b_lin.sense(signal={"eda": sig_b_high, "sampling_rate": rate})

        assert result_a_lin is not None and result_b_lin is not None

        mu_a_A_lin = result_a_lin.mu[1]
        mu_a_B_lin = result_b_lin.mu[1]
        diff_lin = abs(mu_a_A_lin - mu_a_B_lin)

        print(
            f"\n[R2 跨被试鲁棒性 | rate={rate}Hz highpass]\n"
            f"  被试 A（高密度）暖机 scr∈{{4,6,9}}，测试帧 scr=9\n"
            f"  被试 B（低密度）暖机 scr∈{{0,1,2}}，测试帧 scr=3\n"
            f"  percentile:  μa_A={mu_a_A_pct:+.4f}, μa_B={mu_a_B_pct:+.4f}, "
            f"|差|={diff_pct:.4f}\n"
            f"  linear:      μa_A={mu_a_A_lin:+.4f}, μa_B={mu_a_B_lin:+.4f}, "
            f"|差|={diff_lin:.4f}\n"
            f"  diff_lin - diff_pct = {diff_lin - diff_pct:.4f}（>0 表示 adaptive 改善）"
        )
        print(
            "[R2 语义] 跨被试轴=SCR事件密度；幅度轴已被 per-call standardize 消除，需真数据闭合 I1b"
        )

        # 合法性断言（恒成立）
        assert -1.0 <= mu_a_A_pct <= 1.0, f"A percentile μa={mu_a_A_pct} 超出 [-1,1]"
        assert -1.0 <= mu_a_B_pct <= 1.0, f"B percentile μa={mu_a_B_pct} 超出 [-1,1]"

        # 核心命题：percentile 跨被试密度差 < linear 跨被试密度差（自适应改善跨被试可比性）
        # margin=0.1 容忍合成信号的量级噪声，但若两者差异相近（< margin）说明未真自适应
        margin = 0.1
        assert diff_pct < diff_lin - margin, (
            f"[R2] percentile 未改善跨被试可比性：\n"
            f"  percentile 跨被试差异={diff_pct:.4f}，linear 跨被试差异={diff_lin:.4f}\n"
            f"  期望 diff_pct < diff_lin - {margin}（即减少至少 {margin}），\n"
            f"  实际 diff_lin - diff_pct = {diff_lin - diff_pct:.4f}。\n"
            "若两者差异相近，说明 percentile 未真自适应，回报 algo-lead。"
        )
