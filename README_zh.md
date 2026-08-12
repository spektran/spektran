<div align="center">

[English](README.md) | **中文**

<img src="assets/logo.jpg" alt="SPEKTRAN" width="560">

### 气体传感领域的 MNIST

**AI Agent-Ready 光学气体传感仿真引擎与 ML 基准**<br>
*HITRAN 级物理精度。9 项任务。25 种基线。5 种模态。自然语言驱动全链条 ML 流水线。*

<br>

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/spektran/spektran/blob/main/notebooks/quickstart_colab.ipynb)
[![GitHub Stars](https://img.shields.io/github/stars/spektran/spektran?style=flat-square)](https://github.com/spektran/spektran/stargazers)
[![CI](https://img.shields.io/github/actions/workflow/status/spektran/spektran/ci.yml?branch=main&style=flat-square&label=CI)](https://github.com/spektran/spektran/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/spektran?style=flat-square&color=blue)](https://pypi.org/project/spektran/)
[![Python](https://img.shields.io/badge/python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://pypi.org/project/spektran/)
[![License](https://img.shields.io/badge/code-Apache%202.0-green?style=flat-square)](LICENSE)
[![License](https://img.shields.io/badge/data-CC%20BY%204.0-green?style=flat-square)](LICENSE-DATA)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.21790394-blue?style=flat-square)](https://doi.org/10.5281/zenodo.21790394)
[![AI Agent Ready](https://img.shields.io/badge/%F0%9F%A4%96_AI_Agent-Ready-blueviolet?style=flat-square)](AGENTS.md)

[**文档**](https://spektran.github.io/spektran/) &nbsp;&middot;&nbsp;
[**在线演示**](https://huggingface.co/spaces/spektran/spektran-demo) &nbsp;&middot;&nbsp;
[**排行榜**](https://spektran.github.io/spektran/leaderboard/) &nbsp;&middot;&nbsp;
[**数据集**](https://huggingface.co/spektran) &nbsp;&middot;&nbsp;
[**基线模型**](https://huggingface.co/spektran/spektran-baselines-v0)

</div>

<br>

> **你不需要是光谱学专家。** &nbsp;无论你从事回归、去噪、领域泛化还是异常检测研究，SPEKTRAN 都能提供 9 项开箱即用、基于真实物理构建的基准任务——具备 MNIST 般的易用性，却扎根于一个机器学习能够产生直接产业影响的真实领域。
>
> **你甚至不需要写代码。** &nbsp;SPEKTRAN 是完全的 **AI Agent-Ready** 项目——告诉 Claude Code、Cursor 或任何 AI 编程助手你想做什么，它就能通过自然语言操控整条流水线。详见 [AGENTS.md](AGENTS.md)。

<br>

## 核心亮点

<table>
<tr>
<td width="50%" valign="top">

### 仿真引擎
- **10 种气体分子** — CH4、H2O、CO2、CO、NH3、NO、NO2、SO2、HCl、HF
- **5 种检测模态** — TDLAS（DA + WMS）、NDIR、CRDS、FTIR、DOAS
- **先进线型** — Voigt 线型与 Hartmann-Tran 线型
- **46+ 种虚拟仪器**，配备真实噪声链路
- **WMS 1f–4f** 解调 + 2f/1f 免标定比值法

</td>
<td width="50%" valign="top">

### 机器学习基准
- **9 项任务（T1–T9）** — 回归、去噪、OOD、迁移、多组分
- **25 种基线模型** — Ridge、SpektralNet、CNN、Transformer、U-Net、RF、PINN 等
- **官方数据划分** — 训练集 / 验证集 / 测试集 / 留出仪器集
- **AI Agent-Ready CLI** — 全命令 `--json` 输出，可发现式 API
- **公开排行榜**，托管于 GitHub Pages

</td>
</tr>
</table>

<br>

## 快速开始

**零安装** — 一行代码即可从 Hugging Face 加载数据：

```python
from datasets import load_dataset
ds = load_dataset("spektran/spektran-ch4-v0")       # CH4 基准
ds = load_dataset("spektran/spektran-co2-v0", "da") # CO2 基准
ds = load_dataset("spektran/spektran-industrial-v0", "so2")  # 工业排放 SO2
ds = load_dataset("spektran/spektran-multigas-v0", "ch4_co2_h2o")  # 多气体混合
```

**完整引擎** — 在本地模拟光谱：

```bash
pip install spektran
```

```python
from spektran.physics import simulate_absorbance

nu, absorbance = simulate_absorbance(
    molecule="CH4", concentration_ppm=100.0,
    temperature_K=296.0, pressure_atm=1.0,
    path_length_m=10.0,
    wavenumber_start_cm1=6046.0, wavenumber_end_cm1=6048.0,
)
```

<details>
<summary><b>更多示例</b> — WMS、多组分、命令行工具</summary>

<br>

**WMS 2f 信号：**

```python
from spektran.physics.wms import WMSConfig, simulate_wms
```

完整 WMS 示例见 [`examples/wms_ch4.py`](examples/wms_ch4.py)。

**多组分（CH4 + H2O 干扰气体）：**

```python
from spektran.physics import absorption_coefficient
from spektran.physics.hitran import demo_ch4_2nu3, demo_h2o

alpha_ch4 = absorption_coefficient(nu, demo_ch4_2nu3(), 100e-6, 296.0, 1.0)
alpha_h2o = absorption_coefficient(nu, demo_h2o(), 0.01, 296.0, 1.0)
```

详见 [`examples/multispecies_ch4_h2o.py`](examples/multispecies_ch4_h2o.py)。

**命令行工具**（所有命令支持 `--json` 供 AI Agent 使用）：

```bash
spektran info --json                      # 项目发现（Agent 自举）
spektran list tasks --json                # 可用基准任务
spektran train --baseline ridge --json    # 自动生成数据并训练
spektran generate configs/datasets/ch4-t1-train-v0.yaml --out data --json
spektran benchmark --task T1-concentration --truth data/test.h5 --predictions preds.csv
```

</details>

<br>

## AI Agent Ready

SPEKTRAN 从底层架构就为 AI Agent 时代而设计。每条 CLI 命令都输出结构化 JSON，每项资源都可被发现，完整的 ML 流水线——**仿真 → 生成 → 训练 → 评测**——全程零手动操作。

**兼容**: Claude Code、Cursor、GitHub Copilot、Windsurf、Cline，以及一切能执行 Shell 命令的 AI Agent。

**Agent 接口文件**: [`AGENTS.md`](AGENTS.md) — Agent 读取此文件后即可理解并操控整个项目。

```
你: "用 ridge 基线在 T1 任务上训练，然后告诉我分数"

Agent: spektran train --baseline ridge --task T1 --json
       → {"baseline": "ridge", "scores": {"T1": {"mae_ppm": 2.84, "mape_pct": 29.87}}}
```

<details>
<summary><b>Agent 工作流示例</b></summary>

<br>

**发现** — Agent 自举启动：
```bash
spektran info --json           # 这个项目是什么？有哪些资源？
spektran list tasks --json     # 9 项任务及其指标、可用基线
spektran list baselines --json # 25 种基线及预计算分数
spektran status --json         # 已生成的数据和训练状态
```

**一键训练** — Agent 训练任何基线：
```bash
spektran train --baseline ridge --json        # 自动生成缺失数据
spektran train --baseline transformer --json  # 适用于任何已注册基线
spektran train --baseline cnn1d --task T1 --json  # 指定特定任务
```

**横向对比** — Agent 脚本化运行排行榜：
```bash
for baseline in ridge cnn1d transformer; do
  spektran train --baseline $baseline --task T1 --json
done
```

</details>

<br>

## 基准任务

| 任务 | 领域 | 主要指标 |
|:-----|:-------|:---------------|
| **T1** 浓度回归 | DA 光谱 → ppm | MAE |
| **T2** 光谱去噪 | 含噪光谱 → 干净光谱 | RMSE |
| **T3** 跨仪器泛化 | 留出仪器 | 相对 T1 的性能衰减 |
| **T4** WMS 浓度反演 | 2f 信号 → ppm | MAE |
| **T5** 漂移补偿 | 时间序列扫描 | 阿伦方差 |
| **T6** OOD 仪器检测 | 分布内 vs OOD | AUROC |
| **T7** 跨模态迁移 | TDLAS → NDIR | 相对 T1 的性能衰减 |
| **T8** 多组分回归 | CH4 + H2O → 双组分 ppm | 综合 MAE |
| **T9** 温度回归 | 光谱 → 气体温度（K） | MAE |

<br>

## 排行榜

**T1 浓度回归 + T3 跨仪器泛化**（v0 数据划分，CH4 DA）：

| 模型 | T1 MAE ↓ | T1 MAPE ↓ | T3 MAE ↓ | T3 性能衰减 |
|:------|:--------:|:---------:|:--------:|:--------------:|
| **SpektralNet** | **2.27** | 22.5% | **3.51** | 1.54x |
| 岭回归（Ridge Regression） | 2.84 | 29.9% | 3.72 | **1.31x** |
| 随机森林（Random Forest） | 5.27 | 24.1% | 10.89 | 2.07x |
| 物理信息神经网络（PINN） | 7.29 | 49.1% | 15.62 | 2.14x |
| 分块 Transformer（Patchified Transformer） | 7.39 | **22.7%** | 10.81 | 1.46x |
| 多层感知机（MLP/BPNN） | 8.08 | 44.5% | 9.85 | 1.22x |
| 一维 CNN（1D CNN） | 15.58 | 42.2% | 28.30 | 1.82x |
| 双向 LSTM（BiLSTM） | 29.47 | 61.7% | 51.04 | 1.73x |
| CNN-LSTM-Attention | 38.39 | 69.4% | 71.03 | 1.85x |

> 核心发现：**线性模型在该基准上占据主导地位**，因为 Beer-Lambert 吸光度与浓度呈线性关系。SpektralNet 通过物理特征增强 Ridge 而非增加模型深度来实现最优性能。

<details>
<summary><b>其他任务结果</b></summary>

| 任务 | 最佳模型 | 得分 |
|:-----|:-----------|:------|
| **T2** 去噪 | 一维 U-Net（1D U-Net） | RMSE 3.62e-3 |
| **T4** WMS | Ridge | MAE 15.15 ppm |
| **T5** 漂移 | 移动平均（Moving Average） | MAE 0.270 ppm |
| **T6** OOD | PCA + 马氏距离（Mahalanobis） | AUROC 0.672 |
| **T7** 跨模态 | Ridge（TDLAS→NDIR） | MAE 130.68 ppm（46x） |
| **T8** 多组分 | Ridge（双通道） | CH4 0.89 / H2O 3937 ppm |
| **T9** 温度 | Ridge | MAE 9.4 K |
| **T1-CRDS** 浓度（CRDS） | Ridge（tau） | MAE 36.5 ppm |
| **T1-FTIR** 浓度（FTIR） | Ridge（spectrum） | MAE 83.7 ppm |
| **T1-DOAS** 浓度（DOAS） | Ridge（diff OD） | MAE 1.40 ppm |

</details>

[**完整排行榜 →**](https://spektran.github.io/spektran/leaderboard/) &nbsp;&middot;&nbsp;
[**提交结果 →**](https://spektran.github.io/spektran/leaderboard/#submitting-results)

<br>

## 真实场景应用

SPEKTRAN 的基准任务对应着工业和环境科学中的真实问题：

- **甲烷泄漏检测** — 油气设施、垃圾填埋场、畜牧业（T1、T3）
- **CO2 监测** — 温室气体定量、室内空气质量、过程控制（[CO2 数据集](https://huggingface.co/datasets/spektran/spektran-co2-v0)）
- **工业排放监测** — SO2、NO、CO 烟气连续分析（[工业数据集](https://huggingface.co/datasets/spektran/spektran-industrial-v0)）
- **多气体混合物** — 燃烧排放中分离重叠光谱组分（[多气体数据集](https://huggingface.co/datasets/spektran/spektran-multigas-v0)）
- **医学呼气分析** — ppb 级痕量气体生物标志物检测（T1、T9）
- **仪器无关部署** — 模型跨硬件迁移，无需重新标定（T3、T7）
- **抗漂移野外传感器** — 恶劣环境下的长期自主监测（T5）

<br>

## 工作原理

```
 HITRAN line data        Instrument configs         Benchmark
 ───────────────        ──────────────────         ─────────
 Line positions    ──►  Virtual instruments   ──►  Official splits
 Line strengths         (noise, fringes,           (train/val/test/
 Broadening params       drift, chirp)              held-out)
        │                      │                       │
        ▼                      ▼                       ▼
   Forward physics  ──►  Noisy spectra   ──►   Evaluate & rank
   (Voigt / HTP)         with provenance       on leaderboard
```

<br>

## 值得信赖的物理基础

- **双实现交叉验证** — 独立的参考实现与 HITRAN/hapi 交叉校验
- **文献锚定噪声模型** — 仪器噪声参数调研自 18 篇已发表文献中的实测系统
- **仿真-真实差距报告** — 已知差距来源记录于 [G5 报告](gates/reports/)
- **自动化质量门禁** — G1–G5 检查点在 CI 中强制执行

<br>

## 引用

```bibtex
@software{spektran,
  title     = {SPEKTRAN: Simulation Engine and ML Benchmark for Optical Gas Sensing},
  url       = {https://github.com/spektran/spektran},
  doi       = {10.5281/zenodo.21790394},
  version   = {0.6.0},
  license   = {Apache-2.0}
}
```

详见 [CITATION.cff](CITATION.cff)。Zenodo DOI：[10.5281/zenodo.21790394](https://doi.org/10.5281/zenodo.21790394)。

## 贡献指南

详见 [CONTRIBUTING.md](CONTRIBUTING.md)。我们欢迎新的基线模型、检测模态与谱线数据库方面的贡献。

<div align="center">
<sub>代码许可：Apache-2.0 &nbsp;·&nbsp; 数据与模式许可：CC BY 4.0</sub>
</div>
