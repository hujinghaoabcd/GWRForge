# GWRForge

[English](README.md)

**GWRForge** 是一个面向高级地理加权回归模型的可复现实验与基准测试项目，模型实现来自 [pyGWRx](https://github.com/hujinghaoabcd/pyGWRx)。当前主要研究两个自主开发模型：

- **LG-GWR（Latent-Geometry GWR，潜在几何地理加权回归）**：融合地理坐标与上下文属性学习一套共享潜在几何，并在学习空间中构造局部权重和完成局部回归。
- **GR-GWR（Geographic-Regime GWR，地理区域地理加权回归）**：根据 GWR 局部系数模式发现空间连通区域，并通过边界惩罚和 ICM 优化完成区域感知的局部回归。

本仓库用于方法验证、真实数据比较、论文级实验和结果复现。模型源码**不会复制到本项目中重新实现**，所有实验代码均通过 `pygwrx` 公共 API 调用模型。

## 总体实验思路

本项目采用以下论文实验结构：

> **真实数据拟合实验作为主体 + 受控模拟验证结构恢复能力 + 少量预测实验作为补充。**

### LG-GWR 实验

| 实验部分 | 主要目的 |
|---|---|
| 数值退化检验 | 验证关闭属性作用后，Separable 形式能够退化为普通 GWR。 |
| L1–L6 模拟 | 分别检验欧氏几何、非各向同性、地理—属性联合几何、属性主导结构、噪声属性和参数特定几何负向控制。 |
| 真实数据拟合 | 比较 OLS、GWR、MGWR、SGWR、LG-GWR Joint 和 LG-GWR Separable，报告 AICc、调整 R²、RSS、RMSE、MAE、ENP 和残差空间自相关。 |
| 潜在几何诊断 | 分析学习距离相关性、邻域重叠、变量几何贡献和邻域变化。 |
| 消融与敏感性 | 检验潜在维度、属性删除或打乱、重启次数、带宽更新和标准化的作用。 |
| 补充预测 | 检查拟合改善能否在随机留出或空间留出样本中部分保持。 |

### GR-GWR 实验

| 实验部分 | 主要目的 |
|---|---|
| 数值退化检验 | 验证只有一个区域时，GR-GWR 能够退化为普通 GWR。 |
| R1–R6 模拟 | 检验无区域、规则及不规则连通区域、不平衡区域、平滑过渡和相同机制非连通分布。 |
| 真实数据拟合 | 比较 OLS、GWR、MGWR、区域基线和 GR-GWR，并报告拟合指标与区域诊断。 |
| 区域诊断 | 报告区域数量与规模、区域内系数方差、区域间差异、连通性、边界数量和稳定性。 |
| 消融与敏感性 | 检验 ICM、边界惩罚、连通性约束、坐标—系数平衡、初始化方式和最小区域规模。 |
| 补充预测 | 检查识别出的区域结构是否具有有限的样本外价值。 |

## 当前状态

| 内容 | 状态 |
|---|---|
| 实验包与命令行脚本 | 已实现 |
| 数值退化和正确性检验 | 已实现并通过 |
| L1–L6 与 R1–R6 数据生成 | 已实现 |
| 小规模 pilot 全流程 | 已运行 |
| 参数选择、消融、统计、绘图和报告 | 已实现 |
| 论文正式配置 | 已写入 `configs/full.json` |
| 全部论文级正式结果 | 尚未全部运行完成，不能用 pilot 数值替代正式结果 |

Pilot 结果只用于工程验证，证明实验程序、指标计算、结果保存和报告流程可以正常运行。论文中的正式结论必须来自完整重复实验，而不能直接引用 pilot 的均值或显著性结果。

## 安装

建议使用 Python 3.11 或更高版本。

```bash
python -m venv .venv

# Linux/macOS
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

该命令将从 PyPI 安装 `pyGWRx==0.1.2`，并以可编辑模式安装 GWRForge 实验包。

## 快速运行

建议先运行轻量级 pilot：

```bash
python scripts/run_pipeline.py \
  --config configs/pilot.json \
  --profile pilot
```

运行论文正式配置：

```bash
python scripts/run_pipeline.py \
  --config configs/full.json \
  --profile full
```

正式配置计算量较大，包括 20 × 20 模拟格网、100 个独立测试位置、每个场景 100 次 Monte Carlo 重复、重复空间验证以及 LG-GWR 多重初始化优化。

## 分阶段运行

```bash
python scripts/00_environment_check.py
python scripts/01_correctness.py
python scripts/02_lg_simulation.py --config configs/pilot.json
python scripts/03_gr_simulation.py --config configs/pilot.json
python scripts/04_real_data.py --config configs/pilot.json
python scripts/06_parameter_selection.py --config configs/pilot.json
python scripts/07_ablation.py --config configs/pilot.json
python scripts/08_benchmark.py --config configs/pilot.json
python scripts/09_statistics.py
python scripts/10_figures.py
python scripts/05_report.py --profile pilot
```

## 结果输出规则

所有运行结果写入 `results/`，该目录默认不提交到 Git。输出包括：

- 可直接统计分析的 CSV 表格；
- 环境信息和运行日志；
- 模拟与真实数据图件；
- 配对统计检验结果；
- 固定抽样索引和断点文件；
- 自动生成的 Markdown 报告。

失败、警告和未收敛记录也是实验结果的一部分，不允许在汇总时静默删除。

## 项目结构

```text
GWRForge/
├── configs/                    # Pilot 与论文正式实验配置
├── reports/                    # 方法方案、实现审计和 pilot 报告
├── scripts/                    # 各实验阶段的可执行脚本
├── src/pygwrx_experiments/     # 模拟、模型适配、指标、运行器和绘图
├── tests/                      # 快速工程测试与数值测试
├── .github/workflows/          # 持续集成测试
├── pyproject.toml
└── requirements.txt
```

## 可复现性原则

1. 所有重复实验必须固定并记录随机种子。
2. 使用留出样本时，只能使用训练或拟合数据完成标准化。
3. 可比较模型应使用相同或尽可能接近的调参预算。
4. Pilot 与正式实验结果必须分目录保存。
5. 必须报告失败次数、未收敛比例和运行异常，不能只保留成功结果。
6. 不能仅根据预测误差改善声称恢复了正确几何或正确区域。
7. 预测只作为补充证据，真实数据拟合和结构恢复才是主要实验。

## 测试

```bash
pytest
```

GitHub Actions 会在提交和拉取请求时自动运行相同的测试套件。

## 与 pyGWRx 的关系

GWRForge 是独立实验仓库；模型实现、内置数据和公共 API 由 `pyGWRx` 提供。模型内部算法问题应提交到 pyGWRx 仓库，实验设计、脚本、配置、统计和报告问题应提交到本仓库。

## 作者

**Jinghao Hu（胡京浩）**
