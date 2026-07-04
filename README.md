# AFlow：自动化智能体工作流生成与优化

[![Arxiv](https://img.shields.io/badge/arXiv-AFlow-b31b1b)](https://arxiv.org/abs/2410.10762)
[![PR Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen)](https://github.com/FoundationAgents/AFlow/pulls)

> 如果你在使用或复现代码时遇到困难，可以联系原作者：Email: didi4goooogle@gmail.com，Wechat: 18831933368。部分算子在从 MetaGPT 迁移到当前仓库的过程中可能仍存在问题。

## 中文学习指南

如果你想从 0 入门这个仓库，或者计划把 AFlow 二次开发到自己的数据、内部模型、约束输入格式和输出格式上，请优先阅读：

[AFlow 学习与二次开发指南](./learning.md)

这份文档重点说明了项目架构、设计思想、核心运行链路、重点代码阅读顺序，以及如何接入自己的数据集和模型。

## 项目简介

AFlow 是一个用于自动生成和优化智能体工作流的框架。它使用蒙特卡洛树搜索思想，在“代码表示的工作流空间”中搜索更有效的工作流，从而用机器自动探索替代大量手工工作流设计。

简单来说，AFlow 优化的不是模型参数，而是智能体的工作流代码：

```text
初始工作流
  -> 优化模型分析历史表现和失败案例
  -> 生成新的工作流代码
  -> 在验证集上评测
  -> 根据分数、日志和经验继续下一轮优化
```

实验显示，AFlow 在多个任务上有潜力超过人工设计的工作流。项目也在持续扩展更多基准评测和开放任务支持。

<p align="center">
<a href=""><img src="assets/AFLOW-performance.jpg" alt="AFlow 性能表现" title="AFlow 性能表现" width="80%"></a>
</p>

## 框架组成

AFlow 的核心由以下几部分组成：

- **节点**：大语言模型调用的基本单元。原项目中可参考 `metagpt_core/action_nodes/action_node.py`，它提供了控制模型、温度、格式和提示词的接口。
- **算子**：由多个节点或能力封装出的预定义操作，用于提升搜索效率。常见操作包括生成、格式化、评审、修订、集成、测试、编程器等。当前仓库可重点查看 `scripts/operators.py`。
- **工作流**：由多个大语言模型调用节点及其连接关系组成，可以用图、神经网络结构或代码来表示。当前实现中，工作流主要以 Python 代码形式保存在 `workspace/<DATASET>/workflows/round_x/graph.py`。
- **优化器**：使用大语言模型和类似蒙特卡洛树搜索的流程，不断选择、扩展、评估和更新工作流。核心代码见 `scripts/optimizer.py`。
- **评测器**：负责在指定任务上评估工作流表现，并将分数、成本和日志反馈给优化器。核心代码见 `scripts/evaluator.py` 和 `benchmarks/`。

<p align="center">
<a href=""><img src="assets/AFLOW-method.jpg" alt="AFlow 框架方法" title="AFlow 框架方法" width="80%"></a>
</p>

## 项目结构

```text
Aflow/
├── run.py                         # 主入口：启动优化实验
├── run_baseline.py                # 运行 baseline
├── requirements.txt               # Python 依赖
├── config/
│   └── config2.example.yaml       # 模型配置示例
├── scripts/
│   ├── optimizer.py               # 优化主循环
│   ├── async_llm.py               # 模型调用封装
│   ├── operators.py               # 智能体操作积木
│   ├── evaluator.py               # 评测调度
│   ├── workflow.py                # 工作流抽象
│   ├── formatter.py               # LLM 输出格式解析与校验
│   ├── prompts/                   # 优化提示词和任务提示词
│   └── optimizer_utils/           # 图、数据、经验、收敛判断等工具
├── benchmarks/
│   ├── benchmark.py               # 基准评测基类
│   ├── math.py                    # MATH 评测
│   ├── gsm8k.py                   # GSM8K 评测
│   ├── humaneval.py               # HumanEval 评测
│   └── ...                        # 其他数据集评测
├── workspace/
│   └── <DATASET>/workflows/       # 自动生成和保存的工作流
├── data/
│   └── datasets/                  # 验证集和测试集
└── learning.md                    # 中文学习与二次开发指南
```

## 数据集

### 实验数据集

原论文实验使用了六个数据集：

```text
HumanEval
MBPP
GSM8K
MATH
HotpotQA
DROP
```

仓库中提供了这些数据集的评测代码。数据可以通过以下链接获取：

[数据集下载链接](https://drive.google.com/uc?export=download&id=1DNoegtZiUhWtvkd2xoIuElmIi4ah7k8e)

也可以使用 `data/download_data.py` 下载。

<p align="center">
<a href=""><img src="assets/AFLOW-experiment.jpg" alt="AFlow 实验结果" title="AFlow 实验结果" width="80%"></a>
</p>

### 自定义数据集

如果你要接入自己的任务，可以参考 `benchmarks/` 目录。

通常需要：

1. 继承 `BaseBenchmark`
2. 实现 `evaluate_problem`
3. 实现 `calculate_score`
4. 实现 `get_result_columns`
5. 在 `scripts/evaluator.py` 中注册新的数据集
6. 在 `run.py` 中添加新的 `ExperimentConfig`

更详细的二次开发步骤请看：[AFlow 学习与二次开发指南](./learning.md)。

## 快速开始

### 1. 配置 Python 环境

建议使用 Python 3.9。

```bash
conda create -n aflow python=3.9
conda activate aflow
pip install -r requirements.txt
```

如果你使用的是 micromamba，也可以：

```bash
micromamba create -n aflow python=3.9 pip
micromamba activate aflow
pip install -r requirements.txt
```

### 2. 配置模型参数

复制配置示例：

```bash
cp config/config2.example.yaml config/config2.yaml
```

然后在 `config/config2.yaml` 中配置模型名称、`base_url`、`api_key` 和温度等参数。

配置格式示例：

```yaml
models:
  internal-opt-model:
    base_url: "http://your-model-endpoint/v1"
    api_key: "your-api-key"
    temperature: 0.7

  internal-exec-model:
    base_url: "http://your-model-endpoint/v1"
    api_key: "your-api-key"
    temperature: 0
```

其中：

```text
opt_model：用于优化工作流，通常选择更强的模型
exec_model：用于执行工作流，处理具体样本
```

### 3. 配置运行参数

可以通过命令行参数配置优化过程，也可以修改 `run.py` 中的默认值。

常用参数：

```text
--dataset              必填，数据集类型，例如 HumanEval/MBPP/GSM8K/MATH/HotpotQA/DROP
--sample               采样数量，即每轮从历史工作流中重采样的数量
--optimized_path       优化结果保存路径，默认 workspace
--initial_round        初始轮数
--max_rounds           最大优化轮数
--check_convergence    是否启用收敛早停
--validation_rounds    验证轮数
--if_force_download    是否强制重新下载数据集
--opt_model_name       优化模型名称
--exec_model_name      执行模型名称
```

### 4. 运行优化

使用默认参数运行：

```bash
python run.py --dataset MATH
```

指定模型和轮数运行：

```bash
python run.py --dataset MATH \
  --opt_model_name internal-opt-model \
  --exec_model_name internal-exec-model \
  --max_rounds 1
```

第一次调试建议先设置 `--max_rounds 1`，确认模型、数据和评测链路可以跑通。

## 复现实验结果

原论文实验的原始数据可以通过以下链接获取：

[实验原始数据](https://drive.google.com/uc?export=download&id=1Sr5wjgKf3bN8OC7G6cO3ynzJqD4w6_Dv)

其中包含每轮生成的工作流、提示词、验证集轨迹、各数据集最优工作流以及对应测试集结果。

你也可以使用 `data/download_data.py` 下载相关数据，然后通过 `run.py` 中不同的 `ExperimentConfig` 复现实验。

## 二次开发建议

如果你希望把 AFlow 用到自己的数据和内部模型上，建议按下面顺序做：

```text
1. 先跑通现有 MATH 或 GSM8K 示例
2. 配置兼容 OpenAI 接口的内部模型
3. 准备少量自己的验证/测试 JSONL 数据
4. 新增自己的基准评测类
5. 在 evaluator.py 和 run.py 中注册新数据集
6. 复制一个现有工作区作为初始工作流
7. 先跑 max_rounds=1
8. 确认结果 CSV、log.json、results.json 正常
9. 再新增内部业务算子
10. 最后逐步扩大数据量和优化轮数
```

完整学习路线见：[AFlow 学习与二次开发指南](./learning.md)。

## 路线图

- 支持更多搜索算法
- 支持工作流中的多模型搜索
- 支持排行榜
- 支持更多基准评测
- 支持多模态任务

## 引用

如果你在研究中使用 AFlow，请引用原论文：

```bibtex
@inproceedings{
   zhang2025aflow,
   title={{AF}low: Automating Agentic Workflow Generation},
   author={Jiayi Zhang and Jinyu Xiang and Zhaoyang Yu and Fengwei Teng and Xiong-Hui Chen and Jiaqi Chen and Mingchen Zhuge and Xin Cheng and Sirui Hong and Jinlin Wang and Bingnan Zheng and Bang Liu and Yuyu Luo and Chenglin Wu},
   booktitle={The Thirteenth International Conference on Learning Representations},
   year={2025},
   url={https://openreview.net/forum?id=z5uVAKwmjf}
}
```
