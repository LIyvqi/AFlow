# -*- coding: utf-8 -*-
# @Date    : 8/23/2024 20:00 PM
# @Author  : didi
# @Desc    : Entrance of AFlow.

import argparse
from typing import Dict, List

# download:
#   负责检查/下载公开数据集。入口启动时先确保 data/datasets 可用，
#   后面的 Evaluator/Benchmark 才能读取验证集或测试集。
from data.download_data import download

# Optimizer:
#   AFlow 的核心控制器。run.py 只负责准备参数和配置；
#   真正的“读取历史 workflow -> 让 LLM 生成新 workflow -> 评测 -> 记录结果”
#   都在 Optimizer 里完成。
from scripts.optimizer import Optimizer

# LLMsConfig:
#   负责从 config/config2.yaml 读取多个模型配置。
#   这里会取出 opt_model 和 exec_model 两套配置，分别用于优化和执行。
from scripts.async_llm import LLMsConfig


class ExperimentConfig:
    """单个实验/数据集的入口配置。

    AFlow 在入口处只关心三件事：
    1. dataset: 数据集名称，用来决定 workspace/<DATASET> 和 Benchmark。
    2. question_type: 题目类型，影响优化 prompt 的任务说明。
    3. operators: 允许优化器组合使用的算子列表。

    注意：这里不是模型配置，也不是数据本身，只是告诉 Optimizer
    “这类任务应该用什么题型理解、可以使用哪些 workflow 积木”。
    """

    def __init__(self, dataset: str, question_type: str, operators: List[str]):
        self.dataset = dataset
        self.question_type = question_type
        self.operators = operators


# AFlow 的入口配置表。
#
# 当命令行传入 --dataset MATH 时，程序会到这个字典里取出 MATH 对应的
# ExperimentConfig。换句话说，这个表就是“数据集名称 -> 运行配置”的映射。
#
# 如果你以后要接入自己的数据集，通常需要在这里新增一项，例如：
#
#   "MyDataset": ExperimentConfig(
#       dataset="MyDataset",
#       question_type="qa",
#       operators=["Custom", "AnswerGenerate", "ScEnsemble"],
#   )
#
# 同时还要在 benchmarks/ 和 scripts/evaluator.py 里注册对应评测逻辑。
EXPERIMENT_CONFIGS: Dict[str, ExperimentConfig] = {
    # QA 类任务：通常让模型生成答案，再用自一致/集成类算子挑选答案。
    "DROP": ExperimentConfig(
        dataset="DROP",
        question_type="qa",
        operators=["Custom", "AnswerGenerate", "ScEnsemble"],
    ),
    "HotpotQA": ExperimentConfig(
        dataset="HotpotQA",
        question_type="qa",
        operators=["Custom", "AnswerGenerate", "ScEnsemble"],
    ),

    # 数学类任务：除了普通生成，也允许 Programmer 生成/执行 Python 辅助计算。
    "MATH": ExperimentConfig(
        dataset="MATH",
        question_type="math",
        operators=["Custom", "ScEnsemble", "Programmer"],
    ),
    "GSM8K": ExperimentConfig(
        dataset="GSM8K",
        question_type="math",
        operators=["Custom", "ScEnsemble", "Programmer"],
    ),

    # 代码类任务：常见流程是生成代码、运行测试、根据失败反馈修正。
    "MBPP": ExperimentConfig(
        dataset="MBPP",
        question_type="code",
        operators=["Custom", "CustomCodeGenerate", "ScEnsemble", "Test"],
    ),
    "HumanEval": ExperimentConfig(
        dataset="HumanEval",
        question_type="code",
        operators=["Custom", "CustomCodeGenerate", "ScEnsemble", "Test"],
    ),
    "LiveCodeBench": ExperimentConfig(
        dataset="LiveCodeBench",
        question_type="code",
        operators=["Custom", "CustomCodeGenerate", "ScEnsemble", "Test"],
    ),
}


def parse_args():
    """解析命令行参数。

    run.py 的典型启动方式：

        python run.py --dataset MATH --max_rounds 1

    这个函数只负责把命令行字符串转换成 args 对象，不做实际业务逻辑。
    真正使用这些参数的位置在文件底部的 if __name__ == "__main__"。
    """

    parser = argparse.ArgumentParser(description="AFlow Optimizer")

    # 必填参数：指定要跑哪个数据集。
    #
    # choices=list(EXPERIMENT_CONFIGS.keys()) 的意思是：
    # 只能选择入口配置表里已经注册过的数据集，防止拼错名字后程序跑到一半才失败。
    #
    # 例子：
    #   python run.py --dataset MATH
    #
    # 这个参数会影响三件事：
    #   1. 使用 EXPERIMENT_CONFIGS 里的哪一份 ExperimentConfig。
    #   2. 读取哪个数据文件，例如 data/datasets/math_validate.jsonl。
    #   3. 结果保存到哪个目录，例如 workspace/MATH/。
    parser.add_argument(
        "--dataset",
        type=str,
        choices=list(EXPERIMENT_CONFIGS.keys()),
        required=True,
        help="Dataset type",
    )

    # sample 控制优化时从历史高分 workflow 中采样多少个候选。
    # 这个值越大，优化器可参考的历史范围越广，但也可能引入更多上下文和成本。
    #
    # AFlow 每一轮优化时，不是完全凭空让 LLM 写新 workflow，
    # 而是会从历史结果中挑一些表现较好的 round 作为参考。
    #
    # 简单理解：
    #   sample 小：更聚焦，prompt 更短，探索范围较窄。
    #   sample 大：参考更多历史方案，探索更广，但可能更贵、更慢。
    #
    # 学习阶段可以保持默认 4。
    parser.add_argument("--sample", type=int, default=4, help="Sample count")

    # 优化结果保存根目录。默认会写到：
    #   workspace/<DATASET>/workflows/round_x/
    #
    # 里面通常包含 graph.py、prompt.py、评测日志、经验文件等。
    #
    # 例如：
    #   --dataset MATH --optimized_path workspace
    #
    # 会写入：
    #   workspace/MATH/workflows/round_1/
    #   workspace/MATH/workflows/round_2/
    #
    # 如果你想把某次实验单独放起来，可以改成：
    #   --optimized_path workspace_debug
    parser.add_argument(
        "--optimized_path",
        type=str,
        default="workspace",
        help="Optimized result save path",
    )

    # initial_round 表示从第几轮开始。
    # 默认从 1 开始：第 1 轮通常先评测初始 workflow，作为后续优化的基线。
    #
    # 常见情况：
    #   初次运行：保持 1。
    #   接着已有结果继续跑：可以根据 workspace 里已有 round 调整。
    #
    # 注意：如果 workspace 中没有对应 round 的 graph.py / prompt.py，
    # 后面的 GraphUtils.load_graph 会找不到文件。
    parser.add_argument("--initial_round", type=int, default=1, help="Initial round")

    # max_rounds 是最多优化多少轮。
    # 学习/调试时建议先设成 1，避免一次运行调用太多 LLM。
    #
    # 每一轮大致会做：
    #   1. 读取历史 workflow 和经验。
    #   2. 调用 opt_model 生成新 workflow。
    #   3. 调用 exec_model 在验证集上评测。
    #   4. 写入分数、日志和经验。
    #
    # 所以 max_rounds 越大，搜索越充分，但时间和费用也越高。
    #
    # 推荐学习命令：
    #   python run.py --dataset MATH --max_rounds 1
    parser.add_argument("--max_rounds", type=int, default=20, help="Max iteration rounds")

    # 是否开启收敛早停。
    # 注意：这里 type=bool 在 argparse 里并不适合解析 "False" 这类字符串；
    # 当前代码保持原样，但真实使用时可以优先依赖默认值，或后续改成更稳的解析方式。
    #
    # 它的作用是：
    #   如果最近若干轮 top score 已经稳定，Optimizer 可以提前停止。
    #
    # 小坑：
    #   python 里 bool("False") 其实是 True，因为非空字符串都是真。
    #   所以当前写法下，不建议用 --check_convergence False 这种方式关闭。
    #
    # 后续如果要修，可以改成：
    #   type=lambda x: x.lower() == "true"
    # 或改成 argparse 的 store_true / store_false。
    parser.add_argument("--check_convergence", type=bool, default=True, help="Whether to enable early stop")

    # 每个 workflow 在验证集上重复评测几轮。
    # 多轮可以降低随机性，但会增加 LLM 调用成本和运行时间。
    #
    # 如果模型输出随机性很强，可以适当加大这个值，用多次平均分更稳。
    # 如果只是看代码链路是否跑通，保持 1 最省时间。
    parser.add_argument("--validation_rounds", type=int, default=1, help="Validation rounds")

    # 是否强制重新下载数据集。
    # 这里用 lambda 把 "true"/"false" 字符串转换成布尔值。
    #
    # 默认 False：如果本地已经有数据，就不重复下载。
    #
    # 例子：
    #   python run.py --dataset MATH --if_force_download true
    #
    # 如果你接入的是自己的内部数据集，后面很可能会跳过 download，
    # 或者把它换成自己的数据准备函数。
    parser.add_argument(
        "--if_force_download",
        type=lambda x: x.lower() == "true",
        default=False,
        help="Whether enforce dataset download.",
    )

    # opt_model_name:
    #   优化模型名称。这个模型负责阅读历史结果、失败日志和现有 workflow，
    #   然后生成下一轮 graph.py / prompt.py。
    #
    # 这个值必须出现在 config/config2.yaml 的 models 下面。
    #
    # 它更像“流程设计师”：
    #   - 看上一轮 workflow 哪里表现不好。
    #   - 参考 log.json 和 processed_experience.json。
    #   - 生成新的 workflow 代码。
    #
    # 通常建议用能力更强的模型，因为它要写代码、改 prompt、总结失败经验。
    parser.add_argument(
        "--opt_model_name",
        type=str,
        default="claude-3-5-sonnet-20241022",
        help="Specifies the name of the model used for optimization tasks.",
    )

    # exec_model_name:
    #   执行模型名称。这个模型负责在 workflow 运行时处理具体题目样本。
    #
    # 简单理解：
    #   opt_model 像“架构师”，决定 workflow 怎么改；
    #   exec_model 像“执行者”，按照 workflow 去解题。
    #
    # 它会被传入 Workflow / Operator，后续每条样本的实际模型调用
    # 大多都会走这套配置。
    #
    # 通常可以用成本更低、速度更快、输出稳定的模型。
    parser.add_argument(
        "--exec_model_name",
        type=str,
        default="gpt-4o-mini",
        help="Specifies the name of the model used for execution tasks.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    # Python 入口保护。
    #
    # 只有直接运行 python run.py 时，下面的代码才会执行。
    # 如果其他文件 import run.py，只会加载上面的类、函数和常量，不会直接启动实验。
    args = parse_args()

    # 1. 根据 --dataset 找到对应的问题类型和可用算子。
    #
    # 例如：
    #   python run.py --dataset MATH
    #
    # 这里会得到：
    #   dataset="MATH"
    #   question_type="math"
    #   operators=["Custom", "ScEnsemble", "Programmer"]
    config = EXPERIMENT_CONFIGS[args.dataset]

    # 2. 从 config/config2.yaml 读取模型配置。
    #
    # LLMsConfig.default() 会读取 YAML 里的 models 字段，得到一个模型配置管理器。
    # 后面再用 args.opt_model_name / args.exec_model_name 取出具体模型。
    models_config = LLMsConfig.default()

    # opt_llm_config:
    #   给 Optimizer 里的 optimize_llm 使用。
    #   它负责调用大模型生成新的 workflow 代码和 prompt。
    opt_llm_config = models_config.get(args.opt_model_name)
    if opt_llm_config is None:
        raise ValueError(
            f"The optimization model '{args.opt_model_name}' was not found in the 'models' section of the configuration file. "
            "Please add it to the configuration file or specify a valid model using the --opt_model_name flag. "
        )

    # exec_llm_config:
    #   会继续传入 workflow/operator。
    #   当 workflow 处理某条题目时，具体的 LLM 调用会使用这套配置。
    exec_llm_config = models_config.get(args.exec_model_name)
    if exec_llm_config is None:
        raise ValueError(
            f"The execution model '{args.exec_model_name}' was not found in the 'models' section of the configuration file. "
            "Please add it to the configuration file or specify a valid model using the --exec_model_name flag. "
        )

    # 3. 确保公开数据集存在。
    #
    # download(["datasets"]) 会检查/下载 data/datasets。
    # 如果你接入的是内部数据集，通常可以把这里替换成自己的数据准备逻辑，
    # 或者在数据已经准备好的情况下跳过下载。
    download(["datasets"], force_download=args.if_force_download)  # remove download initial_rounds in new version.

    # 4. 创建 Optimizer。
    #
    # 这是 run.py 最关键的一步：把入口层得到的所有信息交给优化器。
    #
    # 参数大致可以分成四类：
    #   任务信息：dataset、question_type、operators
    #   模型信息：opt_llm_config、exec_llm_config
    #   保存位置：optimized_path
    #   搜索控制：sample、initial_round、max_rounds、validation_rounds、check_convergence
    optimizer = Optimizer(
        dataset=config.dataset,
        question_type=config.question_type,
        opt_llm_config=opt_llm_config,
        exec_llm_config=exec_llm_config,
        check_convergence=args.check_convergence,
        operators=config.operators,
        optimized_path=args.optimized_path,
        sample=args.sample,
        initial_round=args.initial_round,
        max_rounds=args.max_rounds,
        validation_rounds=args.validation_rounds,
    )

    # 5. 启动优化。
    #
    # Graph 模式会执行自动优化闭环：
    #
    #   读取历史 workflow
    #   -> 采样高分 workflow
    #   -> 构造优化 prompt
    #   -> 调用 opt_model 生成新的 graph.py / prompt.py
    #   -> 调用 Evaluator 在验证集上评测
    #   -> 保存分数、日志和经验
    #   -> 进入下一轮
    #
    # 具体实现从 scripts/optimizer.py 的 Optimizer.optimize() 开始。
    optimizer.optimize("Graph")

    # Test 模式：用于测试指定 workflow，不进行新一轮优化。
    #
    # 如果你只是想评测已有 workflow，可以注释掉上面的 Graph 模式，
    # 改用下面这一行。不过 test() 里默认测试哪些 round，还需要去 optimizer.py 调整。
    # optimizer.optimize("Test")
