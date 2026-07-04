# -*- coding: utf-8 -*-
# @Date    : 2025-03-31
# @Author  : Zhaoyang
# @Desc    : 

# 这个文件是 AFlow 的“模型调用适配层”。
#
# 在项目链路里，它的位置大致是：
#
#   run.py
#     -> LLMsConfig.default() 读取 config/config2.yaml
#     -> models_config.get(...) 取出 opt_model / exec_model 配置
#     -> Optimizer / Workflow / Operator 创建 AsyncLLM
#     -> AsyncLLM 通过 OpenAI-compatible API 调用模型
#
# 它本身不关心具体任务是数学、代码、问答还是审核打表；
# 它只负责把“模型配置 + prompt”变成一次真实的异步 LLM 调用。

# AsyncOpenAI 是 OpenAI Python SDK 里的异步客户端。
# 只要你的模型服务兼容 OpenAI Chat Completions API，
# 通常只需要配置 base_url 和 api_key，就可以复用这里的调用逻辑。
from openai import AsyncOpenAI

# BaseFormatter / FormatError 用于“结构化输出校验”。
# 比如 Optimizer 要求模型返回 modification / graph / prompt 三个字段，
# 就会通过 formatter 给 prompt 加格式要求，并校验模型响应是否符合格式。
from scripts.formatter import BaseFormatter, FormatError

import yaml
from pathlib import Path
from typing import Dict, Optional, Any


class LLMConfig:
    """单个模型的运行配置。

    这个类是一个轻量配置容器，不负责调用模型。

    它接收的 config 通常来自两种地方：
    1. config/config2.yaml 中某个模型条目。
    2. 外部代码手动传入的 dict。

    最终 AsyncLLM 会拿着这里的字段去创建 AsyncOpenAI 客户端。
    """

    def __init__(self, config: dict):
        # 单个模型的运行配置。内部模型若兼容 OpenAI 接口，主要改 base_url 和 api_key。
        #
        # model:
        #   真实传给 chat.completions.create(model=...) 的模型名。
        #   在 LLMsConfig.get() 中，默认会使用 YAML 里的 key 作为模型名。
        self.model = config.get("model", "gpt-4o-mini")

        # temperature:
        #   控制输出随机性。越高越发散，越低越稳定。
        #   对审核打表、结构化抽取这类任务，通常更偏向低温度。
        self.temperature = config.get("temperature", 1)

        # key:
        #   API key。注意 YAML 里通常写 api_key，
        #   LLMsConfig.get() 会把 api_key 映射成这里的 key。
        self.key = config.get("key", None)

        # base_url:
        #   OpenAI-compatible 服务地址。
        #   如果使用内部模型网关，通常改这里。
        self.base_url = config.get("base_url", "https://oneapi.deepwisdom.ai/v1")

        # top_p:
        #   nucleus sampling 参数，和 temperature 一起影响输出随机性。
        #   默认 1 表示不额外截断候选 token 分布。
        self.top_p = config.get("top_p", 1)


class LLMsConfig:
    """多个模型配置的管理器。

    config/config2.yaml 里通常会配置多个模型，例如：

        models:
          strong-opt-model:
            base_url: "http://your-endpoint/v1"
            api_key: "..."
            temperature: 0.7

          cheap-exec-model:
            base_url: "http://your-endpoint/v1"
            api_key: "..."
            temperature: 0

    LLMsConfig 会把这些配置读进 self.configs。
    然后外部通过 get("模型名") 获取单个 LLMConfig。
    """
    
    # _instance 目前没有实际使用，保留在这里可能是早期想做 singleton。
    _instance = None  # For singleton pattern if needed

    # _default_config 用来缓存从 YAML 读出来的配置。
    # 这样同一次进程里多次调用 LLMsConfig.default() 时，不会重复读文件。
    _default_config = None
    
    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        """用一个字典初始化模型配置管理器。

        config_dict 的形状一般是：

            {
                "gpt-4o-mini": {
                    "base_url": "...",
                    "api_key": "...",
                    "temperature": 0
                },
                ...
            }
        """
        self.configs = config_dict or {}
    
    @classmethod
    def default(cls):
        """读取并缓存默认配置文件。

        run.py 中的：

            models_config = LLMsConfig.default()

        调的就是这里。

        返回值不是单个模型配置，而是 LLMsConfig 管理器。
        后续还要通过 .get(model_name) 取出具体模型。
        """
        if cls._default_config is None:
            # 默认从 config/config2.yaml 读取模型列表。
            #
            # 这里列了三个路径，是为了兼容从不同工作目录启动脚本的情况。
            # 当前仓库正常从项目根目录运行时，一般命中第一个：
            #   config/config2.yaml
            config_paths = [
                Path("config/config2.yaml"),
                Path("config2.yaml"),
                Path("./config/config2.yaml")
            ]
            
            # 找到第一个存在的配置文件。
            config_file = None
            for path in config_paths:
                if path.exists():
                    config_file = path
                    break
            
            # 如果三个位置都没有配置文件，说明还没有从 example 复制出 config2.yaml。
            if config_file is None:
                raise FileNotFoundError("No default configuration file found in the expected locations")
            
            # YAML 顶层通常是 models，每个 key 是模型名称。
            with open(config_file, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
            
            # Your YAML has a 'models' top-level key that contains the model configs
            #
            # 如果 YAML 是：
            #
            #   models:
            #     gpt-4o-mini:
            #       api_key: ...
            #
            # 这里会把 config_data 从整个 YAML 缩小成 models 下面那层字典。
            if 'models' in config_data:
                config_data = config_data['models']
                
            # 缓存起来，下次 default() 直接返回，不再读文件。
            cls._default_config = cls(config_data)
        
        return cls._default_config
    
    def get(self, llm_name: str) -> LLMConfig:
        """根据模型名获取单个 LLMConfig。

        llm_name 必须是 config/config2.yaml 的 models 下面的 key。

        例如 YAML 中有：

            models:
              internal-opt-model:
                api_key: ...

        那么这里应该调用：

            models_config.get("internal-opt-model")
        """
        if llm_name not in self.configs:
            raise ValueError(f"Configuration for {llm_name} not found")
        
        config = self.configs[llm_name]
        
        # 把 YAML 中的 api_key 映射成 LLMConfig 使用的 key。
        #
        # 为什么要映射？
        #   YAML 里更常见的字段名是 api_key；
        #   LLMConfig 里为了传给 AsyncOpenAI，字段叫 key。
        #
        # 这里还把 llm_name 作为 model 字段。
        # 所以 YAML 的 key 不只是配置名称，也会成为实际请求里的 model 名。
        llm_config = {
            "model": llm_name,  # Use the key as the model name
            "temperature": config.get("temperature", 1),
            "key": config.get("api_key"),  # Map api_key to key
            "base_url": config.get("base_url", "https://oneapi.deepwisdom.ai/v1"),
            "top_p": config.get("top_p", 1)  # Add top_p parameter
        }
        
        # Create and return an LLMConfig instance with the specified configuration
        return LLMConfig(llm_config)
    
    def add_config(self, name: str, config: Dict[str, Any]) -> None:
        """运行时新增或覆盖一个模型配置。

        当前主流程主要从 YAML 读取配置，这个方法更适合测试或动态注入配置。
        """
        self.configs[name] = config
    
    def get_all_names(self) -> list:
        """返回当前可用的所有模型名称。"""
        return list(self.configs.keys())

    
class ModelPricing:
    """模型价格表，用来估算调用成本。

    这里的价格单位是：

        USD / 1K tokens

    注意：
    1. 这个表只影响成本估算，不影响模型调用。
    2. 如果你的内部模型没有配置价格，成本会按 0 计算。
    3. 价格可能会随时间变化，严肃计费场景需要以实际服务商账单为准。
    """

    PRICES = {
        # GPT-4o models
        "gpt-4o": {"input": 0.0025, "output": 0.01},
        "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
        "gpt-4o-mini-2024-07-18": {"input": 0.00015, "output": 0.0006},
        "claude-3-5-sonnet": {"input": 0.003, "output": 0.015},
        "o3":{"input":0.003, "output":0.015},
        "o3-mini": {"input": 0.0011, "output": 0.0025},
    }
    
    @classmethod
    def get_price(cls, model_name, token_type):
        """获取某个模型的输入/输出 token 单价。

        token_type 只能是：
            "input"
            "output"
        """

        # 优先精确匹配。
        # 例如 model_name == "gpt-4o-mini"。
        if model_name in cls.PRICES:
            return cls.PRICES[model_name][token_type]
        
        # 如果精确匹配不到，再做包含匹配。
        # 例如：
        #   model_name = "claude-3-5-sonnet-20241022"
        # 可以匹配到：
        #   "claude-3-5-sonnet"
        for key in cls.PRICES:
            if key in model_name:
                return cls.PRICES[key][token_type]
        
        # 匹配不到价格时返回 0，表示只统计 tokens，不估算费用。
        return 0


class TokenUsageTracker:
    """跟踪一个 AsyncLLM 实例的 token 使用量和估算成本。

    每个 AsyncLLM 都有自己的 TokenUsageTracker。
    这意味着 opt_model 和 exec_model 的 token 使用会分别统计在各自实例里。
    """

    def __init__(self):
        # 每个 AsyncLLM 实例维护自己的 token 和成本统计。
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0

        # usage_history 保存每次调用的明细，方便后续排查某次调用的成本。
        self.usage_history = []
    
    def add_usage(self, model, input_tokens, output_tokens):
        """记录一次模型调用的 token 和成本。

        response.usage 里通常会给出：
            prompt_tokens      -> 输入 token
            completion_tokens  -> 输出 token

        这里把它们转换成一个 usage_record，并累加到总计里。
        """

        # ModelPricing.get_price 返回的是每 1000 tokens 的价格，
        # 所以这里要 input_tokens / 1000。
        input_cost = (input_tokens / 1000) * ModelPricing.get_price(model, "input")
        output_cost = (output_tokens / 1000) * ModelPricing.get_price(model, "output")
        total_cost = input_cost + output_cost
        
        # 单次调用的明细记录。
        usage_record = {
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "input_cost": input_cost,
            "output_cost": output_cost,
            "total_cost": total_cost,
            "prices": {
                "input_price": ModelPricing.get_price(model, "input"),
                "output_price": ModelPricing.get_price(model, "output")
            }
        }
        
        # 累计总量。
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_cost += total_cost

        # 保存历史明细。
        self.usage_history.append(usage_record)
        
        return usage_record
    
    def get_summary(self):
        """返回当前 AsyncLLM 实例的累计 token / cost 统计。"""
        return {
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            "total_cost": self.total_cost,
            "call_count": len(self.usage_history),
            "history": self.usage_history
        }


class AsyncLLM:
    """异步 LLM 调用封装。

    这是本文件最核心的类。

    外部拿到 AsyncLLM 后，可以像调用函数一样调用它：

        llm = AsyncLLM(config)
        answer = await llm(prompt)

    之所以可以这样写，是因为类里实现了 __call__。
    """

    def __init__(self, config, system_msg:str = None):
        """
        Initialize the AsyncLLM with a configuration
        
        Args:
            config: Either an LLMConfig instance or a string representing the LLM name
                   If a string is provided, it will be looked up in the default configuration
            system_msg: Optional system message to include in all prompts
        """
        # 支持直接传模型名；这时会自动从 config/config2.yaml 找配置。
        #
        # 例如：
        #   AsyncLLM("gpt-4o-mini")
        #
        # 会自动等价于：
        #   LLMsConfig.default().get("gpt-4o-mini")
        if isinstance(config, str):
            llm_name = config
            config = LLMsConfig.default().get(llm_name)
        
        # 这里使用 OpenAI-compatible 客户端；内部模型只要兼容该协议，通常无需改代码。
        self.config = config

        # AsyncOpenAI 是异步客户端。
        # 后续调用 self.aclient.chat.completions.create(...) 时需要 await。
        self.aclient = AsyncOpenAI(api_key=self.config.key, base_url=self.config.base_url)

        # 可选 system message。
        # 如果传入，它会在每次调用时作为 messages 的第一条 system 消息。
        self.sys_msg = system_msg

        # 用来统计这个 AsyncLLM 实例的 token 和成本。
        self.usage_tracker = TokenUsageTracker()
        
    async def __call__(self, prompt):
        # 统一的模型调用入口。Operator 和 Optimizer 最终都会走到这里。
        #
        # message 是 OpenAI Chat Completions API 的标准 messages 格式：
        #   [
        #       {"role": "system", "content": "..."},
        #       {"role": "user", "content": "..."}
        #   ]
        #
        # 当前封装只支持单轮 user prompt，不保存多轮对话历史。
        message = []
        if self.sys_msg is not None:
            message.append({
                "content": self.sys_msg,
                "role": "system"
            })

        # 用户输入 prompt 作为 user 消息。
        message.append({"role": "user", "content": prompt})

        # 发起异步模型调用。
        #
        # 这里使用 chat.completions.create，说明后端需要兼容 OpenAI Chat Completions。
        #
        # 传入的关键参数：
        #   model       -> LLMConfig.model
        #   messages    -> 上面组装的对话消息
        #   temperature -> 输出随机性
        #   top_p       -> nucleus sampling
        response = await self.aclient.chat.completions.create(
            model=self.config.model,
            messages=message,
            temperature=self.config.temperature,
            top_p = self.config.top_p,
        )

        # 记录 token 和成本，评测时会随 workflow 输出一起汇总。
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        
        # 记录本次调用的 token / cost，并累加到 usage_tracker。
        usage_record = self.usage_tracker.add_usage(
            self.config.model,
            input_tokens,
            output_tokens
        )
        
        # OpenAI Chat Completions 的文本结果一般在：
        #   response.choices[0].message.content
        ret = response.choices[0].message.content

        # 当前代码会直接 print 模型输出和成本。
        # 这对调试很直观，但批量跑实验时可能让终端输出很多。
        # 如果后续要做生产化，可以考虑改成 logger 或通过参数控制是否打印。
        print(ret)
        
        print(f"Token usage: {input_tokens} input + {output_tokens} output = {input_tokens + output_tokens} total")
        print(f"Cost: ${usage_record['total_cost']:.6f} (${usage_record['input_cost']:.6f} for input, ${usage_record['output_cost']:.6f} for output)")
        
        return ret
    
    async def call_with_format(self, prompt: str, formatter: BaseFormatter):
        """调用模型，并用 formatter 校验/解析输出。

        普通 __call__ 只返回原始文本。

        call_with_format 适合这种场景：
            模型必须按某种结构返回，比如 XML、JSON、固定字段。

        在 AFlow 里，Optimizer 生成新 workflow 时会使用这个方法，
        因为优化模型必须返回 modification / graph / prompt。
        
        Args:
            prompt: The prompt to send to the LLM
            formatter: An instance of a BaseFormatter to validate and parse the response
            
        Returns:
            The formatted response data
            
        Raises:
            FormatError: If the response doesn't match the expected format
        """
        # 先把格式要求注入 prompt，再调用模型。
        #
        # formatter.prepare_prompt(prompt) 通常会在原 prompt 后面追加：
        #   “请按照以下 XML/JSON 格式输出……”
        formatted_prompt = formatter.prepare_prompt(prompt)

        # 仍然走统一的 __call__，所以 token 统计、成本统计也会生效。
        response = await self.__call__(formatted_prompt)
        
        # 对模型输出做结构化校验；失败会抛 FormatError 给上层兜底处理。
        #
        # validate_response 返回：
        #   is_valid: 是否符合格式
        #   parsed_data: 解析后的结构化数据
        is_valid, parsed_data = formatter.validate_response(response)
        
        if not is_valid:
            # 格式不符合时，抛出 FormatError。
            # 上层 Optimizer 会捕获这个异常，并尝试 fallback 解析。
            error_message = formatter.format_error_message()
            raise FormatError(f"{error_message}. Raw response: {response}")
        
        return parsed_data
    
    def get_usage_summary(self):
        """获取当前 AsyncLLM 实例的 token / cost 汇总。"""
        return self.usage_tracker.get_summary()    
    

def create_llm_instance(llm_config):
    """创建 AsyncLLM 实例的统一工厂函数。

    外部代码不需要关心传入的是哪种配置形式：
        1. LLMConfig 实例
        2. 模型名字符串
        3. 配置字典

    这个函数都会尽量转换成 AsyncLLM。
    
    Args:
        llm_config: Either an LLMConfig instance, a dictionary of configuration values,
                            or a string representing the LLM name to look up in default config
    
    Returns:
        An instance of AsyncLLM configured according to the provided parameters
    """
    # 对外的统一工厂函数：无论传 LLMConfig、模型名还是 dict，都返回 AsyncLLM。
    #
    # Case 1:
    #   已经是标准 LLMConfig，直接创建 AsyncLLM。
    if isinstance(llm_config, LLMConfig):
        return AsyncLLM(llm_config)
    
    # Case 2:
    #   传入模型名字符串，例如 "gpt-4o-mini"。
    #   AsyncLLM 构造函数内部会自动从 LLMsConfig.default() 查配置。
    elif isinstance(llm_config, str):
        return AsyncLLM(llm_config)  # AsyncLLM constructor handles lookup
    
    # Case 3:
    #   传入普通 dict，先转成 LLMConfig，再创建 AsyncLLM。
    elif isinstance(llm_config, dict):
        llm_config = LLMConfig(llm_config)
        return AsyncLLM(llm_config)
    
    else:
        # 传入不支持的类型时，尽早报错，避免后面调用模型时才出现更难定位的问题。
        raise TypeError("llm_config must be an LLMConfig instance, a string, or a dictionary")
