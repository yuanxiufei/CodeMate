"""bailian 目录的公共能力（可复用组件）。
这个模块把“可复用、与业务无关”的部分集中起来，供脚本直接 import 使用：
- 终端输出编码处理（Windows 下常见）
- 从项目 .env 加载 DashScope API Key
- 初始化 DashScope OpenAI Compatible 的 LangChain ChatOpenAI
- 构建常见 Prompt 模板（PromptTemplate / ChatPromptTemplate / Few-shot）
"""

import os
import sys
from pathlib import Path

from langchain_core.prompts import (
    ChatMessagePromptTemplate,
    ChatPromptTemplate,
    FewShotPromptTemplate,
    PromptTemplate,
)
from langchain_openai import ChatOpenAI
from pydantic import SecretStr


# Windows 终端默认编码可能是 GBK，模型输出里如果包含 emoji/特殊字符会导致打印报错或乱码。
# 这里把 stdout 调整为 UTF-8，并用 replace 避免因为个别字符导致程序中断。
def configure_stdout_utf8() -> None:
    """把 stdout 设置为 UTF-8，避免 Windows 控制台编码导致的异常/乱码。"""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# 项目级别配置：从当前文件目录开始向上查找 .env 并加载 KEY=VALUE（不依赖系统级环境变量）。
def load_env_from_parents(filename: str = ".env") -> None:
    """从当前文件目录开始向上查找 .env，并把 KEY=VALUE 加载进 os.environ。"""
    current = Path(__file__).resolve()
    for parent in current.parents:
        env_path = parent / filename
        if not env_path.is_file():
            continue
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            os.environ.setdefault(key, value)
        return


# 初始化 DashScope OpenAI 兼容接口的聊天模型（Qwen Max），开启 streaming 以便流式返回内容。
def get_dashscope_llm(
    *,
    model: str = "qwen-max-latest",
    streaming: bool = True,
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key_env: str = "DASHSCOPE_API_KEY",
) -> ChatOpenAI:
    """创建 DashScope OpenAI Compatible 的 ChatOpenAI（读取 DASHSCOPE_API_KEY）。"""
    load_env_from_parents(".env")
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise RuntimeError(f"请在项目目录创建 .env 并设置 {api_key_env}=xxx")
    return ChatOpenAI(
        model=model,
        base_url=base_url,
        api_key=SecretStr(api_key),
        streaming=streaming,
    )


# 少样本提示词模板（Few-shot Prompt Template）
# 结构：prefix（任务要求）+ examples（示例）+ suffix（输入格式）
def build_few_shot_translation_prompt() -> FewShotPromptTemplate:
    """构建一个“英译中”的 few-shot 提示词模板（prefix + examples + suffix）。"""
    example_template = "输入：{input}\n输出：{output}"
    example_prompt = PromptTemplate.from_template(example_template)
    examples = [
        {"input": "hello", "output": "你好"},
        {"input": "how are you", "output": "我很好"},
    ]
    return FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
        prefix="请将以下英文翻译成中文：",
        suffix="输入：{text}\n输出：",
        input_variables=["text"],
    )


# PromptTemplate：最基础的字符串模板，适合简单场景（直接 format 生成 prompt 文本）。
def build_prompt_template() -> PromptTemplate:
    """构建最基础的 PromptTemplate（适合简单字符串 prompt）。"""
    return PromptTemplate.from_template("你是一个资深中医助手，叫小医仙。请给出建议：{something}")


# ChatPromptTemplate：消息级模板，适合 chat 模型（system/user 多轮结构更清晰）。
def build_chat_prompt_template() -> ChatPromptTemplate:
    """构建 ChatPromptTemplate（system/user 消息模板）。"""
    return ChatPromptTemplate.from_messages(
        [
            ("system", "你是一个资深{name}助手叫小医仙，擅长{domain}，请给出建议。"),
            ("user", "用户问题：{question}"),
        ]
    )


# ChatMessagePromptTemplate：可以先创建单条消息模板，再组装成 ChatPromptTemplate。
def build_chat_message_prompt_template() -> ChatPromptTemplate:
    """用 ChatMessagePromptTemplate 先定义单条消息模板，再组合成 ChatPromptTemplate。"""
    system_message_template = ChatMessagePromptTemplate.from_template(
        "你是一个资深{name}助手叫小医仙，擅长{domain}，请给出建议。",
        role="system",
    )
    human_message_template = ChatMessagePromptTemplate.from_template(
        "用户问题：{question}",
        role="user",
    )
    return ChatPromptTemplate.from_messages(
        [
            system_message_template,
            human_message_template,
        ]
    )
