import sys

from langchain_openai import ChatOpenAI
# from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate, ChatMessagePromptTemplate, few_shot_with_templates
from pydantic import SecretStr


# 这个文件用于演示 LangChain Prompt 的几种常见写法（含 few-shot），并通过 DashScope 的 OpenAI 兼容接口进行流式调用。
# 主要分为四部分：
# 1) PromptTemplate：纯字符串模板（更像传统的 format）
# 2) ChatPromptTemplate：面向 chat 的消息模板（system/user 结构）
# 3) ChatMessagePromptTemplate：先定义单条消息模板，再组装成 chat 模板
# 4) FewShotPromptTemplate：少样本提示词（prefix + examples + suffix）
#
# 运行方式（示例）：
#   python app/bailian/bailian_prompt_old.py
#
# 注意：此文件中 api_key 是硬编码示例，实际项目建议改为从环境变量或 .env 加载，避免泄露。

# Windows 终端默认编码可能是 GBK模型输出里如果包含 emoji/特殊字符会导致打印报错。
# 这里统一把 stdout 调整为 UTF-8，并用 replace 避免因为个别字符导致程序中断。
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# 初始化 DashScope OpenAI 兼容接口的聊天模型（Qwen Max），开启 streaming 以便流式返回内容。
# base_url 指向 DashScope 的 OpenAI Compatible API。
# streaming=True：后续 llm.stream / chain.stream 会返回一个可迭代对象，逐步产出增量内容 chunk。
llm = ChatOpenAI(
    model="qwen-max-latest",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=SecretStr("sk-ab58d5f4edc64ccf95fb7d50af022356"),
    streaming=True
)

# 第一种用PromptTemplate创建提示词模板：{something} 为变量占位符，format 时传入具体内容即可生成最终 prompt 文本。
# 特点：生成的是一段文本 prompt（不是消息列表）。
# 适用：非常简单的单轮指令场景。
# template = PromptTemplate.from_template("你是一个资深中医助手，叫小医仙。请给出建议：{something}")
# prompt = template.format(something="我最近有点烦躁")

# 第二种用ChatPromptTemplate创建 prompt 模板,使用 ChatPromptTemplate 进行模板化
# 特点：生成的是 chat messages（system/user 等角色消息），更适配 ChatOpenAI。
# 适用：需要明确 system 指令、并保持对话结构清晰的场景。
# chat_  = ChatPromptTemplate.from_messages([
#     ("system", "你是一个资深{name}助手叫小医仙，擅长{domain}，请给出建议。"),
#     ("user", "用户问题：{question}")
# ])
# prompt = chat_.format(name="编程", domain="web开发", question="如何构建一个简单的vue应用？）")


# 第三种用ChatMessagePromptTemplate创建 prompt 模板, ChatMessagePromptTemplate进行模板化
# 特点：把每一条消息先定义成“可复用的小模板”，再组合成一个 ChatPromptTemplate。
# 适用：你的 system/human 模板在多个场景复用、或者希望更细粒度组合时。

# 创建系统提示
# system_message_template = ChatMessagePromptTemplate.from_template("你是一个资深{name}助手叫小医仙，擅长{domain}，请给出建议。", role="system")

# 创建用户问题
# human_message_template = ChatMessagePromptTemplate.from_template("用户问题：{question}", role="user")

# 创建 chat_prompt_template
# chat_prompt_template = ChatPromptTemplate.from_messages([
# system_message_template,
# human_message_template,
#  ])

# 传入 name, domain, question，生成 prompt 文本。
# prompt = chat_prompt_template.format(name="建模师", domain="3D建模", question="你擅长什么？")

# 流式请求：llm.stream 会不断产出 chunk（增量内容），逐步打印到终端。
# 说明：如果 prompt 是字符串，通常是“单轮文本”调用；如果 prompt 是消息列表，则是“chat messages”调用。
# resp = llm.stream(prompt)
# for chunk in resp:
#     print(chunk.content, end="")




# 第四种少样本提示词模板（Few-shot Prompt Template）
# 核心结构：
# - prefix：任务说明（让模型知道要做什么）
# - examples：少量输入/输出示例（引导模型模仿格式与行为）
# - suffix：用户输入的固定格式（包含变量占位符），通常以“输出：”结尾让模型继续生成
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate
# 1) 定义示例模板：用于描述“示例的输入/输出”长什么样（examples 里的 key 要和这里一致）
example_template = "输入：{input}\n输出：{output}"
example_prompt = PromptTemplate.from_template(example_template)

# 2) 准备示例数据：2-3 组典型示例即可
# 注意：示例质量会直接影响模型输出格式与风格。
examples = [
    {"input": "hello", "output": "你好"},
    {"input": "how are you", "output": "我很好"},
]

# 3) 构建提示模板：prefix（任务说明）+ examples（示例）+ suffix（用户输入格式）
# input_variables 需要包含 suffix 中出现的变量（这里是 text）。
few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix="请将以下英文翻译成中文：",
    suffix="输入：{text}\n输出：",
    input_variables=["text"],
)

# 4) 格式化生成最终提示词：此处只是生成 prompt 文本（不是模型输出）
# 你可以先 print(prompt) 观察最终拼接出来的提示词长什么样。
prompt = few_shot_prompt.format(text="Thank you!")
#print(prompt)

# 构建链：prompt -> llm -> output
# 这是一种更“LangChain 化”的写法：把 prompt 与模型组成可执行链。
chain = few_shot_prompt | llm

# 模型调用, 流式返回内容print(chain), 打印到终端, 并刷新缓冲区
# print(chain) 打印的是链对象的结构描述，不是模型输出。
print("这是chain的输出结果:", chain)

# 流式调用链
# 传参方式：input={"text": "..."} 对应 few_shot_prompt 的 input_variables=["text"]。
resp = chain.stream(input={"text": "Thank you for your help."})


for chunk in resp:
    print(chunk.content, end="")
