"""LangChain Prompt 示例（当前使用版本）。

本文件主要用于“运行演示”，把可复用的初始化与模板构建逻辑放在 common.py：
- configure_stdout_utf8(): 处理 Windows 终端编码
- get_dashscope_llm(): 初始化 DashScope OpenAI Compatible 的 ChatOpenAI
- build_*_template(): 构建不同风格的 Prompt 模板（含 few-shot）
"""

from app.bailian.common import (
    build_chat_message_prompt_template,
    build_chat_prompt_template,
    build_few_shot_translation_prompt,
    build_prompt_template,
    configure_stdout_utf8,
    get_dashscope_llm,
)

# Windows 终端下为了避免 emoji/特殊字符导致报错或乱码，先统一 stdout 编码
configure_stdout_utf8()

# 获取 DashScope 的 ChatOpenAI（从项目 .env 读取 DASHSCOPE_API_KEY）
llm = get_dashscope_llm()

# 第一种用PromptTemplate创建提示词模板：{something} 为变量占位符，format 时传入具体内容即可生成最终 prompt 文本。
# template = build_prompt_template()
# prompt = template.format(something="我最近有点烦躁")

# 第二种用ChatPromptTemplate创建 prompt 模板,使用 ChatPromptTemplate 进行模板化
# chat_prompt = build_chat_prompt_template()
# messages = chat_prompt.format_messages(name="编程", domain="web开发", question="如何构建一个简单的vue应用？")


# 第三种用ChatMessagePromptTemplate创建 prompt 模板, ChatMessagePromptTemplate进行模板化

# 创建系统提示
# chat_prompt_template = build_chat_message_prompt_template()
# messages = chat_prompt_template.format_messages(name="建模师", domain="3D建模", question="你擅长什么？")

# 流式请求：llm.stream 会不断产出 chunk（增量内容），逐步打印到终端。
# resp = llm.stream(prompt)  # 传入字符串 prompt
# resp = llm.stream(messages)  # 传入消息列表 messages
# for chunk in resp:
#     print(chunk.content, end="")




# 第四种少样本提示词模板（Few-shot Prompt Template）
few_shot_prompt = build_few_shot_translation_prompt()

# 4) 格式化生成最终提示词：此处只是生成 prompt 文本（不是模型输出）
prompt = few_shot_prompt.format(text="Thank you!")
#print(prompt)

# 构建链：prompt -> llm -> output
chain = few_shot_prompt | llm

# 模型调用, 流式返回内容print(chain), 打印到终端, 并刷新缓冲区
print("这是chain的输出结果:", chain)

# 流式调用链
resp = chain.stream(input={"text": "Thank you for your help."})


for chunk in resp:
    print(chunk.content, end="")
