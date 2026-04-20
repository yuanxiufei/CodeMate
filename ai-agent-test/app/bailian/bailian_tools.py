import sys

from langchain_openai import ChatOpenAI
# from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate, ChatMessagePromptTemplate
from pydantic import SecretStr



# Windows 终端默认编码可能是 GBK模型输出里如果包含 emoji/特殊字符会导致打印报错。
# 这里统一把 stdout 调整为 UTF-8，并用 replace 避免因为个别字符导致程序中断。
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# 初始化 DashScope OpenAI 兼容接口的聊天模型（Qwen Max），开启 streaming 以便流式返回内容。
llm = ChatOpenAI(
    model="qwen-max-latest",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=SecretStr("sk-ab58d5f4edc64ccf95fb7d50af022356"),
    streaming=True
)

# 第一种用PromptTemplate创建提示词模板：{something} 为变量占位符，format 时传入具体内容即可生成最终 prompt 文本。
# template = PromptTemplate.from_template("你是一个资深中医助手，叫小医仙。请给出建议：{something}")
# prompt = template.format(something="我最近有点烦躁")

# 第二种用ChatPromptTemplate创建 prompt 模板,使用 ChatPromptTemplate 进行模板化
# chat_  = ChatPromptTemplate.from_messages([
#     ("system", "你是一个资深{name}助手叫小医仙，擅长{domain}，请给出建议。"),
#     ("user", "用户问题：{question}")
# ])
# prompt = chat_.format(name="编程", domain="web开发", question="如何构建一个简单的vue应用？）")


# 第三种用ChatMessagePromptTemplate创建 prompt 模板, ChatMessagePromptTemplate进行模板化

# 创建系统提示
system_message_template = ChatMessagePromptTemplate.from_template("你是一个资深{name}助手叫小医仙，擅长{domain}，请给出建议。", role="system")

# 创建用户问题
human_message_template = ChatMessagePromptTemplate.from_template("用户问题：{question}", role="user")

# 创建 chat_prompt_template
chat_prompt_template = ChatPromptTemplate.from_messages([
system_message_template,
human_message_template,
 ])

# 传入 name, domain, question，生成 prompt 文本。
prompt = chat_prompt_template.format(name="建模师", domain="3D建模", question="你擅长什么？")



# 流式请求：llm.stream 会不断产出 chunk（增量内容），逐步打印到终端。
resp = llm.stream(prompt)
for chunk in resp:
    print(chunk.content, end="")
