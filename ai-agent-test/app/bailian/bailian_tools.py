from langchain_openai  import  ChatOpenAI
from pydantic import  SecretStr
llm = ChatOpenAI(
    model="qwen-max-latest",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=SecretStr("sk-ab58d5f4edc64ccf95fb7d50af022356"),
    streaming=True
)
resp = llm.stream("100+100=?")
for chunk in resp:
    print(chunk.content, end="")