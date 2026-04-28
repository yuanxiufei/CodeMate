"""DashScope（阿里云百炼）OpenAI Compatible API 最简流式调用示例。

要点：
- OpenAI() 里 base_url 指向 DashScope 的 OpenAI Compatible 入口
- stream=True 会以增量 chunk 形式返回内容，需要循环拼接/打印
"""

from openai import OpenAI

client = OpenAI(
    api_key="sk-ab58d5f4edc64ccf95fb7d50af022356",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    default_headers={
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.90 Safari/537.36",
    }
)

completion = client.chat.completions.create(
    model="qwen3.6-plus",
    messages=[
        {"role": "system", "content": "你是一个超级牛逼的中医，可以回答所有的中医相关的问题"},
        {"role": "user", "content": "确实牛逼"}
    ],
    # 开启流式：每次迭代返回一小段增量内容（delta）
    stream=True
)
for chunk in completion:
    # OpenAI Compatible 的流式返回：文本通常在 delta.content 里
    content = chunk.choices[0].delta.content
    if content:
        print(content, end="", flush=True)
