from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("DASHSEEK_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    default_headers={
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.90 Safari/537.36",
    }
)

completion = client.chat.completions.create(
    model="deepseek-r1:7b",
    messages=[
        {"role": "system", "content": "你是一个超级牛逼的中医，可以回答所有的中医相关的问题"},
        {"role": "human", "content": "确实牛逼"}
    ],
    stream=True
)
for chunk in completion:
    print(chunk.choices[0].delta.content, end="", flush=True)