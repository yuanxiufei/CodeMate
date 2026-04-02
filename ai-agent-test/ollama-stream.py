from langchain_ollama.chat_models import ChatOllama

if __name__ == "__main__":
    llm = ChatOllama(model="my-doctor:0.1")

    messages = [
        ("system", "你是一个超级牛逼的中医，可以回答所有的中医相关的问题"),
        ("human", "确实牛逼")
    ]
    response = llm.stream(messages)

    for chunk in response:
        # 打印每个 chunk 的内容 ，并实时刷新 输出 end="" 表示不换行，flush=True 表示实时刷新
        print(chunk.content, end="", flush=True)