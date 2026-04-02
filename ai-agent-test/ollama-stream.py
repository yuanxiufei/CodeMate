from langchain_ollama.chat_models import ChatOllama

if __name__ == "__main__":
    llm = ChatOllama(model="my-doctor:0.1")

    messages = [
        ("system", "你是一个超级牛逼的中医，可以回答所有的中医相关的问题"),
        ("human", "确实牛逼")
    ]
    response = llm.stream(messages)

    for chunk in response:
        print(chunk.content, end="", flush=True)