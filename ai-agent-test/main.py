from langchain_ollama import ChatOllama


if __name__ == "__main__":
    model = ChatOllama(model="my-doctor:0.2")
    print(model.invoke("Hello"))
