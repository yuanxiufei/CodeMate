from langchain_ollama.chat_models import ChatOllama


if __name__ == "__main__":
    model = ChatOllama(model="my-doctor:0.2")
    print(model.invoke("Hello"))
