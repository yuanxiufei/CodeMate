from langchain_ollama.chat_models import ChatOllama


if __name__ == "__main__":
    llm = ChatOllama(model="my-doctor:0.1")
    print(llm.invoke("Hello"))
