from dotenv import load_dotenv
from search import search_prompt

load_dotenv()


def main():
    print("Chat RAG iniciado. Digite 'sair' ou 'quit' para encerrar.\n")
    while True:
        try:
            question = input("Você: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nEncerrando o chat.")
            break

        if not question:
            continue

        if question.lower() in ("sair", "quit", "exit"):
            print("Encerrando o chat.")
            break

        answer = search_prompt(question)
        print(f"\nAssistente: {answer}\n")


if __name__ == "__main__":
    main()
