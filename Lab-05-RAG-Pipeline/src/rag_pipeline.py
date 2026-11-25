from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

DB_DIR = "./chroma_db"
COLLECTION_NAME = "university_docs"


def main():
    # 1. Load Vector Store and LLM
    embeddings = OllamaEmbeddings(model="qwen3-embedding:0.6b")
    vector_store = Chroma(
        persist_directory=DB_DIR,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    llm = ChatOllama(model="llama3.2:3b", temperature=0.3)

    # 2. Define Query Expansion Prompt
    expansion_prompt = ChatPromptTemplate.from_template(
        """You are an AI assistant. Generate 3 different perspectives or alternative versions of the following user query to retrieve relevant documents from a vector database. Provide each generated query on a new line.

Original query: {question}"""
    )

    query_generator = expansion_prompt | llm | StrOutputParser()

    # 3. Final QA Prompt
    qa_prompt = ChatPromptTemplate.from_template(
        """Answer the question based ONLY on the following context. If the answer is not contained in the context, state that it is not available.

Context:
{context}

Question: {question}

Answer:"""
    )

    print("\n--- University AI Assistant with Query Expansion ---")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        user_query = input("Ask a university question (or exit): ").strip()
        if user_query.lower() in ["exit", "quit"]:
            break
        if not user_query:
            continue

        print("\n[Query Expansion] Generating alternate queries...")
        # Generate variations
        generated_queries_str = query_generator.invoke({"question": user_query})
        queries = [q.strip() for q in generated_queries_str.strip().split("\n") if q.strip()]
        queries.append(user_query)  # Always include original query

        print(f"[Query Expansion] Searching vector store across {len(queries)} query variations...")

        # Multi-query retrieval with deduplication
        retrieved_docs_dict = {}
        for q in queries:
            docs = retriever.invoke(q)
            for doc in docs:
                # Deduplicate based on page content
                retrieved_docs_dict[doc.page_content] = doc

        unique_docs = list(retrieved_docs_dict.values())
        context_text = "\n\n".join([doc.page_content for doc in unique_docs])

        # Generate response using combined context
        formatted_prompt = qa_prompt.format(context=context_text, question=user_query)
        response = llm.invoke(formatted_prompt)

        print(f"\nAnswer:\n{response.content}\n" + "-" * 50 + "\n")


if __name__ == "__main__":
    main()