from langchain_core.prompts import ChatPromptTemplate
from utils import format_docs


def build_rag_chain(llm, retriever):
    # 🔁 Step 1: Question rewriter
    rewrite_prompt = ChatPromptTemplate.from_template("""
Given the conversation history and the latest user question,
rewrite the question into a standalone question.

Conversation history:
{chat_history}

Follow-up question:
{question}

Standalone question:
""")

    # 🔍 Step 2: Final answer prompt
    answer_prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant answering questions using a document.

Conversation history:
{chat_history}

Document context:
{context}

User question:
{question}

Rules:
- Use document context first
- Use chat history to understand intent
- If the answer is not in the document, say "I don't know"
""")

    def rag_step(inputs):
        question = inputs["question"]
        chat_history = inputs.get("chat_history", [])

        # 🔁 Rewrite follow-up into standalone question
        rewritten = llm.invoke(
            rewrite_prompt.format(
                question=question,
                chat_history=chat_history
            )
        )

        standalone_question = rewritten.content if isinstance(rewritten.content, str) else rewritten.content[0]["text"]

        # 🔍 Retrieve using rewritten question
        docs = retriever.invoke(standalone_question)

        return {
            "question": question,
            "context": format_docs(docs),
            "chat_history": chat_history,
        }

    return rag_step | answer_prompt | llm
