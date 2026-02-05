from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# session_id -> chat history
_STORE = {}

def get_session_history(session_id: str):
    if session_id not in _STORE:
        _STORE[session_id] = InMemoryChatMessageHistory()
    return _STORE[session_id]


def add_memory(chain):
    return RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="question",
        history_messages_key="chat_history",
    )
