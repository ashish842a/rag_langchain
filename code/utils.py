def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def extract_text(result):
    if isinstance(result.content, str):
        return result.content

    if isinstance(result.content, list):
        texts = []
        for part in result.content:
            if isinstance(part, dict) and part.get("type") == "text":
                texts.append(part.get("text", ""))
        return "\n".join(texts)

    return str(result.content)
