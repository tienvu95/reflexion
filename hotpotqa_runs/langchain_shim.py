from typing import Any


class BaseLLM:
    pass


class BaseChatModel:
    pass


class SystemMessage:
    def __init__(self, content: str):
        self.content = content


class HumanMessage:
    def __init__(self, content: str):
        self.content = content


class AIMessage:
    def __init__(self, content: str):
        self.content = content


# Minimal placeholders for docstore / wikipedia types used only for type hints.
class Wikipedia:
    def __init__(self, *args, **kwargs):
        raise RuntimeError("Wikipedia docstore shim used: install langchain if you need docstore features")


class Docstore:
    pass


class DocstoreExplorer:
    def __init__(self, ds: Any):
        raise RuntimeError("DocstoreExplorer shim used: install langchain if you need docstore features")
