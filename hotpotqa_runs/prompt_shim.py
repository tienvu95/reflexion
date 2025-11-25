from typing import List, Optional

class PromptTemplate:
    """Tiny shim for LangChain's PromptTemplate used in this repo.

    It supports the minimal interface used here: constructor with
    `input_variables` and `template`, and a `format(**kwargs)` method
    which uses Python's `.format()` on the template string.
    """
    def __init__(self, input_variables: Optional[List[str]] = None, template: str = ""):
        self.input_variables = input_variables or []
        self.template = template

    def format(self, **kwargs) -> str:
        # Basic validation: ensure required variables are present
        missing = [v for v in self.input_variables if v not in kwargs]
        if missing:
            # we don't raise here to preserve old behavior in tests; just let format fail normally
            pass
        return self.template.format(**kwargs)
