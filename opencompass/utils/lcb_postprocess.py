import re

def extract_last_python_block(text):
    if not isinstance(text, str):
        return ""
    blocks = re.findall(r"```python\s*([\s\S]*?)```", text)
    if not blocks:
        return ""
    return "```python\n" + blocks[-1].strip() + "\n```"
