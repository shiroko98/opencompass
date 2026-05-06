import re

def extract_def_postprocess(pred: str, **kwargs) -> str:
    if not pred or not isinstance(pred, str):
        return pred

    # collect import lines
    imports = []
    for line in pred.splitlines():
        if line.startswith("import ") or line.startswith("from "):
            imports.append(line)
    import_block = "\n".join(imports).strip()

    # extract first def block
    func_pattern = r"(def\s+[^\n]+\n(?:[ \t]+[^\n]*\n?)*)"
    m = re.search(func_pattern, pred)
    if not m:
        return pred

    func_block = m.group(1).strip()

    if import_block:
        return f"{import_block}\n\n{func_block}\n"
    return f"{func_block}\n"
