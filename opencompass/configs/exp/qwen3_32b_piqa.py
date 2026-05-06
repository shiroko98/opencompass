import importlib.util


def load_var(pyfile, varname):
    spec = importlib.util.spec_from_file_location("tmp_cfg", pyfile)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, varname)


models = load_var(
    "configs/models/qwen3_32b/qwen3_32b_vllm.py",
    "models"
)

datasets = load_var(
    "configs/datasets/piqa/piqa_ppl_1cf9f0.py",
    "piqa_datasets"
)
