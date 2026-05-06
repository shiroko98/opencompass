from mmengine.config import read_base

with read_base():
    # ✅ 用你已有的 vLLM 模型
    from ..models.qwen3_32b.qwen3_32b_vllm import models
    # ✅ 用我们刚建的 debug dataset
    from ..datasets.mmlu.mmlu_ppl_debug import mmlu_datasets as datasets
