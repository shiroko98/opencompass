from mmengine.config import read_base

with read_base():
    # ✅ 用你现有的 vLLM 模型配置
    from ..models.qwen3_32b.qwen3_32b_vllm_debug import models
    # ✅ MMLU PPL 数据集
    from ..datasets.mmlu.mmlu_ppl import mmlu_datasets as datasets