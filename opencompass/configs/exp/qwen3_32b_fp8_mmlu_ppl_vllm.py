from mmengine.config import read_base

with read_base():
    # 模型：FP8 vLLM
    from ..models.qwen3_32b_fp8.qwen3_32b_fp8_vllm import models
    # 数据集：MMLU PPL
    from ..datasets.mmlu.mmlu_ppl import mmlu_datasets as datasets
