from mmengine.config import read_base

with read_base():
    # 模型
    from opencompass.configs.models.qwen3_32b_fp8.qwen3_32b_fp8_humaneval_vllm import models

    
    # 数据集
    from opencompass.configs.datasets.humaneval.humaneval_gen import humaneval_datasets


