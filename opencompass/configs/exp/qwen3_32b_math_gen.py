from mmengine.config import read_base

with read_base():
    # 模型
    from ..models.qwen3_32b.qwen3_32b_vllm import models
    # 数据集
    from ..datasets.math.math_gen import math_datasets as datasets
