from mmengine.config import read_base

with read_base():
    # 模型
    from ..models.qwen3_32b.qwen3_32b_vllm import models

    # 数据集（你现在只选了 journalism）
    from ..datasets.cmmlu.cmmlu_gen import cmmlu_datasets as datasets
