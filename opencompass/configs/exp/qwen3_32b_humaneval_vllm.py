from mmengine.config import read_base

with read_base():
    # ✅ HumanEval 专用模型（你刚刚新建的）
    from ..models.qwen3_32b.qwen3_32b_humaneval_vllm import models

    # ✅ HumanEval 数据集
    from ..datasets.humaneval.humaneval_gen import humaneval_datasets as datasets
