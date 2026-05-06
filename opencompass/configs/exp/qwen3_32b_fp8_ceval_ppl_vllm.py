# opencompass/configs/exp/qwen3_32b_fp8_ceval_ppl_vllm.py
from mmengine.config import read_base

with read_base():
    # 1️⃣ 模型
    from ..models.qwen3_32b_fp8.qwen3_32b_fp8_vllm_ppl import models
    # 2️⃣ 数据集
    from ..datasets.ceval.ceval_ppl import ceval_datasets as datasets

