from mmengine.config import read_base

with read_base():
    # 1️⃣ 模型配置（和你之前跑 gen 时用的是同一份）
    from opencompass.configs.models.qwen3_32b.qwen3_32b_hf import models

    # 2️⃣ HellaSwag PPL 数据集（你已经确认存在）
    from opencompass.configs.datasets.hellaswag.hellaswag_ppl_47bff9 import hellaswag_datasets

# 3️⃣ 绑定
datasets = hellaswag_datasets
