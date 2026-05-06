from mmengine.config import read_base

with read_base():
    # =========================
    # 模型配置（你已有）
    # =========================
    from ..models.qwen3_32b.qwen3_32b_vllm import models

    # =========================
    # 数据集配置（mmlu_pro gen）
    # =========================
    from ..datasets.mmlu_pro.mmlu_pro_gen import mmlu_pro_datasets as datasets


