# opencompass/configs/exp/qwen3_32b_fp8_gsm8k_gen_vllm.py
from mmengine.config import read_base

with read_base():
    # 你的 FP8 vLLM 模型
    from ..models.qwen3_32b_fp8.qwen3_32b_fp8_vllm_gsm8k_1gpu import models

    # 你的 gsm8k gen 数据集
    from ..datasets.gsm8k.gsm8k_gen import gsm8k_datasets as datasets
