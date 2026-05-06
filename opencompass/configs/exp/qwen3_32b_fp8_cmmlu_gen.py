# opencompass/configs/exp/qwen3_32b_fp8_cmmlu_gen.py

from mmengine.config import read_base

with read_base():
    # 1️⃣ FP8 vLLM 模型
    from ..models.qwen3_32b_fp8.qwen3_32b_fp8_vllm import models


    # 2️⃣ CMMLU 数据集（你自己改过 Prompt 的那套）
    from ..datasets.cmmlu.cmmlu_gen import cmmlu_datasets as datasets
