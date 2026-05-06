# opencompass/configs/exp/qwen3_32b_ceval_gen_vllm.py

from mmengine.config import read_base

with read_base():
    from ..models.qwen3_32b.qwen3_32b_vllm import models
    from ..datasets.ceval.ceval_gen import ceval_datasets as datasets

work_dir = './outputs/qwen3_32b_ceval_gen'

