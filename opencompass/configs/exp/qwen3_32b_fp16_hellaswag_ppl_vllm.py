from mmengine.config import read_base

with read_base():
    from opencompass.configs.models.qwen3_32b.qwen3_32b_vllm_fp16_ppl import models
    from opencompass.configs.datasets.hellaswag.hellaswag_ppl_47bff9 import hellaswag_datasets

datasets = hellaswag_datasets
