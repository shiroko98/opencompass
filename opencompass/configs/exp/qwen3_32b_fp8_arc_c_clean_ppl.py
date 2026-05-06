from mmengine import read_base

with read_base():
    from ..models.qwen3_32b_fp8.qwen3_32b_fp8_vllm import models
    from ..datasets.ARC_c.ARC_c_clean_ppl import ARC_c_datasets as datasets
