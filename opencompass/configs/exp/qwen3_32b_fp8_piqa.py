from mmengine import read_base

with read_base():
    from ..models.qwen3_32b_fp8.qwen3_32b_fp8_vllm import models
    from ..datasets.piqa.piqa_gen import piqa_datasets as datasets
