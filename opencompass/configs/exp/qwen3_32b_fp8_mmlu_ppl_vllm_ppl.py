from mmengine.config import read_base

with read_base():
    from ..models.qwen3_32b_fp8.qwen3_32b_fp8_vllm_ppl import models
    from ..datasets.mmlu.mmlu_ppl import mmlu_datasets as datasets
