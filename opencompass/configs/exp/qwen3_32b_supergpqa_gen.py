from mmengine.config import read_base

with read_base():
    from opencompass.configs.datasets.supergpqa.supergpqa_gen import supergpqa_datasets
    from opencompass.configs.models.qwen3_32b.qwen3_32b_vllm import models
    from opencompass.configs.summarizers.medium import summarizer
