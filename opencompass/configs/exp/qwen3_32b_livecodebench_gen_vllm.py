from mmengine.config import read_base

with read_base():
    from ..models.qwen3_32b.qwen3_32b_vllm_lcb import models
    from ..datasets.livecodebench.livecodebench_gen import LCB_datasets

# ⭐ 只选第一个：code generation
datasets = [
    LCB_datasets[0]
]
