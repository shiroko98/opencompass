# opencompass/configs/models/qwen3_32b_fp8/qwen3_32b_fp8_vllm_base.py

from opencompass.models import VLLM

models = [
    dict(
        type=VLLM,                      # ⭐ 关键：不用 ChatTemplate
        abbr='qwen3-32b-fp8-vllm-base',
        path='/mnt/lab/Models/qwen/Qwen3-32B-FP8',

        model_kwargs=dict(
            tensor_parallel_size=2,
            gpu_memory_utilization=0.90,
            trust_remote_code=True,
        ),

        max_seq_len=4096,
        max_out_len=64,                 # ⭐ 和 HF 对齐，足够出 A/B/C/D
        batch_size=8,

        generation_kwargs=dict(
            temperature=0.0,
            top_p=1.0,
        ),

        run_cfg=dict(num_gpus=2),
    )
]
