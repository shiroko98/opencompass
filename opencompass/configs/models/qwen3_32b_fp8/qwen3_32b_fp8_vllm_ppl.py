# opencompass/configs/models/qwen3_32b_fp8/qwen3_32b_fp8_vllm_ppl.py

from opencompass.models import VLLM

models = [
    dict(
        type=VLLM,
        abbr='qwen3-32b-fp8-vllm-ppl-1gpu',
        path='/mnt/lab/Models/qwen/Qwen3-32B-FP8',

        model_kwargs=dict(
            tensor_parallel_size=1,
            gpu_memory_utilization=0.90,
            trust_remote_code=True,
        ),

        max_seq_len=4096,
        batch_size=2,   # PPL 单卡安全值

        run_cfg=dict(
            num_gpus=1,
            num_procs=1,
        ),
    )
]
