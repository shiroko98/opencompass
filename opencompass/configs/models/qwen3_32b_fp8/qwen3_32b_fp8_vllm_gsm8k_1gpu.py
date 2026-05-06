from opencompass.models import VLLM

models = [
    dict(
        type=VLLM,
        abbr='qwen3-32b-fp8-vllm-plain-1gpu',
        path='/mnt/lab/Models/qwen/Qwen3-32B-FP8',

        model_kwargs=dict(
            tensor_parallel_size=1,      # 单卡
            gpu_memory_utilization=0.90,
            trust_remote_code=True,
        ),

        max_seq_len=4096,
        max_out_len=512,               # GSM8K CoT 必须 >= 256
        batch_size=2,

        generation_kwargs=dict(
            temperature=0.0,
        ),

        run_cfg=dict(
            num_gpus=1,
            num_procs=1,
        ),
    )
]
