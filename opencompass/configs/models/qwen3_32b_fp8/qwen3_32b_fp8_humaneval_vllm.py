from opencompass.models import VLLM

models = [
    dict(
        type=VLLM,
        abbr='qwen3-32b-fp8-humaneval-vllm',
        path='/mnt/lab/Models/qwen/Qwen3-32B-FP8',

        max_out_len=1024,
        max_seq_len=4096,

        model_kwargs=dict(
            trust_remote_code=True,

            # FP8 不要再手动 dtype
            gpu_memory_utilization=0.92,
            max_model_len=4096,
            tensor_parallel_size=1,
        ),

        generation_kwargs=dict(
            temperature=0.0,
            top_p=1.0,
            stop=['\n\n\n'],
        ),

        batch_size=1,
        run_cfg=dict(num_gpus=1),
    )
]
