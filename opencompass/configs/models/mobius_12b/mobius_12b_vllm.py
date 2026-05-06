from opencompass.models import VLLM

models = [
    dict(
        type=VLLM,
        abbr='mobius_r6_12b',

        path='/mnt/lab/Models/pt',

        max_out_len=2048,
        batch_size=64,

        engine_config=dict(
            tensor_parallel_size=2,
            dtype='bfloat16',
            gpu_memory_utilization=0.9,
        ),
    )
]
