from opencompass.models import VLLM

models = [
    dict(
        type=VLLM,
        abbr='qwen3-32b-vllm',
        path='/mnt/lab/Models/qwen/Qwen3-32B',

        # ⚠️ OpenCompass 自己用的
        max_out_len=2048,

        model_kwargs=dict(
            trust_remote_code=True,
            dtype='bfloat16',

            # ⭐⭐⭐ 关键修复点 ⭐⭐⭐
            max_model_len=4096,          # 强制覆盖模型的 40960
            gpu_memory_utilization=0.92,

            tensor_parallel_size=1,
        ),

        generation_kwargs=dict(
            temperature=0.0,
            top_p=1.0,
            stop=['```'],
        ),

        batch_size=1,
        run_cfg=dict(num_gpus=1),
    )
]
