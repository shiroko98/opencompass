from opencompass.models import VLLM

models = [
    dict(
        type=VLLM,
        abbr='qwen3-32b-humaneval-vllm',
        path='/mnt/lab/Models/qwen/Qwen3-32B',

        # HumanEval 需要完整函数体
        max_out_len=1024,
        max_seq_len=4096,

        model_kwargs=dict(
            trust_remote_code=True,
            dtype='bfloat16',

            # 强制覆盖模型默认超大 context，避免 vLLM 报错或浪费显存
            max_model_len=4096,
            gpu_memory_utilization=0.92,

            tensor_parallel_size=1,
        ),

        # ⭐ HumanEval 关键：低随机性 + 强 stop
        generation_kwargs=dict(
            temperature=0.0,
            top_p=1.0,
            stop=['\n\n\n'],   # 防止函数写完后继续解释
        ),

        # 代码生成不建议 batch
        batch_size=1,
        run_cfg=dict(num_gpus=1),
    )
]
