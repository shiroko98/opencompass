from opencompass.models import VLLMwithChatTemplate

api_meta_template = dict(
    round=[
        dict(role='HUMAN', api_role='HUMAN'),
        dict(role='BOT', api_role='BOT', generate=True),
    ],
)

models = [
    dict(
        type=VLLMwithChatTemplate,
        abbr='qwen3-32b-fp8-vllm-tp2-arc',
        path='/mnt/lab/Models/qwen/Qwen3-32B-FP8',

        model_kwargs=dict(
            tensor_parallel_size=2,
            gpu_memory_utilization=0.90,
            trust_remote_code=True,
        ),

        max_seq_len=4096,

        # ✅ 给足输出长度：让它能输出到最终答案字母
        max_out_len=256,
        min_out_len=1,

        # ✅ 不要乱 stop（否则只剩 <think>）
        stop_words=['<|endoftext|>', '<|im_end|>', '<|end|>'],

        batch_size=16,
        generation_kwargs=dict(
            temperature=0.0,
            top_p=1.0,
        ),

        meta_template=api_meta_template,
        run_cfg=dict(num_gpus=2, num_procs=1),
    )
]
