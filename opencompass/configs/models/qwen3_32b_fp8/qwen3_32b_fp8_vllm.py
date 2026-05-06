# opencompass/configs/models/qwen3_32b_fp8/qwen3_32b_fp8_vllm.py

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
        abbr='qwen3-32b-fp8-vllm-tp1',
        path='/mnt/lab/Models/qwen/Qwen3-32B-FP8',

        model_kwargs=dict(
            tensor_parallel_size=1,
            gpu_memory_utilization=0.90,
            trust_remote_code=True,
        ),

        max_seq_len=4096,
        max_out_len=32,
        min_out_len=1,
        batch_size=2,
        generation_kwargs=dict(temperature=0),

        meta_template=api_meta_template,
        stop_words=['<|endoftext|>', '<|im_end|>', '<|end|>'],

        run_cfg=dict(num_gpus=1, num_procs=1),
    )
]
