from opencompass.models import HuggingFacewithChatTemplate

models = [
    dict(
        type=HuggingFacewithChatTemplate,
        abbr='qwen3-32b-hf-chat',

        path='/mnt/lab/Models/qwen/Qwen3-32B',

        model_kwargs=dict(
            device_map='auto',
            trust_remote_code=True,
        ),
        tokenizer_kwargs=dict(
            trust_remote_code=True,
            use_fast=False,
        ),

        # ⭐ SuperGPQA 核心参数
        max_out_len=16,
        batch_size=1,

        generation_kwargs=dict(
            do_sample=False,
            temperature=0.0,
        ),

        run_cfg=dict(num_gpus=1),
    )
]
