from opencompass.models import HuggingFace

models = [
    dict(
        type=HuggingFace,   # ⭐ 关键：不用 chat_template
        abbr='qwen3-32b-hf',
        path='/mnt/lab/Models/qwen/Qwen3-32B',

        model_kwargs=dict(
            device_map='auto',
            trust_remote_code=True,
        ),
        tokenizer_kwargs=dict(
            trust_remote_code=True,
            use_fast=False,
        ),

        max_out_len=64,     # ⭐ 只够输出一个选项
        batch_size=1,
        run_cfg=dict(num_gpus=1),
    )
]
