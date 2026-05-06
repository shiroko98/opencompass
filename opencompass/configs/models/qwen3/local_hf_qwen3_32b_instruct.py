from opencompass.models import HuggingFacewithChatTemplate

models = [
    dict(
        type=HuggingFacewithChatTemplate,
        abbr='qwen3-32b-hf',
        path='/mnt/lab/Models/qwen/Qwen3-32B',
        max_out_len=2048,
        batch_size=1,
        run_cfg=dict(num_gpus=1),
    )
]
