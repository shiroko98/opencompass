import opencompass.models.mobius  # noqa: F401  # 强制触发注册

from mmengine.config import read_base

with read_base():
    from .datasets.gsm8k.gsm8k_gen import gsm8k_datasets

datasets = gsm8k_datasets

from opencompass.models.mobius import Mobius  # noqa: F401

models = [
    dict(
        type=Mobius,
        abbr='mobius-r6-12b-v4.2.1',
        path='/mnt/lab/Models/pt/Mobius-r6-chat-CHNtuned-12b-16k-v4.2.1',
        tokenizer_path='/mnt/lab/Models/pt/RWKV7/RWKV7-2.9B-World3-128k-250225/rwkv_vocab_v20230424.txt',
        max_seq_len=2048,
        max_out_len=128,
        batch_size=1,
        run_cfg=dict(num_gpus=1),
        generation_kwargs=dict(
            temperature=0.0,
            top_p=1.0,
            top_k=0,
        ),
    )
]
