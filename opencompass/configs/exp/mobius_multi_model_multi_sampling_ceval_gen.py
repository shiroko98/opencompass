from copy import deepcopy

from mmengine.config import read_base

from opencompass.models.mobius import Mobius

with read_base():
    from ..datasets.ceval.ceval_gen_mobius_force_choice import ceval_datasets


datasets = ceval_datasets
# datasets = ceval_test_ppl_datasets


model_meta_template = dict(
    round=[
        dict(role='HUMAN', begin='User: ', end='\n\n'),
        dict(role='BOT', begin='Assistant:', end='\n\n', generate=True),
    ],
)


# Fill in your real model checkpoints here. `path` should point to a ChatRWKV
# compatible `.pth` checkpoint, not a HuggingFace directory.
base_model_specs = [
    dict(
        abbr='mobius_r6_12b_v7_1',
        path='/mnt/data/Models/RWKV/Mobius-r6-chat-CHNtuned-12b-28k-v7_1.pth',
        tokenizer_path='rwkv_vocab_v20230424',
        strategy='cuda fp16',
        max_seq_len=4096,
        max_out_len=32,
        batch_size=1,
        run_cfg=dict(num_gpus=1),
    ),
    dict(
        abbr='mobius_r6_12b_v4_2_1',
        path='/mnt/data/Models/RWKV/Mobius-r6-chat-CHNtuned-12b-16k-v4.2.1.pth',
        tokenizer_path='rwkv_vocab_v20230424',
        strategy='cuda fp16',
        max_seq_len=4096,
        max_out_len=32,
        batch_size=1,
        run_cfg=dict(num_gpus=1),
    ),
]


# Each base model will be duplicated once per preset below.
# For PPL datasets, these generation parameters do not affect results.
sampling_presets = [
    dict(
        name='greedy',
        temperature=0.0,
        top_p=0.0,
        top_k=1,
        alpha_frequency=0.0,
        alpha_presence=0.0,
    ),
    dict(
        name='t08_p02_penalty',
        temperature=0.8,
        top_p=0.2,
        top_k=0,
        alpha_frequency=0.4,
        alpha_presence=0.4,
    ),
    dict(
        name='t10_p02_penalty',
        temperature=1.0,
        top_p=0.2,
        top_k=0,
        alpha_frequency=0.4,
        alpha_presence=0.4,
    ),
    dict(
        name='t10_p05_penalty',
        temperature=1.0,
        top_p=0.5,
        top_k=0,
        alpha_frequency=0.4,
        alpha_presence=0.4,
    ),
]

models = [
    dict(
        type=Mobius,
        abbr=f"{spec['abbr']}__{preset['name']}",
        path=spec['path'],
        tokenizer_path=spec.get('tokenizer_path',
                                'rwkv_vocab_v20230424'),
        strategy=spec.get('strategy', 'cuda fp16'),
        max_seq_len=spec.get('max_seq_len', 4096),
        max_out_len=spec.get('max_out_len', 32),
        batch_size=spec.get('batch_size', 1),
        run_cfg=deepcopy(spec.get('run_cfg', dict(num_gpus=1))),
        meta_template=deepcopy(
            spec.get('meta_template', model_meta_template)),
        stop_words=deepcopy(spec.get(
            'stop_words', ['\n\nUser:', '\n\nQ:', '\n\nQuestion:'])),
        generation_kwargs={
            k: deepcopy(v)
            for k, v in preset.items() if k != 'name'
        },
    ) for spec in base_model_specs for preset in sampling_presets
]

work_dir = './outputs/mobius_multi_model_multi_sampling_ceval_gen'
