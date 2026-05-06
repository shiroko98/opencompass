from copy import deepcopy

from mmengine.config import read_base

from opencompass.models.mobius import Mobius
from opencompass.partitioners import NaivePartitioner, NumWorkerPartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLEvalTask, OpenICLInferTask

with read_base():
    from ..datasets.ceval.ceval_gen_mobius_thinking import ceval_datasets


datasets = ceval_datasets

model_meta_template = dict(
    begin=(
        '<|im_start|>System: You are Qwen, created by Alibaba Cloud. '
        'You are a helpful assistant.<|im_end|>\n'
    ),
    round=[
        dict(role='HUMAN', begin='<|im_start|>User: ', end='<|im_end|>\n'),
        dict(
            role='BOT',
            begin='<|im_start|>Assistant: ',
            end='<|im_end|>\n',
            generate=True,
        ),
    ],
)

base_model_specs = [
    dict(
        abbr='mobius_thinking_model',
        path='/mnt/lab/Models/pt/Mobius-r6-chat-CHNtuned-12b-16k-v4.2.1.pth',
        tokenizer_path='/mnt/data/Codes/RWKV/RWKV_tokenizer/rwkv_vocab_v20250609.txt',
        strategy='cuda fp16',
        max_seq_len=4096,
        max_out_len=256,
        batch_size=1,
        run_cfg=dict(num_gpus=1),
    ),
]

models = [
    dict(
        type=Mobius,
        abbr=spec['abbr'],
        path=spec['path'],
        tokenizer_path=spec['tokenizer_path'],
        strategy=spec.get('strategy', 'cuda fp16'),
        max_seq_len=spec.get('max_seq_len', 4096),
        max_out_len=spec.get('max_out_len', 256),
        batch_size=spec.get('batch_size', 1),
        run_cfg=deepcopy(spec.get('run_cfg', dict(num_gpus=1))),
        meta_template=deepcopy(
            spec.get('meta_template', model_meta_template)),
        stop_words=deepcopy(spec.get('stop_words', [
            '<|im_end|>',
            '<|endoftext|>',
            '<|im_start|>User:',
            '<|im_start|>System:',
            '<|im_start|>Assistant:',
        ])),
        generation_kwargs=deepcopy(spec.get('generation_kwargs', dict(
            temperature=0.0,
            top_p=1.0,
            top_k=1,
            alpha_frequency=0.0,
            alpha_presence=0.0,
        ))),
    ) for spec in base_model_specs
]

infer = dict(
    partitioner=dict(
        type=NumWorkerPartitioner,
        num_worker=16,
        num_split=16,
        min_task_size=8,
    ),
    runner=dict(
        type=LocalRunner,
        max_num_workers=16,
        max_workers_per_gpu=2,
        task=dict(type=OpenICLInferTask),
    ),
)

eval = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(
        type=LocalRunner,
        max_num_workers=16,
        task=dict(type=OpenICLEvalTask),
    ),
)

work_dir = './outputs/mobius_thinking_ceval_gen'
