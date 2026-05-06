from copy import deepcopy

from mmengine.config import read_base

from opencompass.models.mobius import Mobius
from opencompass.partitioners import NaivePartitioner, NumWorkerPartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLEvalTask, OpenICLInferTask

with read_base():
    from ..datasets.ceval.ceval_val_ppl_mobius import ceval_val_ppl_datasets


datasets = ceval_val_ppl_datasets


model_meta_template = dict(
    round=[
        dict(role='HUMAN', begin='User: ', end='\n\n'),
        dict(role='BOT', begin='Assistant: ', end='\n\n', generate=True),
    ],
)


base_model_specs = [
    # dict(
    #     abbr='mobius_r6_12b_v7_1',
    #     path='/mnt/lab/Models/pt/Mobius-r6-chat-CHNtuned-12b-28k-v7_1.pth',
    #     tokenizer_path='rwkv_vocab_v20230424',
    #     strategy='cuda fp16',
    #     max_seq_len=4096,
    #     batch_size=1,
    #     run_cfg=dict(num_gpus=1),
    # ),
    dict(
        abbr='mobius_r6_12b_v4_2_1',
        path='/mnt/lab/Models/pt/Mobius-r6-chat-CHNtuned-12b-16k-v4.2.1.pth',
        tokenizer_path='rwkv_vocab_v20230424',
        strategy='cuda fp16',
        max_seq_len=4096,
        batch_size=1,
        run_cfg=dict(num_gpus=1),
    ),
]

models = [
    dict(
        type=Mobius,
        abbr=spec['abbr'],
        path=spec['path'],
        tokenizer_path=spec.get('tokenizer_path',
                                'rwkv_vocab_v20230424'),
        strategy=spec.get('strategy', 'cuda fp16'),
        max_seq_len=spec.get('max_seq_len', 4096),
        max_out_len=1,
        batch_size=spec.get('batch_size', 1),
        run_cfg=deepcopy(spec.get('run_cfg', dict(num_gpus=1))),
        meta_template=deepcopy(
            spec.get('meta_template', model_meta_template)),
    ) for spec in base_model_specs
]

infer = dict(
    partitioner=dict(
        type=NumWorkerPartitioner,
        num_worker=8,
        num_split=8,
        min_task_size=16,
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

work_dir = './outputs/mobius_multi_model_ceval_val_ppl'
