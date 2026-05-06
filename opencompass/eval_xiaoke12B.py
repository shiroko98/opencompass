from opencompass.models import HuggingFaceCausalLM
from opencompass.models.turbomind import TurboMindModel
from opencompass.runners import SlurmSequentialRunner
from opencompass.partitioners import SizePartitioner, NaivePartitioner, NumWorkerPartitioner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask
from opencompass.runners import LocalRunner
from opencompass.models.mobius import Mobius

from mmengine.config import read_base
with read_base():
    # from .datasets.hellaswag.hellaswag_ppl import hellaswag_datasets
    # from .datasets.winogrande.winogrande_ppl_9307fd import winogrande_datasets
    # from .datasets.qbqa.qbqa_ppl import qbqa_datasets
    # from .datasets.collections.chat_medium import datasets
    # from .summarizers.medium import summarizer
    # from .datasets.piqa.piqa_ppl import piqa_datasets
    # from .datasets.siqa.siqa_ppl import siqa_datasets
    # from .datasets.GLUE_CoLA.GLUE_CoLA_ppl_77d0df import CoLA_datasets
    # from .datasets.ARC_e.ARC_e_gen import ARC_e_datasets
    # from .datasets.ARC_c.ARC_c_gen import ARC_c_datasets
    # from .datasets.commonsenseqa_cn.commonsenseqacn_ppl import commonsenseqacn_datasets
    
    # from .datasets.mmlu.mmlu_ppl import mmlu_datasets  27.09

    # from .datasets.gpqa.gpqa_ppl_6bf57a import gpqa_datasets  20.20

    # from .datasets.math.math_gen import math_datasets
    from .datasets.gsm8k.gsm8k_gen import gsm8k_datasets
    
    # from .datasets.humaneval.humaneval_gen import humaneval_datasets

    # from .datasets.cmmlu.cmmlu_ppl import cmmlu_datasets  26.09
    # from .datasets.ceval.ceval_ppl import ceval_datasets  27.53
    
    
datasets = sum([v for k, v in locals().items() if ('datasets' in k)], [])

model_meta_template = dict(
    round=[
            dict(role='HUMAN', begin='User: ', end='\n\n'),
            dict(role='BOT', begin='Assistant: ', end='\n\n', generate=True),
    ],
    begin=f"""User: hi

        Assistant: Hi. I am your assistant and I will provide expert full response in full details. Please feel free to ask any question and I will always answer it.

        """,
    )

api_meta_template = dict(
        round=[
            dict(role='HUMAN', api_role='HUMAN'),
            dict(role='BOT', api_role='BOT', generate=True),
        ]
    )

models = [
    dict(
        type=Mobius,
        # 以下参数为 `HuggingFaceCausalLM` 的初始化参数
        path='/mnt/lab/Models/pt/Mobius-r6-chat-CHNtuned-12b-16k-v4.2.1', 
        # 你的分词器所在的路径
        tokenizer_path='rwkv_vocab_v20230424', # <--- !! 修改这里 !!
        # tokenizer_path='/mnt/lab/q-llama-cpp-1/rwkv_vocab_v20250609.txt',
        max_seq_len=2048,
        meta_template=model_meta_template,
        # 以下参数为各类模型都有的参数，非 `HuggingFaceCausalLM` 的初始化参数
        abbr='xiaoke_12B',            # 模型简称，用于结果展示
        max_out_len=100,            # 最长生成 token 数
        batch_size=32,              # 批次大小
        run_cfg=dict(num_gpus=4),   # 运行配置，用于指定资源需求
        generation_kwargs=dict(
            do_sample=True,
            top_p=0.8,
            top_k=0,
            temperature=0.8,
            alpha_frequency=0.2,  # <-- 添加频率惩罚
            alpha_presence=0.2,   # <-- 添加存在惩罚
        ),
    ),
]
# for m in models:
#     m['max_seq_len'] = 131072

infer = dict(
    partitioner=dict(type=NumWorkerPartitioner, num_worker=32),
    runner=dict(
        type=LocalRunner,
        max_num_workers=32,
        max_workers_per_gpu=8,
        # gpus=[0, 1, 2, 3],
        task=dict(type=OpenICLInferTask)),
)

eval = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(
        type=LocalRunner,
        max_num_workers=64,
        task=dict(type=OpenICLEvalTask)),
)

work_dir = './outputs/xiaoke_12B'


