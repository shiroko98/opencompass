from opencompass.configs.datasets.mmlu_pro.mmlu_pro_ppl import mmlu_pro_ppl_datasets
from opencompass.runners import LocalRunner
from opencompass.partitioners import NaivePartitioner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask
from opencompass.models import HuggingFaceCausalLM
# from opencompass.models.vllm import VLLM

# ✅ 顶层 datasets：必须有
datasets = mmlu_pro_ppl_datasets

models = [
    dict(
        type=HuggingFaceCausalLM,
        abbr='qwen3_32b',
        path='/mnt/lab/Models/qwen/Qwen3-32B',
        tokenizer_path='/mnt/lab/Models/qwen/Qwen3-32B',
        max_out_len=1,
        batch_size=1,
        run_cfg=dict(num_gpus=1),
    ),
]

infer_tasks = [
    dict(
        type=OpenICLInferTask,
        models=models,
        partitioner=dict(type=NaivePartitioner),
        runner=dict(type=LocalRunner),
    )
]

eval_tasks = [
    dict(
        type=OpenICLEvalTask,
        models=models,
        runner=dict(type=LocalRunner),
    )
]
