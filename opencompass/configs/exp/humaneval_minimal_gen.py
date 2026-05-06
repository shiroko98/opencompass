from opencompass.datasets import HumanevalDataset
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.models import HuggingFace

# ===== Prompt（极简，防解释）=====
humaneval_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[dict(
                role='HUMAN',
                prompt=(
                    "Read the following function signature and docstring, "
                    "and fully implement the function described.\n\n"
                    "{prompt}\n\n"
                    "Your response must contain ONLY the Python function code."
                )
            )]
        )
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer)
)

# ===== Dataset（只跑前 5 条）=====
datasets = [
    dict(
        type=HumanevalDataset,
        abbr='humaneval',
        reader_cfg=dict(
            input_columns=['prompt'],
            output_column='task_id',
            train_split='test',
            test_range=[0, 5],   # 👈 关键：只跑 5 条
        ),
        infer_cfg=humaneval_infer_cfg,
        eval_cfg=dict(
            evaluator=dict(type='HumanEvalEvaluator')
        )
    )
]

# ===== 模型（先用你已有的 HF wrapper）=====
models = [
    dict(
        type=HumanevalDataset,
        abbr='humaneval',
        data_path='/mnt/lab/lzy/opencompass/human-eval',  # 👈 新增这一行
        reader_cfg=dict(
            input_columns=['prompt'],
            output_column='task_id',
            train_split='test',
            test_range=[0, 5],
        ),
        infer_cfg=humaneval_infer_cfg,
        eval_cfg=dict(
            evaluator=dict(type='HumanEvalEvaluator')
        )
    )

]
