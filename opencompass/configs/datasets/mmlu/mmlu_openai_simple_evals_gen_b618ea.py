from mmengine.config import read_base
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import MMLUDataset
from opencompass.utils.text_postprocessors import match_answer_pattern

# ✅ 小子集：只跑一个
mmlu_all_sets = [
    "business_ethics",
    "moral_disputes",
    "high_school_psychology",
    "sociology",
]


QUERY_TEMPLATE = r"""
Answer the following multiple choice question.

You MUST respond with exactly ONE line in the following format:
ANSWER: X

Where X is one of ABCD.
Do NOT provide any explanation.
Do NOT output any other text before or after that line.

Question:
{input}

Options:
A) {A}
B) {B}
C) {C}
D) {D}
""".strip()

mmlu_reader_cfg = dict(
    input_columns=['input', 'A', 'B', 'C', 'D'],
    output_column='target',
    train_split='dev'
)

mmlu_datasets = []
for name in mmlu_all_sets:
    mmlu_infer_cfg = dict(
        prompt_template=dict(
            type=PromptTemplate,
            template=dict(
                round=[dict(role='HUMAN', prompt=QUERY_TEMPLATE)],
            ),
        ),
        retriever=dict(type=ZeroRetriever),
        inferencer=dict(
            type=GenInferencer,
            gen_kwargs=dict(
                max_new_tokens=16,
                temperature=0.0,
            ),
        ),
    )

    mmlu_eval_cfg = dict(
        evaluator=dict(type=AccEvaluator),
        pred_postprocessor=dict(
            type=match_answer_pattern,
            answer_pattern=r'(?i)\bANSWER\s*:\s*([A-D])\b'
        )
    )

    mmlu_datasets.append(
        dict(
            abbr=f'lukaemon_mmlu_{name}',
            type=MMLUDataset,
            path='opencompass/mmlu',
            name=name,
            reader_cfg=mmlu_reader_cfg,
            infer_cfg=mmlu_infer_cfg,
            eval_cfg=mmlu_eval_cfg,
        )
    )
