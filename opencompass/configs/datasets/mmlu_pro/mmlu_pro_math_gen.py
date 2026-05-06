from mmengine.config import read_base
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import MMLUProDataset
from opencompass.utils.text_postprocessors import match_answer_pattern

# ✅ 只跑 math 子集
categories = ["math"]

QUERY_TEMPLATE = r"""
Answer the following multiple choice question.

You MUST respond with exactly ONE line in the following format:
ANSWER: X

Where X is one of ABCDEFGHIJKLMNOP.
Do NOT provide any explanation.
Do NOT output any other text before or after that line.

Question:
{question}

Options:
{options_str}
""".strip()

mmlu_pro_datasets = []

for category in categories:
    mmlu_pro_reader_cfg = dict(
        input_columns=['question', 'cot_content', 'options_str'],
        output_column='answer',
        train_split='validation',
        test_split='test',
    )

    mmlu_pro_infer_cfg = dict(
        prompt_template=dict(
            type=PromptTemplate,
            template=dict(
                round=[
                    dict(role='SYSTEM',
                         prompt='You are a multiple-choice answer bot. Output only one line: "ANSWER: X".'),
                    dict(role='HUMAN', prompt=QUERY_TEMPLATE),
                ],
            ),
        ),
        retriever=dict(type=ZeroRetriever),
        inferencer=dict(type=GenInferencer),
    )

    mmlu_pro_eval_cfg = dict(
        evaluator=dict(type=AccEvaluator),
        pred_postprocessor=dict(
            type=match_answer_pattern,
            answer_pattern=r'(?im)^\s*ANSWER\s*:\s*([A-P])\s*$'
        )
    )

    mmlu_pro_datasets.append(
        dict(
            abbr=f'mmlu_pro_{category.replace(" ", "_")}',
            type=MMLUProDataset,
            path='opencompass/mmlu_pro',
            category=category,
            reader_cfg=mmlu_pro_reader_cfg,
            infer_cfg=mmlu_pro_infer_cfg,
            eval_cfg=mmlu_pro_eval_cfg,
        )
    )
