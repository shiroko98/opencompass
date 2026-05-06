from mmengine.config import read_base
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import MMLUProDataset
from opencompass.utils.text_postprocessors import match_answer_pattern

with read_base():
    from .mmlu_pro_categories import categories

SMOKE_TEST_CATEGORIES = [
    'math',          # 最常用，推荐
    # 'physics',
    # 'chemistry',
]

QUERY_TEMPLATE = r"""
You MUST answer in the first line.

Final answer format:
ANSWER: <LETTER>

Only output that line and STOP.

ANSWER:

Question:
{question}

Options:
{options_str}
""".strip()



mmlu_pro_datasets = []

for category in SMOKE_TEST_CATEGORIES:
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
                    dict(
                        role='HUMAN',
                        prompt=QUERY_TEMPLATE
                    ),
                ],
            ),
        ),
        retriever=dict(type=ZeroRetriever),
        inferencer=dict(
            type=GenInferencer,
            max_out_len=32,
            temperature=0.0,
            top_p=1.0,
            stop_words=[
                '\n',          # 一行就停
                'Options:',     # 防止重复
            ],
        ),
    )


    mmlu_pro_eval_cfg = dict(
        evaluator=dict(type=AccEvaluator),
        pred_postprocessor=dict(
            type=match_answer_pattern,
            #answer_pattern=r'(?im)^\s*ANSWER\s*:\s*([A-P])\s*$'
            answer_pattern=r'([A-P])'
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
        ))
