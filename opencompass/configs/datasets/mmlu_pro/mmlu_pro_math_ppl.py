from mmengine.config import read_base
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import PPLInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import MMLUProDataset

categories = ["math"]

QUERY_TEMPLATE = r"""
Answer the following multiple choice question.

Question:
{question}

Options:
{options_str}

Answer:
{option}
""".strip()

mmlu_pro_datasets = []

for category in categories:
    mmlu_pro_reader_cfg = dict(
        input_columns=["question", "options_str"],
        output_column="answer",
        train_split="validation",
        test_split="test",
    )

    mmlu_pro_infer_cfg = dict(
        prompt_template=dict(
            type=PromptTemplate,
            template=dict(
                round=[
                    dict(role="HUMAN", prompt=QUERY_TEMPLATE),
                ],
            ),
        ),
        retriever=dict(type=ZeroRetriever),
        inferencer=dict(
            type=PPLInferencer,
            options=["A", "B", "C", "D"],  # ⭐⭐⭐ 核心修复
        ),
    )


    mmlu_pro_eval_cfg = dict(
        evaluator=dict(type=AccEvaluator),
    )

    mmlu_pro_datasets.append(
        dict(
            abbr=f"mmlu_pro_{category}",
            type=MMLUProDataset,
            path="opencompass/mmlu_pro",
            category=category,
            reader_cfg=mmlu_pro_reader_cfg,
            infer_cfg=mmlu_pro_infer_cfg,
            eval_cfg=mmlu_pro_eval_cfg,
        )
    )
