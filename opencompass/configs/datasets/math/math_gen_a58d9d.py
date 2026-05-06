from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import MATHDataset
from opencompass.evaluator import MATHVerifyEvaluator

# =========================
# Reader
# =========================
math_reader_cfg = dict(
    input_columns=['problem'],
    output_column='solution'
)

# =========================
# Inference（✅ 修复版）
# =========================
math_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(
                    role='HUMAN',
                    prompt=(
                        "Problem:\n{problem}\n\n"
                        "Solve the problem step by step privately.\n"
                        "At the very end, output ONLY the final answer on a single line.\n"
                        "Do NOT add any text before or after the final answer.\n\n"
                        "Final Answer:\n"
                    )
                )
            ]
        )
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(
        type=GenInferencer,
        generation_kwargs=dict(
            temperature=0.0,
            do_sample=False,
            max_new_tokens=256
        )
    )
)

# =========================
# Evaluation（保持 verify）
# =========================
math_eval_cfg = dict(
    evaluator=dict(type=MATHVerifyEvaluator)
)

# =========================
# Dataset
# =========================
math_datasets = [
    dict(
        type=MATHDataset,
        abbr='math',
        path='opencompass/math',
        reader_cfg=math_reader_cfg,
        infer_cfg=math_infer_cfg,
        eval_cfg=math_eval_cfg,
        max_eval_samples=5,  # 调试阶段可以先改成 5
    )
]
