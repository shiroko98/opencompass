from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import HumanevalDataset, HumanEvalEvaluator, humaneval_postprocess_v2

humaneval_reader_cfg = dict(
    input_columns=['prompt'], output_column='task_id', train_split='test',test_range=(0, 10), 
    )

# TODO: allow empty output-column
humaneval_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(
                    role='HUMAN',
                    prompt=(
                        "You are a Python code generator.\n"
                        "Your task is to write the complete implementation of the function below.\n"
                        "\n"
                        "Rules:\n"
                        "1. Output ONLY valid Python code.\n"
                        "2. Do NOT include explanations, comments, or markdown.\n"
                        "3. Do NOT include text outside the function.\n"
                        "4. The output MUST start with `def`.\n"
                        "\n"
                        "{prompt}\n"
                    ),
                ),
            ]
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)


humaneval_eval_cfg = dict(
    evaluator=dict(type=HumanEvalEvaluator),
    pred_role='BOT',
    k=[1, 10, 100],  # the parameter only for humaneval
    pred_postprocessor=dict(type=humaneval_postprocess_v2),
)

humaneval_datasets = [
    dict(
        abbr='openai_humaneval',
        type=HumanevalDataset,
        path='opencompass/humaneval',
        reader_cfg=humaneval_reader_cfg,
        infer_cfg=humaneval_infer_cfg,
        eval_cfg=humaneval_eval_cfg)
]
