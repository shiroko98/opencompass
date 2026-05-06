from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import FixKRetriever
from opencompass.openicl.icl_inferencer import PPLInferencer
from opencompass.openicl.icl_evaluator import AccwithDetailsEvaluator
from opencompass.datasets import MMLUDataset

# None of the mmlu dataset in huggingface is correctly parsed, so we use our own dataset reader
# Please download the dataset from https://people.eecs.berkeley.edu/~hendrycks/data.tar

mmlu_reader_cfg = dict(
    input_columns=['input', 'A', 'B', 'C', 'D'],
    output_column='target',
    train_split='dev')

mmlu_all_sets = [
    'business_ethics',
]


mmlu_datasets = []
for _name in mmlu_all_sets:
    _hint = f'The following are multiple choice questions (with answers) about  {_name.replace("_", " ")}.\n\n'
    question_overall = '{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}'
    mmlu_infer_cfg = dict(
        ice_template=dict(
            type=PromptTemplate,
            template={opt: f'{question_overall}\nAnswer: {opt}\n' for opt in ['A', 'B', 'C', 'D']},
        ),
        prompt_template=dict(
            type=PromptTemplate,
            template={opt: f'{_hint}</E>{question_overall}\nAnswer: {opt}' for opt in ['A', 'B', 'C', 'D']},
            ice_token='</E>',
        ),
        retriever=dict(type=FixKRetriever, fix_id_list=[0, 1, 2, 3, 4]),
        inferencer=dict(type=PPLInferencer),
    )

    mmlu_eval_cfg = dict(evaluator=dict(type=AccwithDetailsEvaluator), )

    mmlu_datasets.append(
        dict(
            abbr=f'lukaemon_mmlu_{_name}',
            type=MMLUDataset,
            path='opencompass/mmlu',
            name=_name,
            reader_cfg=mmlu_reader_cfg,
            infer_cfg=mmlu_infer_cfg,
            eval_cfg=mmlu_eval_cfg,
            test_split='test[:20]',
        ))

del _name, _hint
