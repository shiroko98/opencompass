mmlu_pro_ppl_datasets = [
    dict(
        type='openicl',
        abbr='mmlu_pro_abstract_algebra_ppl',
        path='lukaemon/mmlu_pro',
        name='abstract_algebra',
        reader_cfg=dict(
            input_columns=['question', 'choices'],
            output_column='answer',
        ),
        infer_cfg=dict(
            inferencer=dict(type='PPLInferencer'),
        ),
        eval_cfg=dict(
            evaluator=dict(type='AccEvaluator'),
        ),
    )
]
