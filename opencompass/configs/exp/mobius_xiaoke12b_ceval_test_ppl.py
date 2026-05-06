from mmengine.config import read_base

with read_base():
    from ..datasets.ceval.ceval_test_ppl_mobius import ceval_test_ppl_datasets
    from ..models.mobius_xiaoke_12b import models as base_models

datasets = ceval_test_ppl_datasets

models = []
for model in base_models:
    model = model.copy()
    model.pop('num_gpus', None)
    model['batch_size'] = 1
    model['run_cfg'] = dict(num_gpus=1)
    models.append(model)

work_dir = './outputs/mobius_xiaoke12b_ceval_test_ppl'
