from mmengine.config import read_base

with read_base():
    from ..datasets.mmlu.mmlu_ppl import mmlu_datasets
    from ..models.mobius_xiaoke_12b import models

datasets = mmlu_datasets

work_dir = './outputs/mobius_xiaoke12b_mmlu_ppl'
