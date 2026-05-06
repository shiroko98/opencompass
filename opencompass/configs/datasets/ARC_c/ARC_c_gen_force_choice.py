from mmengine.config import read_base

with read_base():
    from .ARC_c_gen_1e0de5 import ARC_c_datasets

# 把每条样本的 prompt 最后改成强约束输出字母
for ds in ARC_c_datasets:
    # 只对 gen 类型的 reader/prompt 做增强
    if 'reader_cfg' in ds and 'prompt_template' in ds['reader_cfg']:
        pt = ds['reader_cfg']['prompt_template']
        if isinstance(pt, dict) and 'prompt' in pt:
            pt['prompt'] = pt['prompt'].replace(
                'Answer:',
                'Answer (output only one letter among A/B/C/D):'
            )
