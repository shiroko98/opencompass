from opencompass.models import HuggingFace

# =====================================================
# Qwen3-32B HuggingFace (base) 模型配置
# 专门用于：PPL / MMLU / MMLU-Pro
# =====================================================

models = [
    dict(
        type=HuggingFace,
        abbr='qwen3-32b-hf',
        path='/mnt/lab/Models/qwen/Qwen3-32B',

        # ⭐ 关键：PPL 必须关闭 chat_template
        use_chat_template=False,

        model_kwargs=dict(
            device_map='auto',
            trust_remote_code=True,
        ),

        tokenizer_kwargs=dict(
            trust_remote_code=True,
            use_fast=False,
        ),

        # PPL 实际不生成文本，这里只是占位
        max_out_len=64,

        # batch_size 可按显存调
        batch_size=4,

        run_cfg=dict(
            num_gpus=1,
        ),
    )
]
