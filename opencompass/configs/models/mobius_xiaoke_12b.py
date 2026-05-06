from opencompass.models.mobius import Mobius

models = [
    dict(
        type=Mobius,
        abbr='mobius_xiaoke_12b',
        path='/mnt/lab/Models/pt/Mobius-r6-chat-CHNtuned-12b-16k-v4.2.1',
        tokenizer_path='rwkv_vocab_v20230424',
        max_seq_len=2048,
        max_out_len=100,
        batch_size=1,   # ppl 推荐 1
        num_gpus=1,
        meta_template=dict(
            round=[
                dict(role='HUMAN', begin='User: ', end='\n\n'),
                dict(role='BOT', begin='Assistant:', end='\n\n', generate=True),
            ],
        ),
        stop_words=['\n\nUser:', '\n\nQ:', '\n\nQuestion:'],
        generation_kwargs=dict(
            temperature=1.0,
            top_p=0.2,
            top_k=0,
            alpha_frequency=0.4,
            alpha_presence=0.4,
        ),
    )
]
datasets = []
