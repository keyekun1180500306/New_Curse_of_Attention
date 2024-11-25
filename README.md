Hello Chiwun and teacher:
    The newly added code is in:
    (1) RoPEAttention: \PatchTST_supervised\layers\SelfAttention_Family.py
    (2) ReverseSoftmaxAttention: \PatchTST_supervised\layers\SelfAttention_Family.py
    (3) Sigmoid Norm: this only need to replace
    line 59 in \PatchTST_supervised\models\Transformer from
    norm_layer=torch.nn.LayerNorm(configs.d_model)
    to
    norm_layer=torch.nn.Sigmoid()
   
