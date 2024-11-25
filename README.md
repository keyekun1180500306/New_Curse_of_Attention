Hello Chiwun and Zhao，
- The newly modified code (Nov 25) is in:  
- (1) ReverseSoftmaxAttention: /PatchTST_supervised/layers/SelfAttention_Family.py  
- (2) RoPEAttention: /PatchTST_supervised/layers/SelfAttention_Family.py  
- (3) Sigmoid Norm: just to replace from norm_layer=torch.nn.LayerNorm(configs.d_model)  
to norm_layer= torch.nn.Sigmoid()
