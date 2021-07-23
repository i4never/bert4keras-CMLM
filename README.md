# bert4keras-CMLM
- [CMLM](https://arxiv.org/pdf/2012.14388.pdf)的非官方实现。
- 参考&依赖 https://github.com/bojone/bert4keras/blob/master/pretraining/pretraining.py 
- saved模型与0.9.9.版本的`bert4keras.models.BERT`兼容

inputs:
  - token_ids: `[batch_size, 2*N+1, max_len] 其中仅第 N 行包含mask`
  - segment_ids: `[batch_size, 2*N+1, max_len]`
  - target_ids: `[batch_size, max_len]`
  - is_masked: `[batch_size, max_len]`

outputs:
  - loss: `[batch_size, ]`
  - acc: `[batch_size, ]`

outputs实际没有用到，保证第一个维度为batch_size即可
