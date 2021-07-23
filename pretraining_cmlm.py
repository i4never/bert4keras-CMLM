"""
CMLM 方式预训练bert
<UNIVERSAL SENTENCE REPRESENTATION LEARNING WITH CONDITIONAL MASKED LANGUAGE MODEL>
https://arxiv.org/pdf/2012.14388.pdf
"""
from bert4keras.layers import *
from bert4keras.models import BERT

config = {'n': 2,
          'vocab_size': 21128,
          'segment_vocab_size': 2,
          'max_position': 512,
          'coordinate_size': 1000,
          'dropout_rate': 0.1,
          'hidden_size': 768,
          'attention_head_n': 12,
          'attention_block_n': 12,
          'intermediate_size': 3072,
          'hidden_act': 'gelu'
          }


# -------- Custom Layers --------
def search_layer(inputs, name, exclude_from=None):
    """根据inputs和name来搜索层
    说明：inputs为某个层或某个层的输出；name为目标层的名字。
    实现：根据inputs一直往上递归搜索，直到发现名字为name的层为止；
         如果找不到，那就返回None。
    """
    if exclude_from is None:
        exclude_from = set()

    if isinstance(inputs, keras.layers.Layer):
        layer = inputs
    else:
        layer = inputs._keras_history[0]

    if layer.name == name:
        return layer
    elif layer in exclude_from:
        return None
    else:
        exclude_from.add(layer)
        if isinstance(layer, keras.models.Model):
            model = layer
            for layer in model.layers:
                if layer.name == name:
                    return layer
        inbound_layers = layer._inbound_nodes[0].inbound_layers
        if not isinstance(inbound_layers, list):
            inbound_layers = [inbound_layers]
        if len(inbound_layers) > 0:
            for layer in inbound_layers:
                layer = search_layer(layer, name, exclude_from)
                if layer is not None:
                    return layer


class CMLMInputSplit(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(CMLMInputSplit, self).__init__(**kwargs)
        self.supports_masking = False

    def call(self, inputs, mask=None):
        # inputs [bs, 2n+1, ml]
        n = (K.int_shape(inputs)[1] - 1) // 2

        outputs = [tf.concat([inputs[:, :n, :][:, i, :] for i in range(n)], axis=0),
                   inputs[:, n, :],
                   tf.concat([inputs[:, n + 1:, :][:, i, :] for i in range(n)], axis=0)]
        return outputs

    def compute_output_shape(self, input_shape):
        bs, n, ml = input_shape[:3]
        return [(None, ml), (bs, ml), (None, ml)]


class CMLMConcatenate(keras.layers.Concatenate):
    """用来在Embedding后的向量开头位置加入上下文[2N, D]的Embedding
    """

    def compute_mask(self, inputs, mask=None):
        mask_ = K.ones_like(mask[2][:, :config['n']], dtype='bool')
        return K.concatenate([mask_, mask_, mask[2]], axis=self.axis)


class EmbeddingDense(Layer):
    """运算跟Dense一致，但kernel用Embedding层的embeddings矩阵。
    根据Embedding层的名字来搜索定位Embedding层。
    用于输入输出共享Embedding的模型。
    """

    def __init__(
            self, embedding_name, activation='softmax', use_bias=True, **kwargs
    ):
        super(EmbeddingDense, self).__init__(**kwargs)
        self.embedding_name = embedding_name
        self.activation = activations.get(activation)
        self.use_bias = use_bias

    def call(self, inputs):
        if not hasattr(self, 'kernel'):
            embedding_layer = search_layer(inputs, self.embedding_name)
            if embedding_layer is None:
                raise Exception('Embedding layer not found')

            self.kernel = K.transpose(embedding_layer.embeddings)
            self.units = K.int_shape(self.kernel)[1]
            if self.use_bias:
                self.bias = self.add_weight(
                    name='bias', shape=(self.units,), initializer='zeros'
                )

        outputs = K.dot(inputs, self.kernel)
        if self.use_bias:
            outputs = K.bias_add(outputs, self.bias)
        outputs = self.activation(outputs)
        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (self.units,)

    def get_config(self):
        config = {
            'embedding_name': self.embedding_name,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
        }
        base_config = super(EmbeddingDense, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


# -------- Custom Layers End --------

# -------- Layers Build --------
layers = list()
# Embedding
layers += [Embedding(input_dim=config['vocab_size'],
                     output_dim=config['hidden_size'],
                     embeddings_initializer=keras.initializers.TruncatedNormal(stddev=0.02),
                     mask_zero=True,
                     name='Embedding-Token'),
           Embedding(input_dim=config['segment_vocab_size'],
                     output_dim=config['hidden_size'],
                     embeddings_initializer=keras.initializers.TruncatedNormal(stddev=0.02),
                     mask_zero=True,
                     name='Embedding-Segment'),
           Add(name='Embedding-Token-Segment'),
           PositionEmbedding(input_dim=config['max_position'],
                             output_dim=config['hidden_size'],
                             merge_mode='add',
                             embeddings_initializer=keras.initializers.TruncatedNormal(stddev=0.02),
                             name='Embedding-Position'),
           LayerNormalization(hidden_initializer=keras.initializers.TruncatedNormal(stddev=0.02),
                              name='Embedding-Norm'),
           Dropout(rate=config['dropout_rate'], name='Embedding-Dropout')
           ]
# Transformer
for i in range(config['attention_block_n']):
    attention_name = f"Transformer-{i}-MultiHeadSelfAttention"
    feedforward_name = f"Transformer-{i}-FeedForward"
    layers.append(MultiHeadAttention(head_size=config['hidden_size'] // config['attention_head_n'],
                                     heads=config['attention_head_n'],
                                     out_dim=config['hidden_size'],
                                     key_size=config['hidden_size'] // config['attention_head_n'],
                                     kernel_initializer=keras.initializers.TruncatedNormal(stddev=0.02),
                                     name=attention_name))
    layers.append(Dropout(rate=config['dropout_rate'], name=f"{attention_name}-Dropout"))
    layers.append(Add(name=f"{attention_name}-Add"))
    layers.append(LayerNormalization(hidden_initializer=keras.initializers.TruncatedNormal(stddev=0.02),
                                     name=f"{attention_name}-Norm"))
    layers.append(FeedForward(units=config['intermediate_size'],
                              activation=config['hidden_act'],
                              name=f"{feedforward_name}"))
    layers.append(Dropout(rate=config['dropout_rate'], name=f"{feedforward_name}-Dropout"))
    layers.append(Add(name=f"{feedforward_name}-Add"))
    layers.append(LayerNormalization(hidden_initializer=keras.initializers.TruncatedNormal(stddev=0.02),
                                     name=f"{feedforward_name}-Norm"))

# Pooler
layers.append(GlobalAveragePooling1D(name='Context-Mean-Pooler'))

# Input split & Concat
layers.append(CMLMInputSplit(name='Input-Split'))
layers.append(CMLMConcatenate(axis=1, name='Context-Concat'))
layers.append(
    Lambda(lambda x: tf.reshape(x, shape=(-1, config['n'], config['hidden_size'])), mask=None, name='Context-Reshape'))

layer_dict = {l.name: l for l in layers}


# -------- Layers Build End --------

# -------- Build Model --------
def embeder(inputs):
    token_in, seg_in = inputs

    # Embedding
    token_embedding = layer_dict['Embedding-Token'](token_in)
    seg_embedding = layer_dict['Embedding-Segment'](seg_in)

    x = layer_dict['Embedding-Token-Segment']([token_embedding, seg_embedding])
    x = layer_dict['Embedding-Position'](x)
    x = layer_dict['Embedding-Norm'](x)
    x = layer_dict['Embedding-Dropout'](x)
    return x


def attention_blocks(input_):
    x = input_

    # att --> add --> ln --> ffn --> add --> ln
    for i in range(config['attention_block_n']):
        attention_name = f"Transformer-{i}-MultiHeadSelfAttention"
        feedforward_name = f"Transformer-{i}-FeedForward"

        # att
        att_input = x
        x = layer_dict[attention_name]([x, x, x])
        x = layer_dict[f"{attention_name}-Dropout"](x)

        # add
        x = layer_dict[f"{attention_name}-Add"]([att_input, x])

        # ln
        x = layer_dict[f"{attention_name}-Norm"](x)
        # ffn
        ffn_input = x
        x = layer_dict[f"{feedforward_name}"](x)

        x = layer_dict[f"{feedforward_name}-Dropout"](x)
        x = layer_dict[f"{feedforward_name}-Add"]([ffn_input, x])
        x = layer_dict[f"{feedforward_name}-Norm"](x)
    return x


def cls_pooler(input_):
    return layer_dict['Context-Pooler'](input_)


def mean_pooler(input_):
    return layer_dict['Context-Reshape'](layer_dict['Context-Mean-Pooler'](input_))


# [batch_size, 2*N+1, max_len]
token_in = Input(shape=(2 * config['n'] + 1, None), name='Input-Token')
segment_in = Input(shape=(2 * config['n'] + 1, None), name='Input-Segment')

# [batch_size, N, max_len], [batch_size, max_len], [batch_size, N, max_len]
token_in_s = layer_dict['Input-Split'](token_in)
segment_in_s = layer_dict['Input-Split'](segment_in)

# [batch_size, N, hidden_size]
before = mean_pooler(attention_blocks(embeder([token_in_s[0], segment_in_s[0]])))
after = mean_pooler(attention_blocks(embeder([token_in_s[2], segment_in_s[2]])))

# [batch_size, max_len, hidden_size]
emeded_line = embeder([token_in_s[1], segment_in_s[1]])

# [batch_size, max_len+2*N, hidden_size]
output = attention_blocks(layer_dict['Context-Concat']([before, after, emeded_line]))

# model = keras.models.Model([token_in, segment_in], output)

# mlm部分
# [batch_size, max_len]
target_ids = Input(shape=(None,), dtype='int64', name='target_ids')  # 目标id
is_masked = Input(shape=(None,), dtype=K.floatx(), name='is_masked')  # mask标记


def cmlm_loss(inputs):
    """计算loss的函数，需要封装为一个层
    """
    y_true, y_pred, mask = inputs

    # y_pred减去2N的context偏移
    y_pred = y_pred[:, 2 * config['n']:, :]

    loss = K.sparse_categorical_crossentropy(
        y_true, y_pred, from_logits=False
    )
    loss = K.sum(loss * mask) / (K.sum(mask) + K.epsilon())
    return loss


def cmlm_acc(inputs):
    """计算准确率的函数，需要封装为一个层
    """
    y_true, y_pred, mask = inputs

    # y_pred减去2N的context偏移
    y_pred = y_pred[:, 2 * config['n']:, :]

    y_true = K.cast(y_true, K.floatx())
    acc = keras.metrics.sparse_categorical_accuracy(y_true, y_pred)
    acc = K.sum(acc * mask) / (K.sum(mask) + K.epsilon())
    return acc


# [batch_size, max_len+2*N, vocab_size]
proba = EmbeddingDense(embedding_name='Embedding-Token', use_bias=False, activation='softmax')(output)

mlm_loss = Lambda(cmlm_loss, name='cmlm_loss')([target_ids, proba, is_masked])
mlm_acc = Lambda(cmlm_acc, name='cmlm_acc')([target_ids, proba, is_masked])

model = keras.models.Model([token_in, segment_in] + [target_ids, is_masked], [mlm_loss, mlm_acc])

model.summary()

# -------- Build Model End --------

# -------- Train --------
model.compile(
    loss={'cmlm_loss': lambda y_true, y_pred: y_pred, 'cmlm_acc': lambda y_true, y_pred: K.stop_gradient(y_pred)},
    optimizer=keras.optimizers.Adam(1e-4))

# # Train
# # Context 偏移量已在 loss/acc 计算中考虑，参考 https://github.com/bojone/bert4keras/blob/master/pretraining/pretraining.py
# # 中的数据生成即可
# # inputs:
# #     - token_ids: [batch_size, 2*N+1, max_len] 其中仅第 N 行包含mask
# #     - segment_ids: [batch_size, 2*N+1, max_len]
# #     - target_ids: [batch_size, max_len]
# #     - is_masked: [batch_size, max_len]
# # outputs:
# #     - loss: [batch_size, ]
# #     - acc: [batch_size, ]
# model.fit(xxx, callbacks=[keras.callbacks.TensorBoard(log_dir=f"./tb_logs", update_freq='batch')])

# # 兼容 bert4keras.models.BERT
# model.save('./test_cmlm_save.h5')
# bert = BERT(max_position=512, vocab_size=config['vocab_size'], hidden_size=config['hidden_size'],
#             num_hidden_layers=config['attention_block_n'], num_attention_heads=config['attention_head_n'],
#             intermediate_size=config['intermediate_size'], hidden_act=config['gelu'])
# bert.model.load('./test_cmlm_save.h5')
