# coding:utf-8

import tensorflow as tf
import numpy as np


def att_bilstm(inputs,  hidden_size, is_addcnn=False,scope=None):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        def _rnn_cell(num_units, name):
            return tf.nn.rnn_cell.LSTMCell(
                num_units,
                initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                name=name
            )

        outputs, _ = tf.nn.bidirectional_dynamic_rnn(
            _rnn_cell(hidden_size, name=f'{scope}_fw'),
            _rnn_cell(hidden_size, name=f'{scope}_bw'),
            inputs=inputs,
            # sequence_length=seq_length,
            dtype=tf.float32
        )
        o_fw, o_bw = outputs
        H = o_fw + o_bw # sub2_10
        # H = tf.concat([o_fw, o_bw], axis=-ann1)
        if is_addcnn:
            H1 = inception(inputs=H, num_filters=hidden_size, kernel_size1=1, kernel_size2=1,scope='H1')
            H2 = inception(inputs=H, num_filters=hidden_size, kernel_size1=1, kernel_size2=4,scope='H2')
            H3 = inception(inputs=H, num_filters=hidden_size, kernel_size1=1,is_onelayers=True,scope='H3')
            H4 = inception(inputs=H, num_filters=hidden_size, kernel_size1=1, kernel_size2=7, scope='H4')
            H = tf.concat([H, H1], axis=-1)
        # with tf.name_scope('attention'):
        #     w_att = tf.get_variable(
        #         'w_att',
        #         [hidden_size],
        #         initializer=tf.contrib.layers.xavier_initializer(uniform=False)
        #     )
        #
        #     M = tf.tanh(H)
        #     alpha = tf.nn.softmax(tf.einsum('aij,j->ai', M, w_att))
        #     r = tf.einsum('aij,ai->aij', H, alpha)

        # with tf.name_scope('forward'):
        #     W = tf.get_variable("W", shape=[hidden_size, n_tags], dtype=tf.float32,
        #                         initializer=tf.contrib.layers.xavier_initializer())
        #
        #     b = tf.get_variable("b", shape=[n_tags], dtype=tf.float32,
        #                         initializer=tf.constant_initializer(0.01))
        #
        #     context_flat = tf.reshape(lstm, [-ann1, hidden_size])
        #
        #     ntime_steps = tf.shape(lstm)[ann1]
        #     pred = tf.matmul(context_flat, W) + b
        #     scores = tf.reshape(pred, [-ann1, ntime_steps, n_tags])

        h = tf.tanh(H)  # batch_size *N* vec_size (hidden_size)
    return h


def idCNN(inputs, filter_width, embedding_dim, num_filter, num_steps, num_tags, is_train, name=None):
    """
   shape of input = [batch, in_height, in_width, in_channels]
   shape of filter = [filter_height, filter_width, in_channels, out_channels]
   """
    inputs = tf.expand_dims(inputs, 1)

    with tf.variable_scope("idcnn" if not name else name):
        filter_weights = tf.get_variable("idcnn_filter",
                                         shape=[1, filter_width, embedding_dim, num_filter],
                                         initializer=tf.contrib.layers.xavier_initializer())
        layer_input = tf.nn.conv2d(inputs,
                                   filter_weights,
                                   strides=[1, 1, 1, 1],
                                   padding="SAME",
                                   name="init_layer"
                                   )
        final_out_list = []
        total_width_dim = 0
        repeat_times = 3
        layers = [1, 2, 2]

        for j in range(repeat_times):
            for i in range(len(layers)):
                dilation = layers[i]
                isLast = True if i == (len(layers) - 1) else False
                with tf.variable_scope("atrous-conv-layer-%d" % i, reuse=tf.AUTO_REUSE):
                    w = tf.get_variable("filterW",
                                        shape=[1, filter_width, num_filter, num_filter],
                                        initializer=tf.contrib.layers.xavier_initializer())
                    b = tf.get_variable("filterB", shape=[num_filter])
                    conv = tf.nn.atrous_conv2d(layer_input,
                                               w,
                                               rate=dilation,
                                               padding="SAME")
                    conv = tf.nn.bias_add(conv, b)
                    conv = tf.nn.relu(conv)
                    if isLast:
                        final_out_list.append(conv)
                        total_width_dim += num_filter
                    layer_input = conv

        final_out = tf.concat(axis=3, values=final_out_list)
        # keep_prob = 0.5 if is_train else ann1.0
        # final_out = tf.nn.dropout(final_out, keep_prob)
        final_out = tf.cond(
            is_train,
            lambda: tf.nn.dropout(final_out, keep_prob=0.5),
            lambda: final_out)

        final_out = tf.squeeze(final_out, [1])
        final_out = tf.reshape(final_out, [-1, total_width_dim])

        with tf.variable_scope("project"):
            # project to score of tags
            W = tf.get_variable("W", shape=[total_width_dim, num_tags],
                                dtype=tf.float32,
                                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.get_variable("b", initializer=tf.constant(0.001, shape=[num_tags]))
            pred = tf.nn.xw_plus_b(final_out, W, b)

    return tf.reshape(pred, [-1, num_steps, num_tags])


def positional_encoding(T,
                        num_units,
                        zero_pad=True,
                        scale=True,
                        scope="positional_encoding",
                        reuse=tf.AUTO_REUSE):
    with tf.variable_scope(scope, reuse=reuse):
        # broadcast
        position_ind = tf.range(T)

        # First part of the PE function: sin and cos argument
        position_enc = np.array([
            [pos / np.power(10000, 2. * i / num_units) for i in range(num_units)]
            for pos in range(T)])

        # Second part, apply the cosine to even columns and sin to odds.
        position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i
        position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # dim 2i+ann1

        # Convert to a tensor
        lookup_table = tf.convert_to_tensor(position_enc, dtype=tf.float32)

        if zero_pad:
            lookup_table = tf.concat((tf.zeros(shape=[1, num_units]),
                                      lookup_table[1:, :]), 0)
        outputs = tf.nn.embedding_lookup(lookup_table, position_ind)

        if scale:
            outputs = outputs * num_units ** 0.5

        return outputs  # (T,num_units)


def multihead_attention(queries,
                        keys,
                        num_units=512,
                        num_heads=8,
                        dropout_rate=0,
                        is_training=True,
                        causality=False,
                        scope="multihead_attention",
                        reuse=tf.AUTO_REUSE):
    with tf.variable_scope(scope, reuse=reuse):
        # Set the fall back option for num_units
        if num_units is None:
            num_units = queries.get_shape().as_list[-1]

        # Linear projections
        Q = tf.layers.dense(queries, num_units, activation=tf.nn.relu, use_bias=True,
                            kernel_initializer=tf.contrib.layers.xavier_initializer(),
                            bias_initializer=tf.constant_initializer(0.01)
                            )  # (N, T_q, C)
        Q = tf.tanh(Q) * tf.sigmoid(Q)
        K = tf.layers.dense(keys, num_units, activation=tf.nn.relu, use_bias=True,
                            kernel_initializer=tf.contrib.layers.xavier_initializer(),
                            bias_initializer=tf.constant_initializer(0.01)
                            )  # (N, T_k, C)
        K = tf.tanh(K) * tf.sigmoid(K)
        V = tf.layers.dense(keys, num_units, activation=tf.nn.relu, use_bias=True,
                            kernel_initializer=tf.contrib.layers.xavier_initializer(),
                            bias_initializer=tf.constant_initializer(0.01)
                            )  # (N, T_k, C)
        V = tf.tanh(V) * tf.sigmoid(V)
        # Split and concat
        Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)  # (h*N, T_q, C/h)
        K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)
        V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)

        # Multiplication
        outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))  # (h*N, T_q, T_k)

        # Scale
        outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)

        # # Key Masking
        key_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1)))  # (N, T_k)
        key_masks = tf.tile(key_masks, [num_heads, 1])  # (h*N, T_k)
        key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1])  # (h*N, T_q, T_k)

        paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
        outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs)  # (h*N, T_q, T_k)

        # Causality = Future blinding
        if causality:
            diag_vals = tf.ones_like(outputs[0, :, :])  # (T_q, T_k)
            tril = tf.contrib.linalg.LinearOperatorTriL(diag_vals).to_dense()  # (T_q, T_k)
            masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(outputs)[0], 1, 1])  # (h*N, T_q, T_k)

            paddings = tf.ones_like(masks) * (-2 ** 32 + 1)
            outputs = tf.where(tf.equal(masks, 0), paddings, outputs)  # (h*N, T_q, T_k)

        # Activation
        outputs = tf.nn.softmax(outputs)  # (h*N, T_q, T_k)

        # Query Masking
        query_masks = tf.sign(tf.abs(tf.reduce_sum(queries, axis=-1)))  # (N, T_q)
        query_masks = tf.tile(query_masks, [num_heads, 1])  # (h*N, T_q)
        query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]])  # (h*N, T_q, T_k)
        outputs *= query_masks  # broadcasting. (N, T_q, C)

        # Dropouts
        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))

        # Weighted sum
        outputs = tf.matmul(outputs, V_)  # ( h*N, T_q, C/h)

        # Restore shape
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)  # (N, T_q, C)

        # Residual connection
        outputs += queries

        # Normalize
        # outputs = normalize(outputs)  # (N, T_q, C)

    return outputs


def multi_attention(queries,
                    dmodel=512,
                    num_heads=8,
                    scope="multi_attention",
                    reuse=tf.AUTO_REUSE):
    with tf.variable_scope(scope, reuse=reuse):
        # Set the fall back option for num_units
        num_units = dmodel / num_heads

        def liner_projection(Q, K, V):
            # Linear projections
            Q = tf.layers.dense(Q, num_units, activation=tf.nn.relu)  # (N, T_q, C) c=num_units
            K = tf.layers.dense(K, num_units, activation=tf.nn.relu)  # (N, T_k, C)
            V = tf.layers.dense(V, num_units, activation=tf.nn.relu)  # (N, T_k, C)
            outputs = tf.matmul(Q, tf.transpose(K, [0, 2, 1]))
            outputs = outputs / (K.get_shape().as_list()[-1] ** 0.5)
            outputs = tf.nn.softmax(outputs)  # (N, T_q, T_k)
            outputs = tf.matmul(outputs, V)  # ( N, T_q, C)
            return outputs

        headi = liner_projection(queries, queries, queries)
        for i in range(num_heads - 1):
            headi = tf.concat([headi, liner_projection(queries, queries, queries)], axis=-1)  # (N, T_k, h*C)
        multi_head = tf.layers.dense(headi, dmodel, activation=tf.nn.relu)
        multi_head += queries

        # Normalize
        multi_head = normalize(multi_head)  # (N, T_q, C)

    return multi_head


def inception(inputs,
              num_filters=None,
              kernel_size1=1,
              kernel_size2=1,
              is_onelayers=False,
              is_resnet=True,
              scope="inception",
              reuse=tf.AUTO_REUSE):
    with tf.variable_scope(scope, reuse=reuse):
        outputs = tf.layers.conv1d(
            inputs=inputs, filters=num_filters, kernel_size=kernel_size1, activation=tf.nn.relu, use_bias=True,
            kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
            bias_initializer=tf.constant_initializer(0.01),
            padding='same'
        )

        if not is_onelayers:
            outputs = tf.layers.conv1d(
                inputs=outputs, filters=num_filters, kernel_size=kernel_size2, activation=None, use_bias=True,
                kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                bias_initializer=tf.constant_initializer(0.01),
                padding='same'
            )

        if is_resnet:
            outputs += inputs  # Residual connection
    return outputs


def feedforward(inputs,
                num_filters=None,
                kernel_size=1,
                scope="multihead_attention",
                reuse=tf.AUTO_REUSE):
    with tf.variable_scope(scope, reuse=reuse):
        outputs = tf.layers.conv1d(
            inputs=inputs, filters=num_filters, kernel_size=kernel_size, activation=tf.nn.relu, use_bias=True,
            kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
            bias_initializer=tf.constant_initializer(0.01),
            padding='same'
        )

        # outputs = tf.tanh(outputs) * tf.sigmoid(outputs)
        # tf.nn.softplus

        outputs = tf.layers.conv1d(
            inputs=outputs, filters=num_filters, kernel_size=kernel_size, activation=None, use_bias=True,
            kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
            bias_initializer=tf.constant_initializer(0.01),
            padding='same'
        )
        # outputs = tf.tanh(outputs) * tf.sigmoid(outputs)
        outputs += inputs  # Residual connection

    return outputs


def res_block(inputs, hidden_size, kernel_size=1):
    outputs = feedforward(inputs=inputs, num_filters=hidden_size, kernel_size=kernel_size)
    outputs = feedforward(inputs=outputs, num_filters=hidden_size, kernel_size=kernel_size)
    outputs = feedforward(inputs=outputs, num_filters=hidden_size, kernel_size=kernel_size)
    outputs = feedforward(inputs=outputs, num_filters=hidden_size, kernel_size=kernel_size)
    outputs = feedforward(inputs=outputs, num_filters=hidden_size, kernel_size=kernel_size)
    outputs = feedforward(inputs=outputs, num_filters=hidden_size, kernel_size=kernel_size)
    return outputs

def block(inputs, hidden_size, kernel_size=1):
    selfattention = SelfAttention(hidden_size=hidden_size, num_heads=2, attention_dropout=0, train=True)
    inputs = selfattention(inputs=inputs, bias=0.01)
    inputs = feedforward(inputs=inputs, num_filters=hidden_size, kernel_size=kernel_size)
    # inputs = selfattention(inputs=inputs, bias=0.01)
    # inputs = feedforward(inputs=inputs, num_filters=hidden_size, kernel_size=kernel_size)
    # inputs = selfattention(inputs=inputs, bias=0.01)
    # inputs = feedforward(inputs=inputs, num_filters=hidden_size, kernel_size=kernel_size)
    return inputs

def normalize(inputs,
              epsilon=1e-8,
              scope="ln",
              reuse=tf.AUTO_REUSE):
    '''Applies layer normalization.

    Args:
      inputs: A tensor with 2 or more dimensions, where the first dimension has
        `batch_size`.
      epsilon: A floating number. A very small number for preventing ZeroDivision Error.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns:
      A tensor with the same shape and data dtype as `inputs`.
    '''
    with tf.variable_scope(scope, reuse=reuse):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]

        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        beta = tf.Variable(tf.zeros(params_shape))
        gamma = tf.Variable(tf.ones(params_shape))
        normalized = (inputs - mean) / ((variance + epsilon) ** (.5))
        outputs = gamma * normalized + beta

    return outputs

class Attention(tf.layers.Layer):
  """Multi-headed attention layer."""

  def __init__(self, hidden_size, num_heads, attention_dropout, train):
    if hidden_size % num_heads != 0:
      raise ValueError("Hidden size must be evenly divisible by the number of "
                       "heads.")

    super(Attention, self).__init__()
    self.hidden_size = hidden_size
    self.num_heads = num_heads
    self.attention_dropout = attention_dropout
    self.train = train

    # Layers for linearly projecting the queries, keys, and values.
    self.q_dense_layer = tf.layers.Dense(hidden_size, use_bias=False, name="q")
    self.k_dense_layer = tf.layers.Dense(hidden_size, use_bias=False, name="k")
    self.v_dense_layer = tf.layers.Dense(hidden_size, use_bias=False, name="v")

    self.output_dense_layer = tf.layers.Dense(hidden_size, use_bias=False,
                                              name="output_transform")

  def split_heads(self, x):
    """Split x into different heads, and transpose the resulting value.

    The tensor is transposed to insure the inner dimensions hold the correct
    values during the matrix multiplication.

    Args:
      x: A tensor with shape [batch_size, length, hidden_size]

    Returns:
      A tensor with shape [batch_size, num_heads, length, hidden_size/num_heads]
    """
    with tf.name_scope("split_heads"):
      batch_size = tf.shape(x)[0]
      length = tf.shape(x)[1]

      # Calculate depth of last dimension after it has been split.
      depth = (self.hidden_size // self.num_heads)

      # Split the last dimension
      x = tf.reshape(x, [batch_size, length, self.num_heads, depth])

      # Transpose the result
      return tf.transpose(x, [0, 2, 1, 3])

  def combine_heads(self, x):
    """Combine tensor that has been split.

    Args:
      x: A tensor [batch_size, num_heads, length, hidden_size/num_heads]

    Returns:
      A tensor with shape [batch_size, length, hidden_size]
    """
    with tf.name_scope("combine_heads"):
      batch_size = tf.shape(x)[0]
      length = tf.shape(x)[2]
      x = tf.transpose(x, [0, 2, 1, 3])  # --> [batch, length, num_heads, depth]
      return tf.reshape(x, [batch_size, length, self.hidden_size])

  def call(self, x, y, bias, cache=None):
    """Apply attention mechanism to x and y.

    Args:
      x: a tensor with shape [batch_size, length_x, hidden_size]
      y: a tensor with shape [batch_size, length_y, hidden_size]
      bias: attention bias that will be added to the result of the dot product.
      cache: (Used during prediction) dictionary with tensors containing results
        of previous attentions. The dictionary must have the items:
            {"k": tensor with shape [batch_size, i, key_channels],
             "v": tensor with shape [batch_size, i, value_channels]}
        where i is the current decoded length.

    Returns:
      Attention layer output with shape [batch_size, length_x, hidden_size]
    """
    # Linearly project the query (q), key (k) and value (v) using different
    # learned projections. This is in preparation of splitting them into
    # multiple heads. Multi-head attention uses multiple queries, keys, and
    # values rather than regular attention (which uses a single q, k, v).
    q = self.q_dense_layer(x)
    k = self.k_dense_layer(y)
    v = self.v_dense_layer(y)

    if cache is not None:
      # Combine cached keys and values with new keys and values.
      k = tf.concat([cache["k"], k], axis=1)
      v = tf.concat([cache["v"], v], axis=1)

      # Update cache
      cache["k"] = k
      cache["v"] = v

    # Split q, k, v into heads.
    q = self.split_heads(q)
    k = self.split_heads(k)
    v = self.split_heads(v)

    # Scale q to prevent the dot product between q and k from growing too large.
    depth = (self.hidden_size // self.num_heads)
    q *= depth ** -0.5

    # Calculate dot product attention
    logits = tf.matmul(q, k, transpose_b=True)
    logits += bias
    weights = tf.nn.softmax(logits, name="attention_weights")
    if self.train:
      weights = tf.nn.dropout(weights, 1.0 - self.attention_dropout)
    attention_output = tf.matmul(weights, v)

    # Recombine heads --> [batch_size, length, hidden_size]
    attention_output = self.combine_heads(attention_output)

    # Run the combined outputs through another linear projection layer.
    attention_output = self.output_dense_layer(attention_output)
    return attention_output


class SelfAttention(Attention):
  """Multiheaded self-attention layer."""

  def call(self, x, bias, cache=None):
    return super(SelfAttention, self).call(x, x, bias, cache)