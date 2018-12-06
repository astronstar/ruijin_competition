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

        h = tf.tanh(H)  # batch_size *N* vec_size (hidden_size)
    return h


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
    return outputs

