import tensorflow as tf
import numpy as np
from add_model import *

class AttRNN():

    def __init__(
            self,  num_classes,
            vocab_size,word_embedding_size,
            position_size, position_embedding_size,
            # filter_size, num_filters,
            l2_reg_lambda,
            trainable=False
    ):

        self.input_x = tf.placeholder(tf.int64, [None, None], name='input_x')
        self.input_p1 = tf.placeholder(tf.int64, [None, None], name='input_p1')
        self.input_p2 = tf.placeholder(tf.int64, [None, None], name='input_p2')

        self.input_y = tf.placeholder(tf.int64, name='input_y')

        self.learning_rate = tf.placeholder(tf.float32, name='learning_rate')
        self.keep_prob = tf.placeholder(tf.float32, name='learning_rate')

        self.hidden_size = word_embedding_size + 2 * position_embedding_size

        with tf.name_scope('word_embedding'):
            print('init embedding...')
            Embedding_X_zero = tf.Variable(
                tf.zeros([2, word_embedding_size]),
                name='embedding_X_zero',
                dtype=tf.float32,
                trainable=False
            )
            Embedding_X_normal = tf.get_variable(
                'embedding_X_normal',
                shape=[vocab_size-2, word_embedding_size],
                initializer=tf.contrib.layers.xavier_initializer(uniform=True),
                trainable=True
            )
            self.Embedding_X = tf.concat([Embedding_X_zero, Embedding_X_normal], axis=0)

        self.embedding_x = tf.nn.embedding_lookup(self.Embedding_X, self.input_x)

        with tf.name_scope('position_embedding'):
            Embedding_P_zero = tf.Variable(
                tf.zeros([1, position_embedding_size]),
                name='embedding_P_zero',
                dtype=tf.float32,
                trainable=False
            )
            Embedding_P_normal = tf.get_variable(
                'embedding_P_normal',
                shape=[position_size - 1, position_embedding_size],
                initializer=tf.contrib.layers.xavier_initializer(uniform=True)
            )
            self.Embedding_P = tf.concat([Embedding_P_zero, Embedding_P_normal], axis=0)

            self.embedding_p1 = tf.nn.embedding_lookup(self.Embedding_P, self.input_p1)
            self.embedding_p2 = tf.nn.embedding_lookup(self.Embedding_P, self.input_p2)

        with tf.name_scope('embedding_inputs'):
            self.inputs = tf.concat([self.embedding_x, self.embedding_p1, self.embedding_p2], 2)

        # add lstm
        with tf.name_scope('add'):
            # self.inputs=att_bilstm(self.inputs,self.hidden_size,is_addcnn=True,scope='lstm')
            # self.hidden_size = self.hidden_size * 2
            # ##1
            # w_att = tf.get_variable(
            #     'w_att',
            #     [self.hidden_size],
            #     initializer=tf.contrib.layers.xavier_initializer(uniform=False)
            # )
            # M = tf.tanh(self.inputs)
            # alpha = tf.nn.softmax(tf.einsum('aij,j->ai', M, w_att))
            # r = tf.einsum('aij,ai->aj', self.inputs, alpha)
            # self.h = tf.tanh(r)

            self.inputs = res_block(self.inputs, self.hidden_size, 1)
            # self.inputs = res_block(self.inputs, self.hidden_size, 1)
            # self.inputs = res_block(self.inputs, self.hidden_size, 1)

            outputs=self.inputs
            num_filters=500
            filter_size=3
            filter_size1 = 5  # 5
            sequence_length=160
            h = tf.layers.conv1d(
                inputs=outputs, filters=num_filters, kernel_size=filter_size, activation=None, use_bias=True,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                bias_initializer=tf.constant_initializer(0.01)
            )
            h1 = tf.layers.conv1d(
                inputs=outputs, filters=num_filters, kernel_size=filter_size1, activation=None, use_bias=True,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                bias_initializer=tf.constant_initializer(0.01)
            )
            pooled = tf.layers.max_pooling1d(h, sequence_length - filter_size + 1, 1)
            pooled1 = tf.layers.max_pooling1d(h1, sequence_length - filter_size1 + 1, 1)

            pooled_flatten = tf.tanh(tf.reshape(pooled, [-1, num_filters]))
            pooled_flatten1 = tf.tanh(tf.reshape(pooled1, [-1, num_filters]))

            self.pooled_flatten = tf.concat([pooled_flatten, pooled_flatten1], axis=-1)

            # classification
            self.Embedding_C = tf.get_variable(
                'embedding_class',
                shape=[2 * num_filters, num_classes],
                initializer=tf.contrib.layers.xavier_initializer(uniform=True)
            )
            self.bias = tf.Variable(tf.constant(0.1, shape=[num_classes]), name='bias')

        with tf.name_scope('score_prediction'):
            self.scores = tf.matmul(self.pooled_flatten, self.Embedding_C) + self.bias  # batch_size * num_classes
            self.predictions = tf.argmax(self.scores, axis=1)



        # with tf.name_scope('BiLSTM'):
        #     def rnn_cell(hidden_size, name):
        #         return tf.nn.rnn_cell.LSTMCell(
        #             hidden_size,
        #             initializer=tf.contrib.layers.xavier_initializer(uniform=False),
        #             name=name)
        #     outputs, states = tf.nn.bidirectional_dynamic_rnn(
        #         rnn_cell(self.hidden_size,name='fw'),
        #         rnn_cell(self.hidden_size, name='bw'),
        #         inputs=self.inputs,
        #         # sequence_length=160,
        #         dtype=tf.float32
        #     )
        #     # (c_state_fw, h_state_fw), (c_state_bw, h_state_bw) = states
        #     # self.lstm_states = h_state_fw + h_state_bw
        #     o_fw, o_bw = outputs
        #
        # with tf.name_scope('attention'):
        #     w_att = tf.get_variable(
        #         'w_att',
        #         [self.hidden_size],
        #         initializer=tf.contrib.layers.xavier_initializer(uniform=False)
        #     )
        #     H = o_fw + o_bw
        #     M = tf.tanh(H)
        #     alpha = tf.nn.softmax(tf.einsum('aij,j->ai', M, w_att))
        #     r = tf.einsum('aij,ai->aj', H, alpha)
        #
        #     self.h = tf.tanh(r)  # batch_size * vec_size (self.hidden_size)
        #
        # with tf.name_scope('class_embedding'):
        #     self.Embedding_C = tf.get_variable(
        #         'embedding_class',
        #         shape=[self.hidden_size, num_classes],
        #         initializer=tf.contrib.layers.xavier_initializer(uniform=True)
        #     )
        #     self.bias = tf.Variable(tf.constant(0.01, shape=[num_classes]), name='bias')
        #
        # with tf.name_scope('score_prediction'):
        #     self.scores = tf.matmul(self.h, self.Embedding_C) + self.bias  # batch_size * num_classes
        #     self.predictions = tf.argmax(self.scores, axis=1)

        with tf.name_scope('loss'):
            losses = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.input_y, logits=self.scores)
            )

            l2_reg_loss = tf.constant(0.)
            for var in tf.trainable_variables():
                l2_reg_loss += tf.nn.l2_loss(var)

            self.loss = losses + l2_reg_lambda * l2_reg_loss

        with tf.name_scope('accuracy'):
            correct_predictions = tf.equal(self.predictions, self.input_y)
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, 'float'), name='accuracy')

        with tf.name_scope('optimize'):
            # self.global_step = tf.Variable(0, name='global_step', trainable=False)
            # optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)

            # optimizer=tf.train.AdadeltaOptimizer(self.learning_rate)
            # grads_over_vars = optimizer.compute_gradients(self.loss)
            # self.train_op = optimizer.apply_gradients(grads_over_vars, global_step=self.global_step)
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
            self.train_op = optimizer.minimize(self.loss)

        with tf.name_scope('summary'):
            loss_summary = tf.summary.scalar('loss', self.loss)
            acc_summary = tf.summary.scalar('accuracy', self.accuracy)
            self.train_summary_op = tf.summary.merge([loss_summary, acc_summary])
            self.dev_summary_op = acc_summary
