# coding:utf-8
import tensorflow as tf
from attention import *


class LSTMCRF():
    def __init__(self, max_seq_length, vocab_size, vec_size, hidden_size, n_tags, l2_reg_lambda):
        self.input_x = tf.placeholder(dtype=tf.int32, shape=[None, None], name='input_x')
        self.input_y = tf.placeholder(dtype=tf.int32, shape=[None, None], name='input_y')
        self.seq_length = tf.placeholder(dtype=tf.int32, shape=[None], name='seq_length')

        self.learning_rate = tf.placeholder(tf.float32, name='learning_rate')
        self.keep_prob = 1.0  # tf.placeholder(tf.float32, name='keep_prob')
        self.is_train = tf.placeholder(tf.bool, name='is_train')

        with tf.name_scope('word_embedding'):
            print('init embedding...')
            Embedding_zero = tf.Variable(
                tf.zeros([1, vec_size]),
                name='embedding_zero',
                dtype=tf.float32,
                trainable=False
            )
            Embedding_normal = tf.get_variable(
                'embedding_normal',
                shape=[vocab_size, vec_size],
                initializer=tf.contrib.layers.xavier_initializer(uniform=True),
                trainable=True
            )
            self.Embedding = tf.concat([Embedding_zero, Embedding_normal], axis=0)
            self.embedding_x = tf.nn.embedding_lookup(self.Embedding, self.input_x)

        with tf.name_scope('multihead_att'):
            # self.embedding_x = tf.cond(
            #     self.is_train,
            #     lambda: tf.nn.dropout(self.embedding_x, keep_prob=self.keep_prob),
            #     lambda: self.embedding_x)

            self.embedding_x = att_bilstm(self.embedding_x, self.seq_length, hidden_size, scope='bilstm1')
            # self.embedding_x = att_bilstm(self.embedding_x, self.seq_length, hidden_size, scope='bilstm2')
            # self.scores = idCNN(inputs=self.embedding_x,
            #                     filter_width=5,
            #                     embedding_dim=hidden_size,
            #                     num_filter=hidden_size,
            #                     num_steps=max_seq_length,
            #                     num_tags=n_tags,
            #                     is_train=self.is_train)
        with tf.name_scope('BiLSTM'):
            def rnn_cell(hidden_size, name):
                return tf.nn.rnn_cell.LSTMCell(
                    hidden_size,
                    initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                    name=name)

            rnn_outputs, states = tf.nn.bidirectional_dynamic_rnn(
                rnn_cell(hidden_size, name='fw'),
                rnn_cell(hidden_size, name='bw'),
                inputs=self.embedding_x,
                sequence_length=self.seq_length,
                dtype=tf.float32
            )
            o_fw, o_bw = rnn_outputs

            # self.lstm = o_fw + o_bw
            self.lstm = tf.concat([o_fw, o_bw], axis=-1)

        # with tf.name_scope("Gate"):
        #     beta1 = tf.layers.dense(self.lstm, hidden_size*2, activation=tf.nn.sigmoid, use_bias=True,
        #                             kernel_initializer=tf.contrib.layers.xavier_initializer(),
        #                             bias_initializer=tf.constant_initializer(0.01)
        #                             )
        #     beta2 = tf.layers.dense(self.lstm, hidden_size*2, activation=tf.nn.sigmoid, use_bias=True,
        #                             kernel_initializer=tf.contrib.layers.xavier_initializer(),
        #                             bias_initializer=tf.constant_initializer(0.01)
        #                             )
        #     m = tf.layers.dense(self.lstm, hidden_size*2, activation=tf.nn.tanh, use_bias=True,
        #                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
        #                         bias_initializer=tf.constant_initializer(0.01)
        #                         )
            # self.lstm = beta1 * self.lstm + beta2 * m
            # self.lstm = beta1 * m

        with tf.name_scope('forward'):
            W = tf.get_variable("W", shape=[hidden_size * 2, n_tags], dtype=tf.float32,
                                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.get_variable("b", shape=[n_tags], dtype=tf.float32,
                                initializer=tf.constant_initializer(0.01))

            context_flat = tf.reshape(self.lstm, [-1, hidden_size * 2])
            ntime_steps = tf.shape(self.lstm)[1]

            # filter_layers_dense = tf.layers.dense(
            #     context_flat, hidden_size * 4, activation=tf.nn.relu, use_bias=False,
            #     kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
            # )
            # pred = tf.layers.dense(filter_layers_dense, n_tags, use_bias=True,
            #                        kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
            #                        bias_initializer=tf.constant_initializer(0.01))
            pred = tf.matmul(context_flat, W) + b
            self.scores = tf.reshape(pred, [-1, ntime_steps, n_tags])

        # with tf.name_scope('prediction'):
        #     self.predictions = tf.argmax(self.scores, axis=-ann1)
        #     losses = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        #         labels=self.input_y,
        #         logits=self.scores))
        #     l2_reg_loss = tf.constant(0.)
        #     for var in tf.trainable_variables():
        #         l2_reg_loss += tf.nn.l2_loss(var)
        #
        #     self.loss = losses + l2_reg_lambda * l2_reg_loss

        with tf.name_scope('CRF'):
            log_likelihood, self.transition_params = tf.contrib.crf.crf_log_likelihood(
                self.scores, self.input_y, self.seq_length)

            # 进行解码（维特比算法），获得解码之后的序列viterbi_sequence和分数viterbi_score
            self.predictions, viterbi_score = tf.contrib.crf.crf_decode(
                self.scores, self.transition_params, self.seq_length)

            l2_reg_loss = tf.constant(0.)
            for var in tf.trainable_variables():
                l2_reg_loss += tf.nn.l2_loss(var)

            self.loss = tf.reduce_mean(-log_likelihood) + l2_reg_lambda * l2_reg_loss
        #
        with tf.name_scope('train'):
            # grad_clip = ann1.250
            # tvars = tf.trainable_variables()
            # grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), grad_clip)
            #
            # optimizer = tf.train.AdamOptimizer(self.learning_rate)
            # self.train_op = optimizer.apply_gradients(zip(grads, tvars))

            optimizer = tf.train.AdamOptimizer(self.learning_rate)
            self.train_op = optimizer.minimize(self.loss)

        with tf.name_scope('pred'):
            # mask=tf.sequence_mask(self.seq_length,maxlen=max_seq_length,name='seq_length_mask')
            equal_num = tf.cast(tf.equal(self.input_y, self.predictions), tf.float32)
            self.accuracy = 100.0 * tf.reduce_mean(equal_num)
