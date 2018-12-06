# coding:utf-8

import tensorflow as tf
import os
import math
from model import LSTMCRF
from load_data import DataLoader
from to_ann import save_ann
import numpy as np
import time

# model2 hyperparameters
tf.flags.DEFINE_integer('vec_size', 160, 'vec_size (default: 150)')
tf.flags.DEFINE_integer('hidden_size', 160, 'hidden_size (default: 150)')
tf.flags.DEFINE_float('l2_reg_lambda', 1.0e-8, 'l2 regularization lambda (default: 0.0)')
tf.flags.DEFINE_float('learning_rate', 0.005, 'learning rate (default: 1e-4)')

# training parameters
tf.flags.DEFINE_bool('is_train', True, 'is_train (default: True)')
tf.flags.DEFINE_integer('batch_size', 22, 'batch size (default: 64)')
tf.flags.DEFINE_integer('num_epochs', 9, 'number of training epochs (default: 200)')
tf.flags.DEFINE_integer('show_step', 1, 'number of show result (default: 10)')

flags = tf.flags.FLAGS

num = time.strftime("%Y-%m-%d %a %H_%M_%S", time.localtime())
# num = '3'

with tf.Graph().as_default():
    # test path
    test_path = r'../data/mid_test/mid_ann'+num
    if not os.path.exists(test_path):
        os.makedirs(test_path)
    save_path = os.path.join(r'../data/model_save', 'check_points'+num, 'model')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    submit_path = r'../submit/'+num
    if not os.path.exists(submit_path):
        os.makedirs(submit_path)
        
    # prepare data set
    train_dir = r'../data/ruijin_round1_train_20181022\ruijin_round1_train2_20181022'
    testb_dir = r'..\data\ruijin_round1_test_b_20181112'
    dataloader = DataLoader(train_dir, testb_dir)
    dataset = dataloader.dataset
    testdataset = dataloader.test_dataset

    train_data, test_data = [], []
    for i, (words, tags, length) in enumerate(zip(*dataset)):
        if i % 1001 == 0:
            test_data.append((words, tags, length))
        else:
            train_data.append((words, tags, length))

    # hyper
    vec_size = flags.vec_size
    hidden_size = flags.hidden_size
    l2_reg_lambda = flags.l2_reg_lambda
    learning_rate = flags.learning_rate

    is_train = flags.is_train
    batch_size = flags.batch_size
    num_epochs = flags.num_epochs
    num_batches = math.ceil(len(train_data) / batch_size)

    # 16 5 0.005 150 150 1e-6
    # sub2_2 22 8 0.005 150 150 1e-7
    # sub2-3 22 8 0.005 150 150 1e-7 bf_w*2
    # sub2-4 22 9 0.005 150 150 1e-8 *2
    # sub2-5 22 9 0.005 145 145 1e-7 *2 2rnncnn 2hidden
    # sub2-6 20 11 0.005 151 151 1e-7 *ann1 gate
    # sub2-7 22 12 0.005 155 155 1e-7 *ann1 gate
    # sub2-8 20 10 0.005 148 148 1e-8 *ann1 gate sig*than
    # sub2-9 20 13 0.005 148 148 1e-8 *ann1 gate sig*than H1_h3
    # sub2-10 22 9 0.005 150 150 1e-8 *ann1  H1_h3 0.7731

    # sub3_2 22 9 0.005 150 150 1e-8 *2 2*att_lstm H1
    # sub3_3 22 10 0.005 200 200 80 1e-7 *2 gate sig*than a*2 H1
    # sub3_4 22 10 0.005 200 200 150 1e-8 *2  a*2 H1_h3 sgd
    # sub3_5 22 9 0.005 160 160 150 1e-8 *2  a*2 H1_h3 2fnc sgd
    # sub3_6 22 8 0.005 160 160 150 1e-8 *ann1  a*ann1 H1_h3 sgd
    # sub3_7 22 8 0.005 160 160 150 1e-8 *ann1  a*ann1 H1_h3 amd
    # sub3_8 16 8 0.005 120 160 150 1e-8 *ann1  a*ann1 H1_h4 amd
    # sub3_9 20 8 0.008 80 160 150 1e-8 *ann1  a*ann1 H1_h4 amd
    # sub3_10 21 10 0.004 130 200 200 1e-8 *ann1  a*ann1 H1_h4 amd

    # test_b
    # sub1 22 9 0.005 150 150 1e-8 *ann1  H1_h3 amd

    model = LSTMCRF(
        max_seq_length=dataloader.pad_max,
        vocab_size=dataloader.vocab_size,
        vec_size=vec_size,
        hidden_size=hidden_size,
        l2_reg_lambda=l2_reg_lambda,
        n_tags=dataloader.tag_size + 1
        # trainable=True
    )


    def train_input_queue(data, batch_size=32):
        x, y, l = [], [], []
        for i in data:
            x.append(i[0])
            y.append(i[1])
            l.append(i[-1])
        input_queue = tf.train.slice_input_producer([x, y, l], shuffle=True)
        x_queue, y_queue, l_queue = tf.train.batch(input_queue, batch_size=batch_size, capacity=64)
        return [x_queue, y_queue, l_queue]


    def dev_input_queue(data, batch_size=32):
        x, y, l = [], [], []
        for i in data:
            x.append(i[0])
            y.append(i[1])
            l.append(i[-1])
        input_queue = tf.train.slice_input_producer([x, y, l], shuffle=True)
        x_queue, y_queue, l_queue = tf.train.batch(input_queue, batch_size=batch_size, capacity=64)
        return [x_queue, y_queue, l_queue]


    def train_step(data_list, learning_rate):
        train_batch_x, train_batch_y, train_batch_l = sess.run(data_list)
        run_op = [model.train_op, model.loss, model.accuracy, model.input_y, model.predictions]

        feed_dict = {
            model.input_x: train_batch_x,
            model.input_y: train_batch_y,
            model.seq_length: train_batch_l,
            model.learning_rate: learning_rate,
            model.is_train: 1,
        }
        _, cost, acc, y, p = sess.run(run_op, feed_dict=feed_dict)

        return cost, acc


    def dev_step(test_data):

        test_batch_x, test_batch_y, test_batch_l = sess.run(test_data)

        test_feed_dict = {
            model.input_x: test_batch_x,
            model.input_y: test_batch_y,
            model.seq_length: test_batch_l,
            model.is_train: 0,
        }
        run_op = [model.predictions, model.accuracy]
        predictions, acc = sess.run(run_op, feed_dict=test_feed_dict)

        for x, pred, seq_len in zip(test_batch_x, predictions, test_batch_l):
            for w, t in zip(x[:seq_len], pred[:seq_len]):
                if w > 0 and t > 0:
                    print(f'{dataloader.words_set[int(w)-1]}/{dataloader.tags_set[int(t)-1]} ', end='')
                else:
                    print(w, t, end='')
            print()
        return acc


    def t_step(test_data, test_path):
        for test_batch_x, test_batch_l, tid in zip(test_data[0], test_data[1], dataloader.text_ids):
            test_batch_x, test_batch_l = np.asarray(test_batch_x), np.asarray(test_batch_l)
            print(test_batch_x.shape, test_batch_l.shape)
            with open(os.path.join(test_path, tid.replace('.txt', '.ann')), 'w', encoding='utf-8') as f:
                test_feed_dict = {
                    model.input_x: test_batch_x,
                    model.seq_length: test_batch_l,
                    model.is_train: 0,
                }
                run_op = [model.predictions]
                predictions = sess.run(run_op, feed_dict=test_feed_dict)
                predictions = np.asarray(predictions)[0]

                assert len(test_batch_x) == len(test_batch_l) and len(test_batch_x) == len(predictions)
                for x, pred, seq_len in zip(test_batch_x, predictions, test_batch_l):
                    assert len(x[:seq_len]) == len(pred[:seq_len])
                    for w in x[:seq_len]:
                        if w > 0:
                            # print(f'{dataloader.words_set[int(w)-ann1]}',end='')
                            f.write(f'{dataloader.words_set[int(w)-1]}')
                    # print()
                    f.write('\n')
                    for t in pred[:seq_len]:
                        if t > 0:
                            # print(f'{dataloader.tags_set[int(t)-ann1]} ',end='')
                            f.write(f'{dataloader.tags_set[int(t)-1]} ')
                    # print()
                    f.write('\n')


    saver = tf.train.Saver()
    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())
        # prepare trainset testset
        # data_list include [x_queue, y_queue, p1_queue, p2_queue]
        train_data_list = train_input_queue(train_data, batch_size)  # thread data set
        test_data_list = dev_input_queue(test_data, batch_size)

        # threads queue main
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        max_dev_acc = 0
        if is_train:
            for epoch in range(1, num_epochs):
                for step in range(num_batches):
                    print('train start...')
                    cost, acc = train_step(train_data_list, learning_rate)
                    if step % 1 == 0:
                        print('epoch {}, step {}, cost {}, acc {}'.format(epoch, (epoch - 1) * num_batches + step, cost,
                                                                          acc))

                learning_rate = (learning_rate / float(epoch)) if learning_rate > 0.00001 else 0.00001
                for i in range(int(len(test_data) / batch_size)):
                    dev_acc = dev_step(test_data_list)
                    print('\ntest acc: {}, max  {}'.format(dev_acc, max(max_dev_acc, dev_acc)))
                    if dev_acc > max_dev_acc:  # 记录最好的
                        max_dev_acc = dev_acc

            print("Model saved in path: %s" % save_path)
            save_model = saver.save(sess, save_path)
            print("test...")
            t_step(testdataset, test_path)
        else:
            saver.restore(sess, save_path)
            t_step(testdataset, test_path)
        # 关闭线程队列
        coord.request_stop()
        coord.join(threads)

    save_ann(test_path, submit_path)
