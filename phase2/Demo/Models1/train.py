# coding:utf-8

import tensorflow as tf
import os
import math
from models import AttRNN
from d import DataLoader
import numpy as np
import random
import time
# random.seed(1024)

num = '500'
in_path=r'../DataSets/ruijin_round2_test_a/ruijin_round2_test_a'
# in_path=r'../DataSets/ruijin_round2_test_b/ruijin_round2_test_b'
out_path=r'../DataSets/sub/submit_'+num
# out_path=r'../DataSets/sub/submit_cmd1'
if not os.path.exists(out_path):
        os.makedirs(out_path)

with tf.Graph().as_default():
    # test path
    # test_path = r'../data/mid_test/mid_ann' + num
    # if not os.path.exists(test_path):
    #     os.makedirs(test_path)
    save_path = os.path.join(r'../DataSets/model_save', 'check_points_' + num, 'model')
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # prepare data set
    dataloader = DataLoader()
    # dataset = dataloader.train_dataset
    dataset = dataloader.train_set_min1
    # dataset = dataloader.train_set_min2
    # dataset = dataloader.train_set_min3
    # dataset =dataloader.train_set_min4
    # test_dataset = dataloader.test_dataset
    test_dataset = dataloader.test_set_min1
    # test_dataset = dataloader.get_testb_instance()

    train_data, dev_data = [], []
    for i,instance in enumerate(dataset):
        # train_id_list, train_p1_list, train_p2_list, train_y_list=instance
        # if i%100==0:
        #     dev_data.append(instance)
        # elif i%10==0:
        #     train_data.append(instance)
        if random.random()>0.9:
            dev_data.append(instance)
        else:
            train_data.append(instance)
    print(f'train size: {len(train_data)}, dev size:{len(dev_data)}')

    # hyper
    # word_embedding_size = flags.vec_size
    # hidden_size = flags.hidden_size
    l2_reg_lambda = 1e-3
    learning_rate = 0.001   # 0.001

    is_train = True
    batch_size = 20 # 30
    num_epochs = 4
    num_batches = math.ceil(len(train_data) / batch_size)

    model = AttRNN(
        num_classes=len(dataloader.relation_types),
        vocab_size=dataloader.vocab_size,
        word_embedding_size=200,
        position_size=2 * dataloader.position_max + 2,
        position_embedding_size=50,
        l2_reg_lambda=l2_reg_lambda,
        is_train=is_train
    )


    def train_input_queue(data, batch_size=32):
        id_list, p1_list, p2_list, y_list=[],[],[],[]
        for instance in data:
            # print(instance)
            instance=instance[1:]
            id_list.append(instance[0])
            p1_list.append(instance[1])
            p2_list.append(instance[2])
            y_list.append(instance[3])
        input_queue = tf.train.slice_input_producer([id_list, p1_list, p2_list, y_list], shuffle=True)
        id_queue, p1_queue, p2_queue,y_queue = tf.train.batch(input_queue, batch_size=batch_size, capacity=64)
        return [id_queue, p1_queue, p2_queue,y_queue]


    def dev_input_queue(data, batch_size=32):
        id_list, p1_list, p2_list, y_list = [], [], [], []
        for instance in data:
            instance = instance[1:]
            id_list.append(instance[0])
            p1_list.append(instance[1])
            p2_list.append(instance[2])
            y_list.append(instance[3])

        input_queue = tf.train.slice_input_producer([id_list, p1_list, p2_list, y_list], shuffle=True)
        id_queue, p1_queue, p2_queue, y_queue = tf.train.batch(input_queue, batch_size=batch_size, capacity=64)
        return [id_queue, p1_queue, p2_queue, y_queue]


    def train_step(data_list, learning_rate):
        train_batch_id, train_batch_p1,train_batch_p2,train_batch_y = sess.run(data_list)
        # print(train_batch_id)
        run_op = [model.train_op, model.loss, model.accuracy, model.input_y, model.predictions]

        feed_dict = {
            model.input_x: train_batch_id,
            model.input_p1:train_batch_p1,
            model.input_p2:train_batch_p2,
            model.input_y: train_batch_y,
            model.learning_rate: learning_rate,
            # model.is_train: 1,
        }
        _, cost, acc, y, p = sess.run(run_op, feed_dict=feed_dict)

        return cost, acc


    def dev_step(dev_data):
        batch_id, batch_p1, batch_p2, batch_y = sess.run(dev_data)
        test_feed_dict = {
            model.input_x: batch_id,
            model.input_p1: batch_p1,
            model.input_p2: batch_p2,
            model.input_y: batch_y,
            # model.is_train: 0,
        }
        run_op = [model.predictions, model.accuracy]
        predictions, acc = sess.run(run_op, feed_dict=test_feed_dict)

        return acc


    def t_step(test_data, in_path,out_path):
        for (txtid, instances) in test_data:
            try:
                tag, ids, p1, p2, y = zip(*instances)
                tag, ids, p1, p2, y=list(tag),list(ids),list(p1),list(p2),list(y),
            except:
                continue
        # for (txtid, tag, ids, p1, p2, y) in zip(*test_data):
            print(txtid)
            print('batch_size: ',len(tag))
            test_feed_dict = {
                model.input_x: ids,
                model.input_p1: p1,
                model.input_p2: p2,
                # model.input_y: y,
            }
            run_op = [model.predictions]
            predictions = sess.run(run_op, feed_dict=test_feed_dict)
            # print(predictions)
            predictions = np.asarray(predictions)[0]

            pre_tag = [dataloader.relation_types[i] for i in predictions]
            with open(os.path.join(in_path, txtid.replace('.txt', '.ann')), 'r', encoding='utf-8') as fin:
                data = fin.readlines()
            with open(os.path.join(out_path, txtid.replace('.txt', '.ann')), 'w', encoding='utf-8') as fout:
                for line in data:
                    line = line.strip().strip('\n')
                    if line.startswith('T'):
                        fout.write(f'{line}\n')
                count=1
                for (t,pre) in zip(tag,pre_tag):
                    if pre!='Other':
                        # print(pre)
                        fout.write(f'R{count}\t{t}\n')
                        count+=1

    saver = tf.train.Saver()
    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())
        # prepare trainset testset
        # data_list include [x_queue, y_queue, p1_queue, p2_queue]
        train_data_list = train_input_queue(train_data, batch_size)  # thread data set
        dev_data_list = dev_input_queue(dev_data, batch_size=20)

        # threads queue main
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        max_dev_acc = 0
        if is_train:
            for epoch in range(1, num_epochs+1):
                for step in range(num_batches):
                    print('train start...')
                    cost, acc = train_step(train_data_list, learning_rate)
                    if step % 1 == 0:
                        print('epoch {}, step {}, cost {}, acc {}'.format(epoch, (epoch - 1) * num_batches + step, cost,
                                                                          acc))

                learning_rate = (learning_rate / float(epoch)) if learning_rate > 0.00001 else 0.00001
                print('\ndev...')
                pre=0
                times=math.ceil(len(dev_data)/20)
                for i in range(times):
                    dev_acc = dev_step(dev_data_list)
                    pre+=dev_acc
                dev_acc=pre/times
                print('\ndev acc: {}, max  {}'.format(dev_acc, max(max_dev_acc, dev_acc)))
                if dev_acc > max_dev_acc:  # 记录最好的
                    max_dev_acc = dev_acc
            print("Model saved in path: %s" % save_path)
            save_model = saver.save(sess, save_path)

            print("test...")
            t_step(test_dataset, in_path,out_path)
        else:
            print('loading model...')
            saver.restore(sess, save_path)
            t_step(test_dataset, in_path, out_path)
        # 关闭线程队列
        coord.request_stop()
        coord.join(threads)


