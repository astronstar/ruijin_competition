# coding:utf-8

import tensorflow as tf
import os
from models import AttRNN
from d import DataLoader
import numpy as np

num = '1'
in_path=r'../DataSets/ruijin_round2_test_a/ruijin_round2_test_a'
out_path=r'../DataSets/sub/submit_test_'+num
if not os.path.exists(out_path):
        os.makedirs(out_path)
save_path = os.path.join(r'../DataSets/model_save', 'check_points_' + num, 'model')
if not os.path.exists(save_path):
    os.makedirs(save_path)

# prepare data set
dataloader = DataLoader()
# test_dataset = dataloader.get_testb_instance()
test_dataset = dataloader.test_dataset

# hyper
word_embedding_size = 150
position_embedding_size=30
l2_reg_lambda = 1e-5

model = AttRNN(
    num_classes=len(dataloader.relation_types),
    vocab_size=dataloader.vocab_size,
    word_embedding_size=word_embedding_size,
    position_size=2 * dataloader.position_max + 2,
    position_embedding_size=position_embedding_size,
    l2_reg_lambda=l2_reg_lambda,
    # trainable=True
)


def t_step(test_data, in_path,out_path):
    for (txtid, tag, ids, p1, p2, y) in zip(*test_data):
        print(txtid)
        print('batch_size: ',len(tag))

        test_feed_dict = {
            model.input_x: ids,
            model.input_p1: p1,
            model.input_p2: p2,
            model.input_y: y,
        }
        run_op = [model.predictions]
        predictions = sess.run(run_op, feed_dict=test_feed_dict)
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
                    fout.write(f'R{count}\t{t}\n')
                    count+=1

saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print('loading model...')
    saver.restore(sess, save_path)
    t_step(test_dataset, in_path, out_path)


# # coding:utf-8
#
# import os
# import tensorflow as tf
# from models import AttRNN
# from dataloader import DataLoader
# import numpy as np
#
#
# num = '3'
# # in_path=r'../DataSets/ruijin_round2_test_b/ruijin_round2_test_b'
# in_path = r'../DataSets/ruijin_round2_test_a/ruijin_round2_test_a'
# out_path = r'../DataSets/sub/submit_d' + num
# if not os.path.exists(out_path):
#     os.makedirs(out_path)
#
# with tf.Graph().as_default():
#     session_conf = tf.ConfigProto(
#         allow_soft_placement=True,
#         log_device_placement=False)
#     save_path = os.path.join(r'../DataSets/model_save', 'check_points_' + num, 'model')
#     if not os.path.exists(save_path):
#         os.makedirs(save_path)
#
#     # prepare data set
#     dataloader = DataLoader()
#     test_dataset=dataloader.test_dataset
#     # test_dataset = dataloader.get_testb_instance()
#
#     # hyper
#     l2_reg_lambda = 1e-5
#     learning_rate = 0.001
#     word_embedding_size = 50
#     position_embedding_size = 20
#
#     model = AttRNN(
#         num_classes=len(dataloader.relation_types),
#         vocab_size=dataloader.vocab_size,
#         word_embedding_size=word_embedding_size,
#         position_size=2 * dataloader.position_max + 2,
#         position_embedding_size=position_embedding_size,
#         l2_reg_lambda=l2_reg_lambda,
#         # trainable=True
#     )
#
#
#     def t_step(test_data, in_path, out_path):
#         for (txtid, tag, ids, p1, p2, y) in zip(*test_data):
#             print('batch_size: ', len(tag))
#             test_feed_dict = {
#                 model.input_x: ids,
#                 model.input_p1: p1,
#                 model.input_p2: p2,
#                 model.input_y: y,
#             }
#             run_op = [model.predictions]
#             predictions = sess.run(run_op, feed_dict=test_feed_dict)
#             predictions = np.asarray(predictions)[0]
#
#             pre_tag = [dataloader.relation_types[i] for i in predictions]
#             with open(os.path.join(in_path, txtid.replace('.txt', '.ann')), 'r', encoding='utf-8') as fin:
#                 data = fin.readlines()
#             with open(os.path.join(out_path, txtid.replace('.txt', '.ann')), 'w', encoding='utf-8') as fout:
#                 for line in data:
#                     line = line.strip().strip('\n')
#                     if line.startswith('T'):
#                         fout.write(f'{line}\n')
#                 count = 1
#                 for (t, pre) in zip(tag, pre_tag):
#                     if pre != 'Other':
#                         fout.write(f'R{count}\t{t}\n')
#                         count += 1
#
#     saver = tf.train.Saver()
#     with tf.Session(config=session_conf) as sess:
#         sess.run(tf.global_variables_initializer())
#         print("test...")
#         saver.restore(sess, save_path)
#         t_step(test_dataset, in_path, out_path)

