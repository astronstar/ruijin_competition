# coding:utf-8
import os
import pickle
import re
from utils import main,get_train,get_test,get_testb,get_words_set
import random
random.seed(1)

class DataLoader():
    relation_types = ['Test_Disease', 'Symptom_Disease', 'Treatment_Disease', 'Drug_Disease', 'Anatomy_Disease',
                      'Frequency_Drug', 'Duration_Drug', 'Amount_Drug', 'Method_Drug', 'SideEff-Drug', 'Other']
    # relation_types = ['Relation', 'Other']
    relation_to_dict = {}
    for i, relation_type in enumerate(relation_types):
        relation_to_dict[relation_type] = i
    print('relation_to_dict:\n', relation_to_dict)
    stop_words=[' ','$','@','#','￥','&','!','-','(',')',"'",'"',
                '¨','⋯','+','—','！','？','、','~','“','”','‘','’','（','）',]

    def __init__(self, max_sequence_length=160):
        if os.path.exists('../DataSets/mydata/train_instances.pkl') and os.path.exists(
                '../DataSets/mydata/test_txtid.pkl') and os.path.exists(
                '../DataSets/mydata/test_instances.pkl') and os.path.exists('../DataSets/mydata/words_set_b.pkl'):
            print('loading pkl...')
            train_instances_pkl = open('../DataSets/mydata/train_instances.pkl', 'rb')
            train_instances = pickle.load(train_instances_pkl)
            test_txtid_pkl = open('../DataSets/mydata/test_txtid.pkl', 'rb')
            test_txtid = pickle.load(test_txtid_pkl)
            test_instances_pkl = open('../DataSets/mydata/test_instances.pkl', 'rb')
            test_instances = pickle.load(test_instances_pkl)
            words_set_pkl = open('../DataSets/mydata/words_set_b.pkl', 'rb')
            words_set_list = pickle.load(words_set_pkl)
        else:
            print('get pkl...')
            train_instances, test_txtid, test_instances, words_set_list = main()

        train_paper_list = self.load_data(train_instances)
        test_paper_list = self.load_data(test_instances)
        self.word2id_dic = self.word2id(words_set_list)

        self.max_sequence_length = max_sequence_length
        self.position_max = 65
        # all [[instance1],[inctance2]...]
        train_tag_list, train_id_list, train_p1_list, train_p2_list, train_y_list = self.train_padding(train_paper_list,
                                                                                                       self.word2id_dic,
                                                                                                       self.max_sequence_length,
                                                                                                       self.position_max)
        # all papers [[[instance1],[instance2]...],[[instance1],[instance2]...],...]
        test_tag_list, test_id_list, test_p1_list, test_p2_list, test_y_list = self.test_padding(test_paper_list,
                                                                                                 self.word2id_dic,
                                                                                                 self.max_sequence_length,
                                                                                                 self.position_max)
        print(self.word2id_dic)
        print(f'max_sequence_length: {max_sequence_length}')
        self.vocab_size = len(self.word2id_dic)
        print(f'vocab_size: {self.vocab_size}')
        self.trainset_size = len(train_y_list)
        self.testset_size = len(test_y_list)
        print(f'trainset_size: {self.trainset_size}')
        print(f'testaset_size: {self.testset_size}')

        self.train_dataset = (train_id_list, train_p1_list, train_p2_list, train_y_list)
        self.test_dataset = (test_txtid, test_tag_list, test_id_list, test_p1_list, test_p2_list, test_y_list)
        # add
        train_instances_min1, train_instances_min2, train_instances_min3 = self.get_train_min(train_instances)

        self.train_min1_dataset = self.get_train_instance(train_instances_min1)
        self.train_min2_dataset = self.get_train_instance(train_instances_min2)
        self.train_min3_dataset = self.get_train_instance(train_instances_min3)
        print('len all train_dataset ->', len(list(zip(*self.train_dataset))))
        print('len train_dataset_min1 ->', len(list(zip(*self.train_min1_dataset))))
        print('len train_dataset_min2 ->', len(list(zip(*self.train_min2_dataset))))
        print('len train_dataset_min3 ->', len(list(zip(*self.train_min3_dataset))))

    def get_train_min(self,train_instances):
        train_instances_min1, train_instances_min2, train_instances_min3 = [], [], []
        for paper in train_instances:
            temp_instance1, temp_instance2, temp_instance3 = [], [], []
            for instance in paper:
                ann, s, tag = instance
                r=random.random()
                # print(r)
                if tag == 'Other':
                    if r< 0.2:
                        temp_instance1.append(instance)
                    elif (r > 0.25)and (r < 0.5):
                        temp_instance2.append(instance)
                    elif r>0.5:
                        temp_instance3.append(instance)
                else:
                    temp_instance1.append(instance)
                    temp_instance2.append(instance)
                    temp_instance3.append(instance)
            train_instances_min1.append(temp_instance1)
            train_instances_min2.append(temp_instance2)
            train_instances_min3.append(temp_instance3)
        return train_instances_min1,train_instances_min2,train_instances_min3

    def get_train_instance(self,train_instances):
        train_paper_list = self.load_data(train_instances)
        train_tag_list, train_id_list, train_p1_list, train_p2_list, train_y_list = self.train_padding(train_paper_list,
                                                                                                       self.word2id_dic,
                                                                                                       self.max_sequence_length,
                                                                                                       self.position_max)
        train_dataset = (train_id_list, train_p1_list, train_p2_list, train_y_list)
        return train_dataset

    def get_test_instance(self):
        if os.path.exists('../DataSets/mydata/test_txtid.pkl') and os.path.exists('../DataSets/mydata/test_instances.pkl'):
            test_txtid_pkl = open('../DataSets/mydata/test_txtid.pkl', 'rb')
            test_txtid = pickle.load(test_txtid_pkl)
            test_instances_pkl = open('../DataSets/mydata/test_instances.pkl', 'rb')
            test_instances = pickle.load(test_instances_pkl)
        else:
            test_txtid, test_instances = get_test()

        test_paper_list = self.load_data(test_instances)
        test_tag_list, test_id_list, test_p1_list, test_p2_list, test_y_list = self.test_padding(test_paper_list,
                                                                                                      self.word2id_dic,
                                                                                                      self.max_sequence_length,
                                                                                                      self.position_max)
        self.testb_dataset = (test_txtid, test_tag_list, test_id_list, test_p1_list, test_p2_list, test_y_list)
        return self.testb_dataset

    def get_testb_instance(self):
        if os.path.exists('../DataSets/mydata/testb_txtid.pkl') and os.path.exists('../DataSets/mydata/testb_instances.pkl'):
            testb_txtid_pkl = open('../DataSets/mydata/testb_txtid.pkl', 'rb')
            testb_txtid = pickle.load(testb_txtid_pkl)
            testb_instances_pkl = open('../DataSets/mydata/testb_instances.pkl', 'rb')
            testb_instances = pickle.load(testb_instances_pkl)
        else:
            testb_txtid, testb_instances = get_testb()

        testb_paper_list = self.load_data(testb_instances)
        testb_tag_list, testb_id_list, testb_p1_list, testb_p2_list, testb_y_list = self.test_padding(testb_paper_list,
                                                                                                 self.word2id_dic,
                                                                                                 self.max_sequence_length,
                                                                                                 self.position_max)
        self.testb_dataset = (testb_txtid, testb_tag_list, testb_id_list, testb_p1_list, testb_p2_list, testb_y_list)
        return self.testb_dataset


    def load_data(self, paper_instcne):
        paper_list = []
        for paper in paper_instcne:
            tag_list, words_list, p1_list, p2_list, y_list = [], [], [], [], []
            for instance in paper:
                if len(instance) < 3:
                    print(instance)
                    continue
                tag, x, p1, p2, y = self.get_words_pos_y(instance)
                tag_list.append(tag)
                words_list.append(x)  # train or test words [['a','boy'],['the','girl']...]
                p1_list.append(p1)  # train or test
                p2_list.append(p2)  # train or test
                y_list.append(y)  # train or test
            paper_list.append([tag_list, words_list, p1_list, p2_list, y_list])
        return paper_list

    # def load_train_data(self, train_instances):
    #     words_list, p1_list, p2_list, y_list = [], [], [], []
    #     for paper in train_instances:
    #         for instance in paper:
    #             if len(instance) < 3:
    #                 print(instance)
    #                 continue
    #             x, p1, p2, y = self.get_words_pos_y(instance)
    #             words_list.append(x)  # train or test words [['a','boy'],['the','girl']...]
    #             p1_list.append(p1)  # train or test
    #             p2_list.append(p2)  # train or test
    #             y_list.append(y)  # train or test
    #
    #     return words_list, p1_list, p2_list, y_list
    #
    # def load_test_data(self, train_instances):
    #     paper_list = []
    #     for paper in train_instances:
    #         words_list, p1_list, p2_list, y_list = [], [], [], []
    #         for instance in paper:
    #             if len(instance) < 3:
    #                 print(instance)
    #                 continue
    #             x, p1, p2, y = self.get_words_pos_y(instance)
    #             words_list.append(x)  # train or test words [['a','boy'],['the','girl']...]
    #             p1_list.append(p1)  # train or test
    #             p2_list.append(p2)  # train or test
    #             y_list.append(y)  # train or test
    #         paper_list.append([words_list, p1_list, p2_list, y_list])
    #     return paper_list

    def get_words_pos_y(self, instance):
        tag, sentence, relation = instance
        tokens = sentence.strip('"').strip('.')

        e1_start, e1_end, e2_start, e2_end = None, None, None, None
        for i in range(len(tokens)):
            if tokens[i] == '<':
                if tokens[i + 1:i + 3] == 'e1':
                    e1_start = i
                elif tokens[i + 1:i + 4] == '/e1':
                    e1_end = i
                elif tokens[i + 1:i + 3] == 'e2':
                    e2_start = i
                elif tokens[i + 1:i + 4] == '/e2':
                    e2_end = i
        if any(x is None for x in [e1_start, e1_end, e2_start, e2_end]):
            print(sentence)

        pattern = '\(.*?\)|\[.*?\]|\【.*\】|\{.*?}'
        words_before = tokens[:e1_start]
        words_before = re.sub(pattern, '', words_before)
        e1 = tokens[e1_start + 4: e1_end]
        words_middle = tokens[e1_end + 5: e2_start]
        words_middle = re.sub(pattern, '', words_middle)
        e2 = tokens[e2_start + 4: e2_end]
        words_after = tokens[e2_end + 5:]
        words_after = re.sub(pattern, '', words_after)

        patternsub = '$| '
        e1 = [re.sub(patternsub, '', e1)]
        e2 = [re.sub(patternsub, '', e2)]
        # words_before = [word for word in words_before]
        # words_middle = [word for word in words_middle]
        # words_after = [word for word in words_after]
        words_before = [word for word in words_before if word not in self.stop_words]
        words_middle = [word for word in words_middle if word not in self.stop_words]
        words_after = [word for word in words_after if word not in self.stop_words]

        words = words_before + e1 + words_middle + e2 + words_after
        # print(sentence)
        # print(words)
        p1 = list(range(-len(words_before), 0)) + [0] * len(e1) \
             + list(range(1, len(words_middle) + len(e2) + len(words_after) + 1))
        p2 = list(range(-(len(words_before) + len(e1) + len(words_middle)), 0)) \
             + [0] * len(e2) + list(range(1, len(words_after) + 1))

        y = self.relation_to_dict[relation]
        return tag, words, p1, p2, y

    def train_padding(self, paper_list, word2id_dict, max_sequence_length, position_max):
        all_tag_list, all_ids_list, all_p1list, all_p2list, all_y_list = [], [], [], [], []
        for paper in paper_list:
            tag_list, words_list, p1_list, p2_list, y_list = paper  # each paper
            ids_list, p1list, p2list = [], p1_list[:], p2_list[:]  # copy list each paper
            for words in words_list:
                ids = [word2id_dict.get(word, word2id_dict['<UNK>']) for word in words]  # each instance
                ids_list.append(ids)  # each paper

            for i, (ids, p1, p2) in enumerate(zip(ids_list, p1list, p2list)):
                assert len(ids) == len(p1)
                assert len(p1) == len(p2)
                ids.extend([word2id_dict['<PAD>'] for i in range(max_sequence_length - len(ids))])  # padding 0

                p1 = [-position_max if pos < -position_max else pos for pos in p1]
                p1 = [position_max if pos > position_max else pos for pos in p1]
                p1 = [i + position_max + 1 for i in p1]
                p1.extend([word2id_dict['<PAD>'] for i in range(max_sequence_length - len(p1))])
                p1list[i] = p1

                p2 = [-position_max if pos < -position_max else pos for pos in p2]
                p2 = [position_max if pos > position_max else pos for pos in p2]
                p2 = [i + position_max + 1 for i in p2]
                p2.extend([word2id_dict['<PAD>'] for i in range(max_sequence_length - len(p2))])
                p2list[i] = p2
            # merge all paper
            all_tag_list.extend(tag_list)
            all_ids_list.extend(ids_list)
            all_p1list.extend(p1list)
            all_p2list.extend(p2list)
            all_y_list.extend(y_list)

        return all_tag_list, all_ids_list, all_p1list, all_p2list, all_y_list

    def test_padding(self, paper_list, word2id_dict, max_sequence_length, position_max):
        all_tag_list, all_ids_list, all_p1list, all_p2list, all_y_list = [], [], [], [], []
        for paper in paper_list:
            tag_list, words_list, p1_list, p2_list, y_list = paper  # each paper
            ids_list, p1list, p2list = [], p1_list[:], p2_list[:]  # copy list each paper
            for words in words_list:
                ids = [word2id_dict.get(word, word2id_dict['<UNK>']) for word in words]  # each instance
                ids_list.append(ids)  # each paper

            for i, (ids, p1, p2) in enumerate(zip(ids_list, p1list, p2list)):
                assert len(ids) == len(p1)
                assert len(p1) == len(p2)
                ids.extend([word2id_dict['<PAD>'] for i in range(max_sequence_length - len(ids))])  # padding 0

                p1 = [-position_max if pos < -position_max else pos for pos in p1]
                p1 = [position_max if pos > position_max else pos for pos in p1]
                p1 = [i + position_max + 1 for i in p1]
                p1.extend([word2id_dict['<PAD>'] for i in range(max_sequence_length - len(p1))])
                p1list[i] = p1

                p2 = [-position_max if pos < -position_max else pos for pos in p2]
                p2 = [position_max if pos > position_max else pos for pos in p2]
                p2 = [i + position_max + 1 for i in p2]
                p2.extend([word2id_dict['<PAD>'] for i in range(max_sequence_length - len(p2))])
                p2list[i] = p2
            # merge all paper
            all_tag_list.append(tag_list)
            all_ids_list.append(ids_list)
            all_p1list.append(p1list)
            all_p2list.append(p2list)
            all_y_list.append(y_list)

        return all_tag_list, all_ids_list, all_p1list, all_p2list, all_y_list

    def mapping_padding(self, paper_list, word2id_dict, max_sequence_length, position_max):
        words_list, p1_list, p2_list, y_list = paper_list
        ids_list, p1list, p2list = [], p1_list[:], p2_list[:]  # copy list
        for words in words_list:
            ids = [word2id_dict.get(word, '<UNK>') for word in words]
            ids_list.append(ids)

        for i, (ids, p1, p2) in enumerate(zip(ids_list, p1list, p2list)):
            assert len(ids) == len(p1)
            assert len(p1) == len(p2)
            ids.extend([word2id_dict['<PAD>'] for i in range(max_sequence_length - len(ids))])  # padding 0

            p1 = [-position_max if pos < -position_max else pos for pos in p1]
            p1 = [position_max if pos > position_max else pos for pos in p1]
            p1 = [i + position_max + 1 for i in p1]
            p1.extend([word2id_dict['<PAD>'] for i in range(max_sequence_length - len(p1))])
            p1list[i] = p1

            p2 = [-position_max if pos < -position_max else pos for pos in p2]
            p2 = [position_max if pos > position_max else pos for pos in p2]
            p2 = [i + position_max + 1 for i in p2]
            p2.extend([word2id_dict['<PAD>'] for i in range(max_sequence_length - len(p2))])
            p2list[i] = p2

        return ids_list, p1list, p2list, y_list

    def word2id(self, words_set):
        # special_words = ['<PAD>', '<UNK>', ]
        words_set = list(words_set)
        word_to_idx = {word: idx for idx, word in enumerate(words_set)}
        return word_to_idx

    def get_text(self, txt_path):
        with open(txt_path, 'r', encoding='utf-8') as ftxt:
            data = ftxt.readlines()
            text = ''
            for row in data:
                s = row.strip('\n')
                s += '$'
                text += s
        return text


if __name__ == '__main__':
    dataloader = DataLoader(max_sequence_length=160)
    print(dataloader.trainset_size)
    print(dataloader.testset_size)
    print(dataloader.vocab_size)
    # print(dataloader.test_dataset)
    # for paper_instance in zip(*dataloader.test_dataset):
    #     txtid, tag, ids, p1, p2, y = paper_instance
    #     print(p1)
    #     print(p2)
    #     for i, (batch_tag, batch_ids, batch_p1, batch_p2, batch_y) in enumerate(zip(tag, ids, p1, p2, y)):
    #         # print(i,batch_tag,batch_ids,batch_p1,batch_p2,batch_y)
    #         pass
    print('len',len(dataloader.train_dataset))
    print(len(dataloader.train_min1_dataset))
    # for i, instance in enumerate(zip(*dataloader.train_dataset)):
    #     if i % 500 == 0:
    #         print(i)
    #         print(instance)
    # print(i)

    print('all',len(list(zip(*dataloader.train_dataset))))
    #     # break
    # for i, instance in enumerate(zip(*dataloader.train_min1_dataset)):
    #     if i % 500 == 0:
    #         print(i)
    #         print(instance)
    # print(i)
    # for i, instance in enumerate(zip(*dataloader.train_min2_dataset)):
    #     if i % 500 == 0:
    #         print(i)
    #         print(instance)
    # print(i)