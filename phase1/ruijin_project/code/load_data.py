# encoding:utf-8
import os
import pickle

# train_txt_path = r'../data/train/0.txt'
# train_ann_path = r'../data/train/0.ann'
# to_path = r'../data/0.tsv'


class DataLoader():

    def __init__(self, train_dir, test_dir):
        self.max_seq_length = 0
        self.pad_max = 150
        if not os.path.exists('../data/words_set_b.pkl'):
            words_set = self.get_words_set(train_dir, test_dir)
            words_set_pkl = open('../data/words_set_b.pkl', 'wb')
            pickle.dump(words_set, words_set_pkl)
        else:
            words_set_pkl = open('../data/words_set_b.pkl', 'rb')
            words_set = pickle.load(words_set_pkl)
        print(words_set)

        tags_set = ['B-Disease', 'I-Disease', 'B-Reason', 'I-Reason', 'B-Symptom', 'I-Symptom', 'B-Test', 'I-Test',
                    'B-Test_Value', 'I-Test_Value', 'B-Drug', 'I-Drug', 'B-Frequency', 'I-Frequency', 'B-Amount',
                    'I-Amount', 'B-Method', 'I-Method', 'B-Treatment', 'I-Treatment', 'B-Operation', 'I-Operation',
                    'B-SideEff', 'I-SideEff', 'B-Anatomy', 'I-Anatomy', 'B-Level', 'I-Level', 'B-Duration',
                    'I-Duration', 'O'
                    ]
        self.words_set, self.tags_set = list(words_set), list(tags_set)
        self.word2id, self.tag2id = self.word_tag_to_ids(self.words_set, self.tags_set)
        print(self.word2id)
        print(self.tag2id)

        self.vocab_size, self.tag_size = len(self.words_set), len(self.tags_set)

        self.train_article_list = self.get_train_article_list(train_dir)  # [('z','O'),()]
        self.test_text_list,  self.text_ids = self.get_test_article_list(test_dir)  # ['z','z',],text_ids is the path

        word_ids_list, tag_id_list, seq_length_list = self.get_word_tag_ids_list(
            self.train_article_list, self.word2id, self.tag2id)  # update max_seq_length
        self.pad_word_ids_list, self.pad_tag_id_list = self.train_padding(word_ids_list, tag_id_list)

        text_word_ids_list, text_seq_length_list = self.get_test_word_ids_list(
            self.test_text_list, self.word2id)
        self.pad_text_word_ids_list = self.test_padding(text_word_ids_list)

        self.dataset = (self.pad_word_ids_list, self.pad_tag_id_list, seq_length_list)
        self.test_dataset = (self.pad_text_word_ids_list, text_seq_length_list)

    def train_padding(self, word_ids_list, tag_ids_list):
        pad_word_ids_list, pad_tag_ids_list = [], []
        for (word_ids, tag_ids) in zip(word_ids_list, tag_ids_list):
            if len(word_ids) < self.pad_max:
                word_ids.extend([0 for i in range(self.pad_max - len(word_ids))])
                tag_ids.extend([0 for i in range(self.pad_max - len(tag_ids))])
            elif len(word_ids) > self.pad_max:
                print(word_ids)
            pad_word_ids_list.append(word_ids)
            pad_tag_ids_list.append(tag_ids)
        return pad_word_ids_list, pad_tag_ids_list

    def test_padding(self, text_list):
        pad_text = []
        pad_word_ids_list = []
        for text in text_list:
            temp_pad = []
            for word_ids in text:
                if len(word_ids) < self.pad_max:
                    word_ids.extend([0 for i in range(self.pad_max - len(word_ids))])
                pad_word_ids_list.append(word_ids)

                temp_pad.append(word_ids)
            pad_text.append(temp_pad)
        return pad_text  # pad_word_ids_list

    def get_train_word_tag(self, txt_path, ann_path):
        with open(txt_path, 'r', encoding='utf-8') as ftxt:
            data = ftxt.readlines()
            # print(data)
            text = ''
            for row in data:
                s = row.strip('\n')
                s += '$'
                text += s
            # print(text)

        with open(ann_path, 'r', encoding='utf-8') as fann:
            ann = fann.readlines()
            start_list, end_list, pos = [], [], []
            for line in ann:
                line = line.strip().strip('\n')
                num, mid, ent = line.split('\t')
                class_type, start, *_, end = mid.split(' ')
                if start in start_list or end in end_list:
                    continue
                start_list.append(int(start))
                end_list.append(int(end))
                pos.append((int(start), int(end), class_type))
            pos = sorted(pos, key=lambda x: x[0])
            # print(pos)

        sens_list = []
        # with open(file_out_path,'w',encoding='utf-8') as fout:
        index = 0
        for i, z in enumerate(text):
            try:
                if index < len(pos):
                    if index > 0 and pos[index - 1][1] > pos[index][0] and index < len(pos) - 1:
                        index += 1
                    if i < pos[index][0]:
                        s = f'{z}\tO\n'
                        sens_list.append((z, 'O'))
                        # print(s)
                        # print(i, pos[index], index, len(pos))

                    elif i >= pos[index][0] and i < pos[index][1]:
                        if i == pos[index][0]:
                            s = f'{z}\tB-{pos[index][-1]}\n'
                            sens_list.append((z, f'B-{pos[index][-1]}'))
                            # print(s)
                            # print(i, pos[index], index, len(pos))
                        else:
                            s = f'{z}\tI-{pos[index][-1]}\n'
                            sens_list.append((z, f'I-{pos[index][-1]}'))
                            # print(s)
                            # print(i, pos[index], index, len(pos))

                        if i == pos[index][1] - 1 and index < len(pos):
                            index += 1
                            # print(index)
                    # print(i, pos[index], index, len(pos))
                elif index >= len(pos) - 1:
                    s = f'{z}\tO\n'
                    sens_list.append((z, 'O'))
                    # print(s)
                    # print(index,len(pos))

            except:
                print("ERROR================")
                print(pos[index], index)
                print(len(pos))
        # print(sens_list)
        return sens_list

    def get_words(self, txt_path):
        with open(txt_path, 'r', encoding='utf-8') as ftxt:
            data = ftxt.readlines()
            # print(data)
            text = ''
            for row in data:
                s = row.strip('\n')
                s += '$'
                text += s
            # print(text)
        return text

    def get_words_set(self, train_dir, test_dir):
        words_set = set()
        words_set.add('$')
        for path in os.listdir(train_dir):
            if path.endswith('.txt'):
                txt_path = os.path.join(train_dir, path)
                words = self.get_words(txt_path)
                for word in words:
                    words_set.add(word)

        for path in os.listdir(test_dir):
            if path.endswith('.txt'):
                txt_path = os.path.join(test_dir, path)
                words = self.get_words(txt_path)
                for word in words:
                    words_set.add(word)
        words_set=list(words_set)
        return words_set

    def get_train_article_list(self, dir):
        article_list = []

        for path in os.listdir(dir):
            if path.endswith('.txt'):
                txt_path = os.path.join(dir, path)
                ann_path = os.path.join(dir, path.replace('.txt', '.ann'))

                word_tag = self.get_train_word_tag(txt_path, ann_path)

                sen_list = []
                count = 1
                for item in word_tag:
                    if (item[0] == '。' or item[0] == '?' or item[0] == '？' or item[0] == '!' and item[1] == 'O') or count >= self.pad_max:
                        # print(item)
                        sen_list.append(item)
                        # print(len(sen_list))
                        article_list.append(sen_list)
                        # if count>150:print(sen_list)
                        sen_list = []
                        count = 1
                    else:
                        sen_list.append(item)
                        count += 1
                if sen_list:
                    article_list.append(sen_list)

        return article_list

    def get_test_article_list(self, dir):
        text_list = []
        article_list = []
        text_ids = []
        for path in os.listdir(dir):
            if path.endswith('.txt'):
                txt_path = os.path.join(dir, path)
                text_ids.append(path)
                words = self.get_words(txt_path)
                sen_list = []
                count = 1
                for item in words:
                    if (item[0] == '。' or item[0] == '?' or item[0] == '？' or item[0] == '!') or count >= self.pad_max:
                        # print(item)
                        sen_list.append(item)
                        article_list.append(sen_list)
                        sen_list = []
                        count = 1
                    else:
                        sen_list.append(item)
                        count += 1
                if sen_list:
                    article_list.append(sen_list)
                text_list.append(article_list)
                article_list=[]

        return text_list,  text_ids

    def word_tag_to_ids(self, words_set, tags_set):
        word2id, tag2id = {}, {}
        for i, item in enumerate(words_set, 1):
            word2id[item] = i
        for i, item in enumerate(tags_set, 1):
            tag2id[item] = i

        return word2id, tag2id

    def get_word_tag_ids_list(self, article_list, word2id, tag2id):
        word_ids_list, tag_id_list, seq_length_list = [], [], []

        for sen_tag in article_list:
            # print(sen_tag)
            word_ids, tag_ids = [], []
            for tuples in sen_tag:
                word_ids.append(word2id[tuples[0]])
                tag_ids.append(tag2id[tuples[1]])

            assert len(word_ids) == len(tag_ids) or len(word_ids) > self.pad_max
            self.max_seq_length = max(self.max_seq_length, len(word_ids))
            word_ids_list.append(word_ids)
            tag_id_list.append(tag_ids)
            seq_length_list.append(len(word_ids))

        return word_ids_list, tag_id_list, seq_length_list

    def get_test_word_ids_list(self, text_list, word2id):
        word_ids_list, seq_length_list = [], []
        text_ids_list, text_long_list = [], []
        for text in text_list:
            text_temp, long_temp = [], []
            for sen in text:  # print(sen_tag)
                word_ids = []
                for z in sen:
                    word_ids.append(word2id[z])

                self.max_seq_length = max(self.max_seq_length, len(word_ids))
                word_ids_list.append(word_ids)
                seq_length_list.append(len(word_ids))

                text_temp.append(word_ids)
                long_temp.append(len(word_ids))

            text_ids_list.append(text_temp)
            text_long_list.append(long_temp)

        return text_ids_list, text_long_list  # ,word_ids_list, seq_length_list


if __name__ == '__main__':
    train_dir=r'../data/ruijin_round1_train_20181022\ruijin_round1_train2_20181022'
    testb_dir=r'..\data\ruijin_round1_test_b_20181112'
    dataloder = DataLoader(train_dir, testb_dir)
    # art = dataloder.dataset
    train_list=dataloder.train_article_list
    test_list=dataloder.test_text_list

    print(dataloder.vocab_size, dataloder.max_seq_length)
    # sen_path=r'E:\competition\ruijin\scripts\train_sen.txt'
    # tag_path=r'E:\competition\ruijin\scripts\train_tag.txt'
    # test_path=r'E:\competition\ruijin\scripts\test_sen.txt'
    #
    # with open(sen_path,'w',encoding='utf-8')as fsen,open(tag_path,'w',encoding='utf-8')as ftag:
    #     for line in train_list:
    #         # print(line)
    #         for xy in line:
    #             fsen.write(f'{xy[0]} ')
    #             ftag.write(f'{xy[-1]} ')
    #         fsen.write('\n')
    #         ftag.write('\n')
    #
    # with open(test_path,'w',encoding='utf-8')as ftest_sen:
    #     text_len=[]
    #     for text in test_list:
    #         count=0
    #         for line in text:
    #             # print(line)
    #             for x in line:
    #                 ftest_sen.write(f'{x[0]} ')
    #             ftest_sen.write('\n')
    #             count+=1
    #         text_len.append(count)
    #     print(text_len)

