import os
import pickle
import re

# relation_types = ['Test_Disease', 'Symptom_Disease', 'Treatment_Disease', 'Drug_Disease', 'Anatomy_Disease',
#                   'Frequency_Drug', 'Duration_Drug', 'Amount_Drug', 'Method_Drug', 'SideEff-Drug']
disease_left_list = ['Test', 'Symptom', 'Treatment', 'Drug', 'Anatomy']
# disease2 = ['Test', '', '', '', '']
drug_left_list = ['Frequency', 'Duration', 'Amount', 'Method', 'SideEff']
# drug2 = ['Test', '', '', '', '']


def get_text(txt_path):
    with open(txt_path, 'r', encoding='utf-8') as ftxt:
        data = ftxt.readlines()
        text = ''
        for row in data:
            s = row.strip('\n')
            s += '$'
            text += s
        # print(text)
    return text


def get_ann(ann_path):
    with open(ann_path, 'r', encoding='utf-8') as f:
        data = f.readlines()
        # entity_ann_list= []
        entity_ann_dict = {}
        relation_ann_list = []
        for line in data:
            line = line.strip().strip('\n')
            if line.startswith('T'):
                entity_id, mid, entity_name = line.split('\t')
                entity_type, start, *_, end = mid.split()
                # entity_ann_list.append((entity_id, entity_type, start, end, entity_name))
                # entity_dic
                entity_ann_dict[entity_id] = [entity_type, start, end, entity_name]
            if line.startswith('R'):
                relation_id, mid = line.split('\t')
                relation_type, arg1, arg2 = mid.split()
                relation_ann_list.append((relation_id, relation_type, arg1, arg2))
                # relation_ann_list.append(mid)   # more mid
    # print(entity_ann_list)
    # print(len(relation_ann_list))
    return entity_ann_dict, relation_ann_list


def get_sent_list_samll(text, mv_len=0):
    sent_list = []
    sent = ''
    for z in text[mv_len:]:  # modify
        sent += z
        if (z in ['。', '！', '？'] and len(sent) > 15) or len(sent) >= 160:
            sent_list.append(sent)
            sent = ''
    # print(sent_list)
    return sent_list


def get_sent_list(text, mv_len=0):
    sent_list = []
    sent = ''
    for z in text[mv_len:]:  # modify
        sent += z
        if len(sent) >= 160:
            # if (z in ['。', '！', '？', '.', '!', '?'] and len(sent) > 15) or len(sent) >= 200:
            sent_list.append(sent)
            sent = ''
    # print(sent_list)
    return sent_list


def combine_relations(sent, left_list, right_list, length):  # 将关系配对
    temp_relation_save, temp_instance_dict = [], {}
    for left in left_list:  # mei ju jie shu append rel (entity_id, entity_type, start, end)
        startl, endl = int(left[2]) - length, int(left[3]) - length
        for right in right_list:
            startr, endr = int(right[2]) - length, int(right[3]) - length
            relation_type = f'{left[1]}_{right[1]}' if left[1] != "SideEff" else f'{left[1]}-{right[1]}'
            ann = f'{relation_type} Arg1:{left[0]} Arg2:{right[0]}'

            if startl < startr:
                if len(sent)>80:
                    if len(sent[startl:endr])>80:   # 中间很长
                        mid=re.sub('。.*。','',sent)
                        s = f'{sent[startl-20:startl]}<e1>{sent[startl:endl]}</e1>{mid}<e2>{sent[startr:endr]}</e2>{sent[endr:endr+20]}'
                    else:
                        s = f'{sent[startl-20:startl]}<e1>{sent[startl:endl]}</e1>{sent[endl:startr]}<e2>{sent[startr:endr]}</e2>{sent[endr:endr+20]}'
                else:
                    s = f'{sent[0:startl]}<e1>{sent[startl:endl]}</e1>{sent[endl:startr]}<e2>{sent[startr:endr]}</e2>{sent[endr:]}'
            else:
                if len(sent) > 80:
                    if len(sent[startr:endl])>80:
                        mid=re.sub('。.*。','',sent)
                        s = f'{sent[startl-20:startl]}<e1>{sent[startl:endl]}</e1>{mid}<e2>{sent[startr:endr]}</e2>{sent[endr:endr+20]}'
                    else:
                        s = f'{sent[startr-20:startr]}<e1>{sent[startr:endr]}</e1>{sent[endr:startl]}<e2>{sent[startl:endl]}</e2>{sent[endl:endl+20]}'
                else:
                    s = f'{sent[0:startr]}<e1>{sent[startr:endr]}</e1>{sent[endr:startl]}<e2>{sent[startl:endl]}</e2>{sent[endl:]}'
            # if len(s)>177 and relation_type!='Other':
            #     print(s)
            #     print(relation_type)
            tag = relation_type
            instance = [s, tag]
            temp_instance_dict[ann] = instance
            temp_relation_save.append(ann)
    return temp_relation_save, temp_instance_dict


def get_relations(sent_list, entity_ann_dict, mv_len=0):  # each paper
    length = mv_len
    relations_save = []
    instances_save = {}
    for sent in sent_list:
        # print(sent)
        disease_left, disease_right, drug_left, drug_right = [], [], [], []
        for entity_ann in entity_ann_dict.items():
            entity_id, (entity_type, start, end, entity_name) = entity_ann  # dict key value
            start, end = int(start), int(end)

            if start >= length and end <= length + len(sent):
                # print(entity_id, entity_type, sent[start - length:end - length])
                if entity_type == 'Disease':
                    disease_right.append((entity_id, entity_type, start, end))
                if entity_type == 'Drug':
                    drug_right.append((entity_id, entity_type, start, end))

                if entity_type in disease_left_list:
                    disease_left.append((entity_id, entity_type, start, end))
                elif entity_type in drug_left_list:
                    drug_left.append((entity_id, entity_type, start, end))
        # disease one sentence
        disease_save, instances_dict = combine_relations(sent, disease_left, disease_right, length)
        relations_save.extend(disease_save)
        instances_save = {**instances_save, **instances_dict}
        # drug one sentence
        drug_save, instances_dict = combine_relations(sent, drug_left, drug_right, length)
        relations_save.extend(drug_save)
        instances_save = {**instances_save, **instances_dict}

        length += len(sent)     # 每句结束需要加上长度
    return relations_save, instances_save


def get_relations_main(txt_path, ann_path, mv_len,small_sent=True):
    text = get_text(txt_path)
    # print(text)
    if small_sent:
        sent_list = get_sent_list_samll(text, mv_len=mv_len)
        # print(sent_list)
    else:
        sent_list = get_sent_list(text, mv_len=mv_len)
        # print(sent_list)
    entity_ann_dict, relation_ann_list = get_ann(ann_path)
    # 更改形式
    relation_ann_list = [f'{relation_type} {arg1} {arg2}' for (relation_id, relation_type, arg1, arg2) in
                         relation_ann_list]

    relations_save, instances_save = get_relations(sent_list, entity_ann_dict, mv_len=mv_len)
    for instance in instances_save.items():
        key, (s, tag) = instance
        if key in relation_ann_list:  # add
            # instances_save[key] = [s, 'Relation']
            pass
        else:
            instances_save[key] = [s, 'Other']

    return relations_save, instances_save


def save_relation_results(dir_in, path_out):
    all_txt_instances = []
    txt_ids = []
    with open(path_out, 'w', encoding='utf-8') as fout:
        for path in os.listdir(dir_in):
            if path.endswith('.txt'):
                txt_path = os.path.join(dir_in, path)
                ann_path = os.path.join(dir_in, path.replace('.txt', '.ann'))

                all_relations_save, all_instances_save = [], {}
                relations_save, instances_save = get_relations_main(txt_path, ann_path, mv_len=0,small_sent=True)
                for i in instances_save:
                    if i not in all_instances_save:
                        # print('small',i)
                        all_instances_save[i]=instances_save[i]
                # print(len(all_instances_save))
                # dup
                # relations_save, instances_save = get_relations_main(txt_path, ann_path, mv_len=0,small_sent=False)
                # for i in instances_save:
                #     if i not in all_instances_save:
                #         # print('dup',i)
                #         all_instances_save[i]=instances_save[i]
                # print(len(all_instances_save))
                relations_save, instances_save = get_relations_main(txt_path, ann_path, mv_len=80,small_sent=False)
                for i in instances_save:
                    if i not in all_instances_save:
                        # print('dup1', i)
                        all_instances_save[i] = instances_save[i]
                # print(len(all_instances_save))

                each_text_instance = []
                for instance in all_instances_save.items():
                    ann, (s, tag) = instance
                    each_text_instance.append((ann, s, tag))
                    fout.write(f'{ann}\n{s}\n{tag}\n\n')

                all_txt_instances.append(each_text_instance)
                txt_ids.append(path)  # 记录文件名
    return txt_ids, all_txt_instances


def get_words_set(train_dir, test_dir):
    words_set = set()
    for path in os.listdir(train_dir):
        if path.endswith('.txt'):
            txt_path = os.path.join(train_dir, path)
            text = get_text(txt_path)
            for word in text:
                words_set.add(word)

    for path in os.listdir(test_dir):
        if path.endswith('.txt'):
            txt_path = os.path.join(test_dir, path)
            text = get_text(txt_path)
            for word in text:
                words_set.add(word)

    words_set = list(words_set)

    return words_set


def get_entity_set(train_path, test_path):
    entity_set = set()
    pattern1 = re.compile('<e1>(.+)</e1>')
    pattern2 = re.compile('<e2>(.+)</e2>')
    patternsub = re.compile('\$| ')
    with open(train_path, 'r', encoding='utf-8') as ftrain:
        data = ftrain.read()
        datalist = data.strip().split('\n\n')
        for instance in datalist:
            text = instance.split('\n')[1]
            e1 = re.findall(pattern1, text)
            e2 = re.findall(pattern2, text)
            e1 = re.sub(patternsub, '', e1[0])
            e2 = re.sub(patternsub, '', e2[0])
            if len(e1) > 1:
                entity_set.add(e1)
            if len(e2) > 1:
                entity_set.add(e2)

    with open(test_path, 'r', encoding='utf-8')as ftest:
        data = ftest.read()
        datalist = data.strip().split('\n\n')
        for instance in datalist:
            text = instance.split('\n')[1]
            e1 = re.findall(pattern1, text)
            e2 = re.findall(pattern2, text)
            e1 = re.sub(patternsub, '', e1[0])
            e2 = re.sub(patternsub, '', e2[0])
            if len(e1) > 1:
                entity_set.add(e1)
            if len(e2) > 1:
                entity_set.add(e2)
    entity_set = list(entity_set)
    return entity_set


def merge_words_set(word_set, entity_set):
    words_set_list = word_set + entity_set
    if '$' in words_set_list:
        words_set_list.remove('$')
        words_set_list.insert(0, '$')
        words_set_list.insert(0, '<UNK>')
        words_set_list.insert(0, '<PAD>')
    else:
        words_set_list.insert(0, '<UNK>')
        words_set_list.insert(0, '<PAD>')

    return words_set_list


def main():
    dir_in_train = r'../DataSets/ruijin_round2_train/ruijin_round2_train'
    dir_in_test = r'../DataSets/ruijin_round2_test_a/ruijin_round2_test_a'

    path_out_train = r'../DataSets/mydatazi/trains.txt'
    path_out_test = r'../DataSets/mydatazi/test.txt'
    if not os.path.exists(r'../DataSets/mydatazi'):
        os.makedirs(r'../DataSets/mydatazi')

    train_txtid, train_instances = save_relation_results(dir_in_train, path_out_train)
    test_txtid, test_instances = save_relation_results(dir_in_test, path_out_test)

    word_set = get_words_set(dir_in_train, dir_in_test)
    entity_set = get_entity_set(path_out_train, path_out_test)
    words_set_list = merge_words_set(word_set, entity_set)

    if not os.path.exists(r'../DataSets/mydatazi/train_instances.pkl'):
        train_instances_pkl = open(r'../DataSets/mydatazi/train_instances.pkl', 'wb')
        pickle.dump(train_instances, train_instances_pkl)

    if not os.path.exists(r'../DataSets/mydatazi/test_txtid.pkl'):
        test_txtid_pkl = open(r'../DataSets/mydatazi/test_txtid.pkl', 'wb')
        pickle.dump(test_txtid, test_txtid_pkl)

    if not os.path.exists(r'../DataSets/mydatazi/test_instances.pkl'):
        test_instances_pkl = open(r'../DataSets/mydatazi/test_instances.pkl', 'wb')
        pickle.dump(test_instances, test_instances_pkl)

    if not os.path.exists(r'../DataSets/mydatazi/words_set_b.pkl'):
        words_set_pkl = open(r'../DataSets/mydatazi/words_set_b.pkl', 'wb')
        pickle.dump(words_set_list, words_set_pkl)

    return train_instances, test_txtid, test_instances, words_set_list


def get_testb():
    dir_in_testb = r'../DataSets/ruijin_round2_test_b/ruijin_round2_test_b'
    path_out_testb = r'../DataSets/mydatazi/testb.txt'
    testb_txtid, testb_instances = save_relation_results(dir_in_testb, path_out_testb)
    if not os.path.exists(r'../DataSets/mydatazi/testb_txtid.pkl'):
        testb_txtid_pkl = open(r'../DataSets/mydatazi/testb_txtid.pkl', 'wb')
        pickle.dump(testb_txtid, testb_txtid_pkl)

    if not os.path.exists(r'../DataSets/mydatazi/testb_instances.pkl'):
        testb_instances_pkl = open(r'../DataSets/mydatazi/testb_instances.pkl', 'wb')
        pickle.dump(testb_instances, testb_instances_pkl)

    return testb_txtid, testb_instances


if __name__ == '__main__':
    print('run utils main...')
    main()
    # ann=get_ann(r'E:\competition\ruijin\phase2\Demo\DataSets\ruijin_round2_test_b\ruijin_round2_test_b\1.ann')
    # print(ann)
