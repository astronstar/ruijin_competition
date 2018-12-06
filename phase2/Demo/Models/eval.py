

def get_ann(ann_path):
    # ann_path = r'E:\competition\ruijin\phase2\Demo\DataSets\ruijin_round2_train\ruijin_round2_train\9_3.ann'
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
                # if relation_type.startswith('Anatomy'):
                relation_ann_list.append((relation_type, arg1, arg2))
                # relation_ann_list.append(mid)   # more mid
    # print(entity_ann_list)
    # print(relation_ann_list)
    return entity_ann_dict, relation_ann_list

# g_path=r'E:\competition\ruijin\phase2\Demo\DataSets\ruijin_round2_test_b\ruijin_round2_test_b\0.ann'
g_path=r'E:\competition\ruijin\phase2\Demo\DataSets\ruijin_round2_test_a\ruijin_round2_test_a\150_2.ann'
p_path1=r'E:\competition\ruijin\phase2\Demo\DataSets\sub\submit_100\150_2.ann'
# p_path1=r'E:\competition\ruijin\phase2\Demo\DataSets\sub\submit_test_1\23.ann'
p_path2=r'E:\competition\ruijin\phase2\Demo\DataSets\sub\submit_2\0.ann'
p_path3=r'E:\competition\ruijin\phase2\Demo\DataSets\sub\submit_3\0.ann'
_,gold_ann=get_ann(g_path)
_,pre_ann1=get_ann(p_path1)
# _,pre_ann2=get_ann(p_path2)
# _,pre_ann3=get_ann(p_path3)
pre_ann=list(set(pre_ann1))
d={}
for i in pre_ann:
    d[i]=0


gl,pl=len(gold_ann),len(pre_ann)
print(gl,pl)
count=0
for i in range(pl):
    if pre_ann[i] in gold_ann:
        count+=1
print(count)
p=count/pl
r=count/gl
f=2*p*r/(p+r)
print(p,r,f)
