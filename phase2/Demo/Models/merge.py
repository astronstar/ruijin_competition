import os


all_papers={}
for i in range(1,3):
    dir_in=f'../DataSets/sub/submit_testb_save{i}'
    for path in os.listdir(dir_in):
        if path.endswith('.ann'):
            ann_path = os.path.join(dir_in, path)
            if path not in all_papers:
                all_papers[path]={}

            with open(ann_path,'r',encoding='utf-8') as f:
                data = f.readlines()
                for line in data:
                    line = line.strip().strip('\n')
                    if line.startswith('R'):
                        relation_id, mid = line.split('\t')
                        try:
                            all_papers[path][mid]+=1
                        except:
                            all_papers[path][mid]=1

path=r'../DataSets/sub/submit_testb_save'
if not os.path.exists(path):
    os.makedirs(path)
in_path=r'../DataSets/ruijin_round2_test_b/ruijin_round2_test_b'
for paper in all_papers:
    b = [key for (key, value) in all_papers[paper].items() if value > 2]
    with open(os.path.join(in_path, paper), 'r', encoding='utf-8') as fin:
        data = fin.readlines()
    with open(os.path.join(path, paper), 'w', encoding='utf-8')as f:
        for line in data:
            line = line.strip().strip('\n')
            if line.startswith('T'):
                f.write(f'{line}\n')
        for i, v in enumerate(b, 1):
            f.write(f'R{i}\t{v}\n')






# print(all_dict)
# b=[key for (key,value) in all_dict.items() if value>2]
# print(b)
# print(len(b))
# print(len(all_dict))
# with open(r'E:\competition\ruijin\phase2\Demo\DataSets\sub\submit\150_2.ann','w',encoding='utf-8') as f:
#     for i,v in enumerate(b,1):
#         f.write(f'R{i}\t{v}\n')












