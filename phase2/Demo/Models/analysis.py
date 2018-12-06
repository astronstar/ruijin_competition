import jieba
# jieba.enable_parallel(4)
jieba.load_userdict('dict.txt')
import jieba.posseg as pseg
import thulac
th=thulac.thulac(filt=True)

relation_types = ['Test_Disease', 'Symptom_Disease', 'Treatment_Disease', 'Drug_Disease', 'Anatomy_Disease',
                      'Frequency_Drug', 'Duration_Drug', 'Amount_Drug', 'Method_Drug', 'SideEff-Drug', 'Other']
stop_words=["[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+"]
path=r'E:\competition\ruijin\phase2\Demo\DataSets\all_train.txt'




def get_neighbor(s):
    e1_start, e1_end, e2_start, e2_end = None, None, None, None
    for i in range(len(s)):
        if s[i] == '<':
            if s[i + 1:i + 3] == 'e1':
                e1_start = i
            elif s[i + 1:i + 4] == '/e1':
                e1_end = i
            elif s[i + 1:i + 3] == 'e2':
                e2_start = i
            elif s[i + 1:i + 4] == '/e2':
                e2_end = i

    return s[e1_start-15:e2_end+15]

words_dict={}
with open(path,'r',encoding='utf-8') as f:
    data=f.read()
    data=data.split('\n\n')
    count=0
    for instance in data:
        t,s,r=instance.split('\n')
        if t.startswith('Treatment_Disease') and r=="Relation":
            words=pseg.cut(get_neighbor(s))
            # words = th.cut(get_neighbor(s),text=True).split()
            for w in words:
                if(w.flag.startswith('v') or w.flag.startswith('n')):
                    try:
                        words_dict[w.word]+=1
                    except:
                        words_dict[w.word]=1
            count+=1
            print(instance)
            # break
    print(count)


d=sorted(words_dict.items(),key=lambda x:x[1],reverse=True)
print(d)
with open('r3.txt','w',encoding='utf-8') as f:
    for i,item in enumerate(d,1):
        f.write(str(item))
        if i%10==0:
            f.write('\n')


















