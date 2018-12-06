# coding=utf-8
import os


def save_ann(dir_in, dir_out):
    for path in os.listdir(dir_in):
        fin_path = os.path.join(dir_in, path)
        fout_path = os.path.join(dir_out, path)
        with open(fin_path, 'r', encoding='utf-8') as fin, open(fout_path, 'w', encoding='utf-8') as fout:
            data = fin.readlines()
            s, tags = '', []
            for i, line in enumerate(data):
                line = line.rstrip('\n')
                if i % 2 == 0:
                    s += line
                else:
                    t = line.split(' ')[:-1]
                    tags.extend(t)
            pos = []
            i = 0
            while i < len(tags):
                if tags[i].startswith('B-'):
                    start = i
                    class_type = tags[i][2:]
                    while True:
                        i += 1
                        if tags[i].startswith('B-'):
                            end = i
                            pos.append((class_type, start, end, s[start:end]))
                            break
                        elif tags[i] == 'O':
                            end = i
                            pos.append((class_type, start, end, s[start:end]))
                            break

                elif tags[i].startswith('I-'):
                    start = i
                    class_type = tags[i][2:]
                    while True:
                        i += 1
                        if tags[i].startswith('B-'):
                            end = i
                            pos.append((class_type, start, end, s[start:end]))
                            break
                        elif tags[i] == 'O':
                            end = i
                            pos.append((class_type, start, end, s[start:end]))
                            break

                elif tags[i] == 'O':
                    i += 1

            print(pos)
            for i, t in enumerate(pos, 1):
                temp = t[3]
                index = temp.find('$')
                if index != -1:
                    m = t[1] + index
                    fout.write(f"T{i}\t{t[0]} {t[1]} {m};{m+1} {t[2]}\t{t[3].replace('$',' ')}\n")
                else:
                    fout.write(f'T{i}\t{t[0]} {t[1]} {t[2]}\t{t[3]}\n')


if __name__ == '__main__':
    ann_path = r'E:\competition\ruijin\data\ann\ann3'  # 2xu yao gai
    submit_path = r'E:\competition\ruijin\data\submit\ann3'
    if not os.path.exists(submit_path):
        os.makedirs(submit_path)
        # save_ann(r'E:\competition\ruijin\data\submit\s7\ann',r'E:\competition\ruijin\data\submit\s7\submit')
        # save_ann(r'E:\competition\ruijin\data\submit\s8\ann', r'E:\competition\ruijin\data\submit\s8\submit')
        save_ann(ann_path, submit_path)
