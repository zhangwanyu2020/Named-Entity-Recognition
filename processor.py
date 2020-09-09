import json

f_r = open('./train_data/relation_train2.csv', 'w', encoding='utf-8')
f_n = open('./train_data/ner_train2.csv', 'w', encoding='utf-8')

query_path = './train_data/query_mapping.json'
query_mapping = json.load(open(query_path, 'r', encoding='utf-8'))

with open('./datasets/train_data.json', 'r', encoding='utf-8') as fr:
    data = fr.readlines()
    fr.close()

p_label = {}
with open('./datasets/relation_label.csv', 'r', encoding='utf-8') as fr:
    labels = fr.readlines()
    fr.close()
for line in labels:
    line = line.strip()
    line = line.split(',')
    p_label[line[1]] = int(line[0])


def get_index(text, entity):
    text = str(text)
    entity = str(entity)
    entity_len = len(entity)
    start = text.index(entity)
    end = start + len(entity)
    return str(start), str(end)


for line in data:
    line = json.loads(line.strip('\n'))
    text = line['text']
    spo_list = line['spo_list']
    relation_ner = {}  # 关系会有重复
    for spo in spo_list:
        p = spo['predicate']
        p = p_label[p]

        s = spo['subject']
        o = spo['object']['@value']
        s_s, s_e = get_index(text, s)
        o_s, o_e = get_index(text, o)

        if p not in relation_ner:
            relation_ner[p] = [s_s, o_s], [s_e, o_e]
        else:
            start, end = relation_ner[p]
            start.extend([s_s, o_s])
            end.extend([s_e, o_e])

    relations = ''
    for k, v in relation_ner.items():
        relations = relations + str(k) + ' '
        query = query_mapping[str(k)]
        ner_start = ' '.join(v[0])
        ner_end = ' '.join(v[1])
        sample_n = query + 'fengefu' + text + 'fengefu' + ner_start + 'fengefu' + ner_end + '\n'
        f_n.write(sample_n)
    #         print(sample_n)
    sample_r = text + 'fengefu' + relations.strip(' ') + '\n'
    f_r.write(sample_r)
    # print(sample_r)





