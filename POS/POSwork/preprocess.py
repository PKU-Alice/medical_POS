import os
import json
from collections import Counter

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cws_train_path = os.path.join(base_dir, 'data/trainset/train_cws.txt')
cws_val_path = os.path.join(base_dir, 'data/devset/val_cws.txt')
cws_test1_path = os.path.join(base_dir, 'data/testset/test_cws.txt')

# POSwork file path
pos_train_path = os.path.join(base_dir, 'data/trainset/train_pos.txt')
pos_val_path = os.path.join(base_dir, 'data/devset/val_pos.txt')
pos_test_path = os.path.join(base_dir, 'data/testset/test_pos.txt')

cws_tag_id_path = os.path.join(base_dir, 'data/cws_tag.json')
pos_tag_id_path = os.path.join(base_dir, 'data/pos_tag.json')
word_id_path = os.path.join(base_dir, 'data/word_id.json')

def preprocess():
    words = []
    for file_path in [cws_train_path, cws_val_path, cws_test1_path]:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                words.extend(line.strip().split())

    chars = []
    for word in words:
        chars.extend(list(word))

    # word_counts：对所有词和字的计数，all_words：所有词和字
    word_counts = Counter([w for w in words if len(w) >= 2] + chars).most_common()

    all_words = [word for word, count in word_counts]

    word2id = {'<PAD>': 0, "<UNK>": 1}  # UNK: unknown token    # PAD 和CNN中的padding一回事
    for index, word in enumerate(all_words):
        word2id.update({word: index + 2})

    word2id_list = word2id.items()
    id2word = {id: word for word, id in word2id_list}

    with open(word_id_path, 'w', encoding='utf-8') as f:
        json.dump([word2id, id2word], f, indent=4, ensure_ascii=False, sort_keys=True)
    # -------------------------------- word_id.json文件Done --------------------------------

    # cws tags
    cws_tag2id = {'B': 0, 'I': 1, 'E': 2, 'S': 3}
    cws_id2tag = {id: tag for tag, id in cws_tag2id.items()}

    with open(cws_tag_id_path, 'w', encoding='utf-8') as f:
        json.dump([cws_tag2id, cws_id2tag], f, indent=4, ensure_ascii=False, sort_keys=True)
    # -------------------------------- ws_tag.json文件Done --------------------------------

    # POS tag 词性标注数据预处理
    with open(pos_train_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    all_tags = []  # all_tags是所有词性标注的label
    for line in lines:
        for word_tag_str in line.strip().split(' '):
            # print('word_tag_str:',word_tag_str)
            if word_tag_str:
                word_tag = word_tag_str.split('/')
                # print(word_tag)
                if isinstance(word_tag, list) and len(word_tag) == 2 and word_tag[1]:
                    all_tags.append(word_tag[1])

    # all_tags包括所有BMIS和pos组合以及单独pos的所有tags
    tag_count = Counter(all_tags).most_common()

    pos2tag_id = {}
    for position in 'BIES':
        for (pos_tag, cnt) in tag_count:
            tag = "{}-{}".format(position, pos_tag)
            pos2tag_id[tag] = len(pos2tag_id) + 1

    for pos_tag, cnt in tag_count:
        pos2tag_id[pos_tag] = len(pos2tag_id) + 1

    pos_id2tag = {id: tag for tag, id in pos2tag_id.items()}

    with open(pos_tag_id_path, 'w', encoding='utf-8') as f:
        json.dump([pos2tag_id, pos_id2tag], f, indent=4, ensure_ascii=False, sort_keys=True)
    # -------------------------------- pos_tag.json文件Done --------------------------------

if __name__ == '__main__':
    print('heiheihei')
    preprocess()