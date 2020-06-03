from preprocess import *
import numpy as np

def load_pos_data(stage='train'):
    '''

    :param stage: 要load的数据是用于训练、验证还是测试
    :param char_base: 基于字还是基于词
    :param write:
    :return:
    '''
    file_path = {'train':pos_train_path, 'valid':pos_val_path, 'test':pos_test_path}[stage]

    with open(file_path,'r',encoding='utf-8') as f:
        lines = f.readlines()
    all_the_tags = []
    for line in lines:
        line_tags = []

        for _word_tag in line.strip().split(' '):
            word_tag = _word_tag.strip().split('/')

            if isinstance(word_tag,list) and len(word_tag) == 2 and word_tag[1]:
                word, tag = word_tag
                if len(word) == 1:
                    line_tags.append([word, "S-" + tag])
                elif len(word) >= 2:
                    words_tags = [[w, 'I-' + tag] for w in word]
                    words_tags[0][1] = 'B-' + tag
                    words_tags[-1][1] = 'E-' + tag
                    line_tags.extend(words_tags)

        all_the_tags.append(line_tags)

    # for line_tag in all_the_tags:
    #     # print(line_tag)
    #     _lines = ["{} {}".format(w, t) for w, t in line_tag]
    #
    #     # print(len(_lines))
    #     _lines.append("")
    #     # print(len(_lines))
    #     lines.extend(_lines)
    return all_the_tags


class DataHelper:
    def __init__(self):

        self.sequence_lenghth=100

        with open(pos_tag_id_path,'r',encoding='utf-8') as f:
            self.tag2id, self.id2tag = json.load(f)
        with open(word_id_path,'r',encoding='utf-8') as f:
            self.word2id, self.id2word = json.load(f)

        self.num_tags = len(self.tag2id)
        self.vocab_size = len(self.word2id)

    def pad(self, sentence):

        if len(sentence) < self.sequence_lenghth:
            sentence = sentence + [0] * (self.sequence_lenghth-len(sentence))

        return sentence[:self.sequence_lenghth]

    def generate_datas(self, batch_size=64, epoch_num=50, stage="train"):
        '''

        :param batch_size:
        :param epoch_num:
        :param stage: 数据用于训练——"train"，用于验证——"valid"，用于测试——"test"
        :return:
        '''

        tagged_sentences = load_pos_data(stage=stage)

        for epoch in range(epoch_num):
            np.random.shuffle(tagged_sentences)

            x_data = []
            y_data = []
            sentence_lengths = []

            for sentence in tagged_sentences:
                words = [w for w,tag in sentence]
                tags = [tag for w,tag in sentence]

                x = [self.word2id.get(word,1) for word in words]
                x = self.pad(x)     # 让每个sentence的长度都等于sentence_length

                y = [self.tag2id[tag] for tag in tags]
                y = self.pad(y)

                x_data.append(x)
                y_data.append(y)

                sentence_length = min(len(words), self.sequence_lenghth)
                sentence_lengths.append(sentence_length)

                if len(y_data) == batch_size:
                    yield np.asarray(x_data), np.asarray(y_data), np.asarray(sentence_lengths)
                    x_data = []
                    y_data = []
                    sentence_lengths = []


















































