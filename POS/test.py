from DataProcessing.dataset import *
import tensorflow as tf
datahelper = DataHelper()

train_data = datahelper.generate_datas(stage = 'train', char_base = True)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for sentences, labels, sentence_lengths in train_data:
        print(sentences)