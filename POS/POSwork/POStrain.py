import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
import tensorflow.contrib.rnn as rnn
tf.reset_default_graph()
from dataset import *
from model import *

data_helper = DataHelper()

with tf.Session() as sess:
  model = PosModel(num_tags = data_helper.num_tags, vocab_size = data_helper.vocab_size)
  model.build()
  sess.run(tf.global_variables_initializer())

  train_data = data_helper.generate_datas(stage = 'train')
  valid_data = data_helper.generate_datas(stage = 'valid')

  for sentences, labels, sentence_lengths in train_data:
    step, prediction, loss, acc, _ = sess.run(
            [model.global_step,model.prediction,model.loss, model.accuracy, model.train_op ],
            feed_dict = { model.sentences: sentences,
                                model.labels: labels,
                                model.seq_length: sentence_lengths})

    if step % 100 == 0:
      valid_correct = []

      for sentences, labels, sentence_lengths in valid_data:
        loss, acc = sess.run([model.loss, model.accuracy],
                              feed_dict={model.sentences: sentences,
                                        model.labels: labels,
                                        model.seq_length: sentence_lengths})
        
        print('验证集的loss为：%3f，准确率为：%3f' %(loss, acc))

  print('==============================================完成==============================================')




























