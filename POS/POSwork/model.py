
import tensorflow as tf
import tensorflow.contrib.rnn as rnn


class PosModel():
    '''Part of Speech tagging'''

    def __init__(self, num_tags, vocab_size, lr=0.001, l2_reg_lambda=0.001,
                 batch_size=64, embedding_dim=200, sequence_length=100,
                 pre_embeddings=None, pre_embedding_trainable=True):
        self.num_tags = num_tags
        self.vocab_size = vocab_size
        self.lr = lr
        self.l2_reg_lambda = l2_reg_lambda
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.embedding_dim = embedding_dim

        self.sentences = tf.placeholder(tf.int32, [None, sequence_length], name='input_sentences')
        self.labels = tf.placeholder(tf.int32, [None, sequence_length], name='input_labels')
        self.seq_length = tf.placeholder(tf.int32, [None, ], name='seq_length')

        # Embedding layer   给每个word训练一个embedding
        with tf.name_scope("embedding"):
            self.words_embedding = tf.Variable(tf.constant
                                               (0.0, shape=[self.vocab_size, self.embedding_dim]),
                                               name='encoder_embedding')

    def bi_lstm_encode(self, sentences, seq_length):
        embedded_sentence = tf.nn.embedding_lookup(self.words_embedding, sentences)
        # [batch_size, sequence_length, embedding_dim]

        hidden_dim = 32
        with tf.variable_scope("bilstm"):
            cell_fw = rnn.LSTMCell(hidden_dim)
            cell_bw = rnn.LSTMCell(hidden_dim)
            (outputs, output_states) = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cell_fw,
                cell_bw=cell_bw,
                inputs=embedded_sentence,
                sequence_length=seq_length,
                dtype=tf.float32)
            output = tf.concat(outputs, axis=2)  # [batch_size,sequence_length,hidden_dim*2]
            output = tf.nn.dropout(output, 0.8)

        W = tf.get_variable(name='W', shape=[2 * hidden_dim, self.num_tags],
                            initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)

        b = tf.get_variable(name='b', shape=[self.num_tags],
                            initializer=tf.zeros_initializer, dtype=tf.float32)

        output = tf.reshape(output, [-1, 2 * hidden_dim])  # [batch_size*sequence_length,hidden_dim*2]
        pred = tf.matmul(output, W) + b
        return pred

    def build(self, use_crf=False):
        logits = self.bi_lstm_encode(self.sentences, self.seq_length)

        self.loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=tf.reshape(self.labels, [-1]),  # [batch_size*sequence_length]
                logits=logits  # [batch_size*sequence_length,num_tags]
            ))

        prediction = tf.cast(tf.argmax(logits, axis=1), tf.int32)  # [batch_size*sequence_length]
        self.prediction = tf.reshape(prediction, [-1, self.sequence_length])

        correct = tf.equal(self.prediction, self.labels)
        self.accuracy = tf.reduce_mean(tf.cast(correct, dtype=tf.float32), name='accuracy')

        self.global_step = tf.Variable(0, name='global_step', trainable=False)

        self.train_op = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss, global_step=self.global_step)












