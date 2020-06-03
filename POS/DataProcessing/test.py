import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
#batch_size = 2
# labels = tf.constant([[0, 0, 0, 1],[0, 1, 0, 0]])
# logits = tf.constant([[-3.4, 2.5, -1.2, 5.5],[-3.4, 2.5, -1.2, 5.5]])
# labels=tf.argmax(labels,1)
labels = [0,1,3,5,6,34,5,6,3]
pre = [0,1,3,5,6,34,5,6,30]
# loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
# loss_s = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(labels,1), logits=logits)
correct = tf.reduce_mean(tf.cast(x=tf.equal(labels,pre),dtype=tf.float32))
global_step = tf.Variable(0, name="global_step", trainable=False)
with tf.Session() as sess:
    print(global_step)


#
#     print ("softmax loss:", sess.run(loss))
#     print ("sparse softmax loss:", sess.run(loss_s))

