import tensorflow as tf
import os
from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets('MNIST_data/',one_hot=True)

n_input=784
n_classes=10
dropout=0.75

def conv2d(x,W,b,strides=1):
    x=tf.nn.conv2d(x,W,strides=[1,strides,strides,1],padding='SAME')
    x=tf.nn.bias_add(x,b)
    return tf.nn.relu(x)

def maxpool2d(x,k=2):
    return tf.nn.max_pool(x,ksize=[1,k,k,1],strides=[1,k,k,1],padding='SAME')

with tf.name_scope('input_layer'):
    x=tf.placeholder(tf.float32,[None,n_input])
    y=tf.placeholder(tf.float32,[None,n_classes])
    keep_prob=tf.placeholder(tf.float32)

def conv_net(x,dropout):
    x=tf.reshape(x,shape=[-1,28,28,1])
    with tf.name_scope('conv1'):
        with tf.name_scope('weights'):
            wc1 = tf.Variable(tf.truncated_normal([5,5,1,32]))
        with tf.name_scope('bias'):
            bc1 = tf.Variable(tf.constant(0.1, shape=[32]))
        conv1=conv2d(x,wc1,bc1)
        conv1=maxpool2d(conv1,k=2)
    with tf.name_scope('conv2'):
        with tf.name_scope('weights'):
            wc2 = tf.Variable(tf.truncated_normal([5,5,32,64]))
        with tf.name_scope('bias'):
            bc2 = tf.Variable(tf.constant(0.1, shape=[64]))
        conv2=conv2d(conv1,wc2,bc2)
        conv2=maxpool2d(conv2,k=2)
	
    with tf.name_scope('fc'):
        with tf.name_scope('weights'):
            wd1 = tf.Variable(tf.truncated_normal([7*7*64,1024]))
        with tf.name_scope('bias'):
            bd1 = tf.Variable(tf.constant(0.1, shape=[1024]))
        fc1=tf.reshape(conv2,[-1,wd1.get_shape().as_list()[0]])
        fc1=tf.add(tf.matmul(fc1,wd1),bd1)
        fc1=tf.nn.relu(fc1)
    
    with tf.name_scope('out'):
        with tf.name_scope('weights'):
            wout = tf.Variable(tf.truncated_normal([1024,n_classes]))
        with tf.name_scope('bias'):
            bout = tf.Variable(tf.constant(0.1, shape=[n_classes]))
        fc1=tf.nn.dropout(fc1,dropout)
        out=tf.add(tf.matmul(fc1,wout),bout)
        out=tf.nn.softmax(out)
    return out


pred=conv_net(x,keep_prob)

with tf.name_scope('accuracy'):
    #pred=tf.nn.softmax(pred)
    correct_pred=tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
    accuracy=tf.reduce_mean(tf.cast(correct_pred,tf.float32))
saver = tf.train.Saver()
init=tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    saver.restore(sess,'./var.ckpt')
    print("testing accuracy:",sess.run(accuracy,feed_dict={x:mnist.test.images[:1000],
                                                           y:mnist.test.labels[:1000],
                                                           keep_prob:1.}))
    
    
