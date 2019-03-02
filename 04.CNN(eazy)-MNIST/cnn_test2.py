import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.pyplot import savefig  
import os
import numpy as np
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


x_reshape=tf.reshape(x,shape=[-1,28,28,1])
with tf.name_scope('conv1'):
    with tf.name_scope('weights'):
        wc1 = tf.Variable(tf.truncated_normal([5,5,1,32]))
    with tf.name_scope('bias'):
        bc1 = tf.Variable(tf.constant(0.1, shape=[32]))
    conv1_tmp=conv2d(x_reshape,wc1,bc1)
    conv1=maxpool2d(conv1_tmp,k=2)
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
    fc1=tf.nn.dropout(fc1,keep_prob)
    out=tf.add(tf.matmul(fc1,wout),bout)
    out=tf.nn.softmax(out)
    out=tf.reshape(tf.argmax(out,1),[])


saver = tf.train.Saver()
init=tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    saver.restore(sess,'./var.ckpt')
    conv1_tmp = sess.run(conv1_tmp,feed_dict={x:mnist.test.images[:32],y:mnist.test.labels[:32],keep_prob:1.})
    reader = tf.train.NewCheckpointReader('./var.ckpt')  
    filter = reader.get_tensor('conv1/weights/Variable')
    fig1,ax1 = plt.subplots(nrows=1, ncols=32, figsize = (32,1))	
    for i in range(31):
        ax1[i].set_xticks([])
        ax1[i].set_yticks([])
        ax1[i].imshow(filter[:,:,0,i])
    ax1[31].set_xticks([])
    ax1[31].set_yticks([])
    gci=ax1[31].imshow(filter[:,:,0,31])
    plt.colorbar(gci)
    plt.title('filter0')
    plt.show()
    savefig('./filter0.jpg')
    # visualize the weight of conv1 for each channel.
    fig2,ax2 = plt.subplots(nrows=1, ncols=32, figsize = (32,1))	
    for i in range(31):
        ax2[i].set_xticks([])
        ax2[i].set_yticks([])
        ax2[i].imshow(conv1_tmp[0,:,:,i])
    ax2[31].set_xticks([])
    ax2[31].set_yticks([])
    gci=ax2[31].imshow(conv1_tmp[0,:,:,31])
    plt.title('feature_map_t')
    plt.colorbar(gci)
    plt.show()
    savefig('./feature_map_t.jpg')
    #visualize the conv1_result of each channel for one sample.
    fig3,ax3 = plt.subplots(nrows=1, ncols=32, figsize = (32,1))	
    for i in range(31):
        ax3[i].set_xticks([])
        ax3[i].set_yticks([])
        ax3[i].imshow(conv1_tmp[i,:,:,0])
    ax3[31].set_xticks([])
    ax3[31].set_yticks([])
    gci=ax3[31].imshow(conv1_tmp[31,:,:,0])
    plt.title('feature_map_n')
    plt.colorbar(gci)
    plt.show()
    savefig('./feature_map_n.jpg')
    #visualize the first channel of conv1_result for a 32 samples.
    print("prediction:",sess.run(out,feed_dict={x:mnist.test.images[:1],
                                                           y:mnist.test.labels[:1],
                                                           keep_prob:1.}))
    print("labels:",np.argmax(mnist.test.labels[0]))
    
    
