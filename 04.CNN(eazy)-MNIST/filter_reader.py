import tensorflow as tf; 
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.pyplot import savefig   
import numpy as np;    
import matplotlib.pyplot as plt;    
  
reader = tf.train.NewCheckpointReader('./var.ckpt')  
filter = reader.get_tensor('conv1/weights/Variable')


fig1,ax1 = plt.subplots(nrows=1, ncols=32, figsize = (32,1))	
for i in range(32):
    ax1[i].set_xticks([])
    ax1[i].set_yticks([])
    ax1[i].imshow(filter[:,:,0,i])
plt.title('filter0')
plt.show()
savefig('./filter0.jpg')
