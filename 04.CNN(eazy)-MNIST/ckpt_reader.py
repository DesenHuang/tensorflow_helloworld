import tensorflow as tf;    
import numpy as np;    
import matplotlib.pyplot as plt;    
  
reader = tf.train.NewCheckpointReader('./var.ckpt')  
  
variables = reader.get_variable_to_shape_map()  
  
for key in variables:  
    print('tensor_name:'+key)
    #print(reader.get_tensor(key))
	
#print(reader.get_tensor(key))