
# pip3 install --upgrade
# https://storage.googleapis.com/tensorflow/mac/tensorflow-0.6.0-py3-none-any.whl
# %%
import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
import numpy as np
import matplotlib.pyplot as plt
import math
import matplotlib as mpl
sess = tf.InteractiveSession()
n_input = 2#784
n_output = 2#10
hidden1 = 1000#10
hidden2 = 100#10
hidden3 = 50#10
n_train=5000
n_input_states=2
n_output_states=2

mu=[500.0,0]
sig_p=[np.power(5.0,2),np.power(25*np.pi/180.0,2)]
P=np.diag(sig_p)
xt=np.transpose(np.random.multivariate_normal(mu, P,n_train))
yt=np.zeros([n_output_states,n_train])
yt[0,:]=xt[0,:]*np.cos(xt[1,:])
yt[1,:]=xt[0,:]*np.sin(xt[1,:])

# xt -= np.mean(xt, axis = 0)
# xt /= np.std(xt, axis = 0)

#net_input = tf.placeholder(tf.float32, [None, n_input])
net_input = tf.placeholder("float")
h = tf.placeholder("float")
def weight_variable(shape,N):
  import numpy as npc
  #initial = tf.truncated_normal(shape, stddev=2.0/np.sqrt(N))
  initial = (tf.random_uniform(shape,-1.0 / math.sqrt(N),1.0 / math.sqrt(N)))
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)
# %% We can write a simple regression (y = W*x + b) as:
b1 = bias_variable([hidden1])
b2 = bias_variable([hidden2])
b3 = bias_variable([hidden3])
bout = bias_variable([n_output])

W1 = weight_variable([n_input, hidden1],n_input*hidden1)
W2 = weight_variable([hidden1, hidden2],hidden2*hidden1)
W3 = weight_variable([hidden2, hidden3],hidden2*hidden3)
Wout = weight_variable([hidden3, n_output],hidden3*n_output)

# W1 = tf.Variable(tf.zeros([n_train,n_train]))
# W2 = tf.Variable(tf.zeros([n_train,n_train]))
# b1 = tf.Variable(tf.zeros([n_train]))
# b2 = tf.Variable(tf.zeros([n_train]))

#h = tf.nn.softmax(tf.matmul(net_input, W1) + b1)

# simple one layer
# h1 = tf.nn.tanh(tf.matmul(net_input, W1) + b1)
# net_output = tf.matmul(h1,W2) + b2

h1 = tf.nn.relu(tf.matmul(net_input, W1) + b1)
h2 = tf.nn.relu(tf.matmul(h1, W2) + b2)
h3 = tf.nn.relu(tf.matmul(h2, W3) + b3)
net_output =tf.matmul(h3, Wout) + bout

# %% We'll create a placeholder for the true output of the network
y_true = tf.placeholder("float")
 
# %% And then write our loss function:
# v=0.0
v=0.0
# for ii in range(50):
#     for jj in range(50):
#         if ii==jj:
#             v+=tf.reduce_mean(h[:,ii]*h[:,jj]-1)
#         if not(ii==jj):
#             v+=tf.reduce_mean(h[:,ii]*h[:,jj])     

# atemp=tf.square(tf.matmul(tf.transpose(h3),h3)/500-np.eye(hidden))
# btemp=tf.matmul(atemp,tf.ones((hidden3,1)) )
# v=0.0*tf.matmul(tf.ones((1,hidden3)),btemp )

cross_entropy = tf.reduce_mean(tf.square(net_output - y_true))

# %% We can tell the tensorflow graph to train w/ gradient descent using
# our loss function and an input learning rate
optimizer = tf.train.GradientDescentOptimizer(0.001).minimize(cross_entropy)
#optimizer=tf.train.AdamOptimizer(1e-2).minimize(cross_entropy)
# %% We now create a new session to actually perform the initialization the
# variables:
sess = tf.Session()
sess.run(tf.initialize_all_variables())
#print(sess.run(b1))
#print(sess.run(accuracy,feed_dict={net_input: xt.reshape(200,1), y_true: yt.reshape(200,1)}))
# print(sess.run(cross_entropy, feed_dict={net_input: xt.reshape(n_train,2), y_true: yt.reshape(n_train,2)}))
# fig=plt.figure()

# %% Now actually do some training:
batch_size = 100
n_epochs = 120
for j in range(n_epochs):
    for i in range(n_train/batch_size): 
        xb=xt[:,batch_size*i:batch_size*i+batch_size].reshape(batch_size,n_input)
        yb=yt[:,batch_size*i:batch_size*i+batch_size].reshape(batch_size,n_output)
        sess.run(optimizer, feed_dict={net_input: xb, y_true: yb})
    print(sess.run(cross_entropy, feed_dict={net_input: xb, y_true: yb}))
    aout=sess.run(cross_entropy, feed_dict={net_input: xt.reshape(n_train,2), y_true: yt.reshape(n_train,2)})
    plt.scatter(i,aout,color='blue')
    plt.xlabel('Angle [rad]')
    plt.ylabel('sin(x)')
    plt.axis('tight')
    #aout=sess.run(net_output, feed_dict={net_input: xb, y_true: yb})
    #ax = plt.subplot(111)
    #ax.set_theta_direction(-1) #clockwise
    #ax.set_theta_offset(np.pi/2) #put 0 degrees (north) at top of plot
    #ax.yaxis.set_ticklabels([]) #hide radial tick labels
    # ax.grid(True)
    # #title = str(home.date)
    # #ax.set_title(title, va='bottom')
    # ax.scatter(xt, yt,color='blue')
    # ax.scatter(xt, aout,color='red')
    # ax.set_ylim([-1,1])
    # #ax.set_rmax(1.0)
    # plt.pause(0.05)


# aout=sess.run(net_output, feed_dict={net_input: xt.reshape(n_train,2), y_true: yt.reshape(n_train,2)})
#
# plt.scatter(aout[:,0],aout[:,1],color='blue')
# plt.xlabel('Angle [rad]')
# plt.ylabel('sin(x)')
# plt.axis('tight')
# #plt.show()

plt.scatter(xt[0,:],xt[1,:])
plt.xlabel('Angle [rad]')
plt.ylabel('sin(x)')
plt.axis('tight')
plt.show()

