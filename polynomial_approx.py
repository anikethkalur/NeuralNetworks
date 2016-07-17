import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# Input, output and constants

num_ip = 1
num_op = 1
num_train_data = 500
hidden1 = 5000
hidden2 = 1000
hidden3 = 100

# true model
xt = np.linspace(-2,2,num_train_data)[:,np.newaxis]
yt = -2 + np.multiply(-3,xt)-2*np.square(xt)-5*np.power(xt,3)+1*np.power(xt,4)+1*np.power(xt,5)

plt.scatter(xt,yt)

# add hidden layer function definition

def add_layer(input, in_size, out_size, activation_func = None):
    # weight = tf.Variable(tf.random_([in_size, out_size]))
    weight = tf.Variable(tf.random_normal([in_size,out_size],0,0.05))
    bias = tf.Variable(tf.zeros([1,out_size]))
    #layer = tf.matmul(input,weight)+bias
    layer = tf.matmul(tf.cast(input,tf.float32), tf.cast(weight,tf.float32)) + bias
    if activation_func is None:
        output = layer
    else:
        output = activation_func(layer)
    return output

# Define NN variables and place holders

x  = tf.placeholder(tf.float32,[None,1])
y_ = tf.placeholder(tf.float32,[None,1])

# Define NN layers and overall definition

layer1 = add_layer(xt,num_ip,hidden1, activation_func=tf.nn.relu)
layer2 = add_layer(layer1,hidden1,hidden2, activation_func=tf.nn.relu)
layer3 = add_layer(layer2,hidden2,hidden3, activation_func=tf.nn.relu)
y = add_layer(layer3,hidden3,num_op, activation_func=None)

# Optimization part

Objective = tf.reduce_mean(tf.square(y_ - y))
Optimizer = tf.train.GradientDescentOptimizer(0.01)
train_step = Optimizer.minimize(Objective)

# Session Initialization

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

#plotting initialization details

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(xt,yt)
plt.ion()
plt.show()

# loop portion
# batch_size = 100
for i in range(1000):
    sess.run(train_step,feed_dict={x:xt,y_:yt})
    if i%50 == 0:
        print(sess.run(Objective,feed_dict={x:xt,y_:yt}))
        #visualize the plot
        try:
            ax.lines.remove(line[0])
        except Exception:
            pass
        pred_value = sess.run(y,feed_dict={x:xt,y_:yt})
        line = ax.plot(xt,pred_value,'r-',lw=3)
        plt.pause(1)
