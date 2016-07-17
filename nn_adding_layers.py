import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def add_layer(input,in_size,out_size, activation_func = None):
    weight = tf.Variable(tf.random_normal([in_size,out_size]))
    bias = tf.Variable(tf.zeros([1,out_size])+ 0.1)
    layer = tf.matmul(input,weight)+ bias
    if activation_func is None:
        output = layer
    else:
        output = activation_func(layer)
    return output

# make some data
x_data = np.linspace(-1,1,300)[:,np.newaxis]
noise = np.random.normal(0,0.05,x_data.shape)
y_data = np.square(x_data) - 0.5 + noise


# plt.scatter(x_data,y_data)
# plt.show()

# define placeholder

xt = tf.placeholder(tf.float32,[None,1])
yt = tf.placeholder(tf.float32,[None,1])


# add hidden layer

layer1 = add_layer(xt,1,10,activation_func=tf.nn.relu)
# op layer

pred = add_layer(layer1,10,1,activation_func=None)

# objective
objective = tf.reduce_mean(tf.reduce_sum(tf.square(yt-pred), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(objective)

#initialization

init = tf.initialize_all_variables()
sess= tf.Session()
sess.run(init)

#plotting
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(x_data,y_data)
plt.ion()
plt.show()


for i in range(1000):
    sess.run(train_step,feed_dict={xt:x_data,yt:y_data})
    if i % 50 == 0:
        print(sess.run(objective,feed_dict={xt:x_data,yt:y_data}))
        #visualize the plot
        try:
            ax.lines.remove(line[0])
        except Exception:
            pass
        pred_value  = sess.run(pred,feed_dict={xt:x_data})
        line = ax.plot(x_data,pred_value, 'r-',lw=3)
        plt.pause(1)