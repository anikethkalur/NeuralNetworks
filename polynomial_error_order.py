import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# Input, output and constants

# error = tf.Variable(tf.zeros([3500,1]))
num_ip = 1
num_op = 1
num_train_data = 500
hidden1 = 10
hidden2 = 10
hidden3 = 10
#hidden4 = 50

# true model
xt = np.linspace(-1,1,num_train_data)[:,np.newaxis]
yt = -2 + np.multiply(-3,xt)-2*np.square(xt)-5*np.power(xt,3)+1*np.power(xt,4)+1*np.power(xt,5)+2*np.power(xt,6)

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

layer1 = add_layer(x,num_ip,hidden1, activation_func=tf.nn.relu)
layer2 = add_layer(layer1,hidden1,hidden2, activation_func=tf.nn.relu)
layer3 = add_layer(layer2,hidden2,hidden3, activation_func=tf.nn.relu)
#layer4 = add_layer(layer3,hidden3,hidden4, activation_func=tf.nn.relu)
y = add_layer(layer3,hidden3,num_op, activation_func=None)

# Optimization part

Objective = tf.reduce_mean(tf.square(y_ - y))
Optimizer = tf.train.GradientDescentOptimizer(0.001)
train_step = Optimizer.minimize(Objective)


# Session Initialization

init = tf.initialize_all_variables()
saver = tf.train.Saver()

sess = tf.Session()
sess.run(init)

#plotting initialization details

# fig = plt.figure()
# ax = fig.add_subplot(1,1,1)
# ax.scatter(xt,yt)
# plt.ion()
# plt.show()

# loop portion
batch_size = 100
for i in range(2000):
    # for j in range(num_train_data/batch_size):
    #     x_batch= xt[batch_size*j:batch_size*j+batch_size,:]
    #     y_batch=yt[batch_size*j:batch_size*j+batch_size,:]
    # error=sess.run(train_step, feed_dict={x: x_batch, y_: y_batch})
    sess.run(train_step,feed_dict={x:xt,y_:yt})
    if i%100 == 0:
        # print(i,"\s",sess.run(Objective,feed_dict={x:x_batch,y_:y_batch}))
        print(i, "\s", sess.run(Objective, feed_dict={x: xt, y_: yt}))

        #visualize the plot
        # try:
        #     ax.lines.remove(line[0])
        # except Exception:
        #     pass
        # pred_value = sess.run(y,feed_dict={x:xt,y_:yt})
        # line = ax.plot(xt,pred_value,'r-',lw=3)
        # plt.pause(1)

# print(i, "\s", sess.run(Objective, feed_dict={x: xt, y_: yt}))

# with tf.Session as sess:
#     sess.run(init)
#     save_path = saver.save(sess,"Errors/Error_order5.ckpt")
#     print("save to :",save_path)

outh1=sess.run(layer1, feed_dict={x: xt, y_: yt})
outh2=sess.run(layer2, feed_dict={x: xt, y_: yt})
outh3=sess.run(layer3, feed_dict={x: xt, y_: yt})

plt.figure(3)
ax = plt.subplot(111)
# ax.grid(True)
ax.plot(xt, outh2[:,0],'b--')
ax.plot(xt, outh2[:,1],'k--')
ax.plot(xt, outh2[:,2],'y--')
ax.plot(xt, outh2[:,3],'g--')
ax.plot(xt, outh2[:,4],'-r')
ax.plot(xt, outh2[:,5],'r--')
ax.plot(xt, outh2[:,6],'r+')
ax.plot(xt, outh2[:,7],'-y')
ax.plot(xt, outh2[:,8],'+g')
ax.plot(xt, outh2[:,9],'-b')
plt.show()

# ax = plt.subplot(111)
# #ax.set_theta_direction(-1) #clockwise
# #ax.set_theta_offset(np.pi/2) #put 0 degrees (north) at top of plot
# #ax.yaxis.set_ticklabels([]) #hide radial tick labels
# ax.grid(True)
# #title = str(home.date)
# #ax.set_title(title, va='bottom')

# ax.scatter(xt, outh3[:,0],color='blue')
# ax.scatter(xt, outh3[:,1],color='black')
# ax.scatter(xt, outh3[:,2],color='yellow')
# ax.scatter(xt, outh3[:,3],color='green')
# ax.scatter(xt, outh3[:,4],color='red')
# ax.scatter(xt, outh3[:,5],color='red')
# ax.scatter(xt, outh3[:,6],color='red')
# ax.scatter(xt, outh3[:,7],color='red')
# ax.scatter(xt, outh3[:,8],color='red')
# ax.scatter(xt, outh3[:,9],color='red')
# #ax.set_ylim([-1,1])
# #ax.set_rmax(1.0)
# plt.show()

print('finished whole implementation')