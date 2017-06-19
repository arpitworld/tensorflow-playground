'''
Sample linear regression implementation
Predict Flight prices given the distance covered by the flight.
'''

import urllib.request
import numpy as np
import tensorflow as tf

# Plot
import matplotlib.pyplot as plt
fig = plt.figure()
fig1 = fig.add_subplot(121)
fig1.set_title('Data')
fig1.set_xlabel('Flight Distance')
fig1.set_ylabel('Flight Price')
fig2 = fig.add_subplot(122)
fig2.set_title('Cost Function')
fig2.set_xlabel('Iterations')
fig2.set_ylabel('Cost Function')

# Getting Data
f = urllib.request.urlopen("http://www.stat.ufl.edu/~winner/data/airq402.dat")
d = f.readlines()

# Raw Data
x_data = [float(r.strip()[19:23]) for r in d]
y_data = [float(r.strip()[68:]) for r in d]

# plot Input Data
fig1.plot(x_data,y_data,'ro')

# Feature scaling and mean normalization
x = (x_data - np.mean(x_data))/(np.max(x_data) - np.min(x_data))
y = (y_data - np.mean(y_data))/(np.max(y_data) - np.min(y_data))

# weights or theta
theta_0 = tf.Variable(1.0)
theta_1 = tf.Variable(1.0)

# hypothesis (The Model) 
h = theta_0 + theta_1*x 

# Model after taking care of Feature scaling and mean normalization - we'll use this for plotting and final prediction
h_data =  ( (theta_0 + theta_1*x)*(np.max(y_data) - np.min(y_data)) ) + np.mean(y_data) 

# Cost Function
cost_function = (1/2)*tf.reduce_mean(tf.square(h-y))

# Gradient Descent
optimizer = tf.train.GradientDescentOptimizer(0.1)
train = optimizer.minimize(cost_function)

# init
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)


train_data = []
for step in range(50):
    evals = sess.run([cost_function, train,theta_0,theta_1])
    evals.insert(0,step)
    train_data.append(evals)

# Plot Model
eval_h_data = sess.run(h_data)
fig1.plot(x_data, eval_h_data)

# Plot cost function as a function of number of iterations
fig2.plot([i[0] for i in train_data], [i[1] for i in train_data])

plt.show()
