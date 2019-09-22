
import tensorflow as tf
import numpy as np
#import matplotlib.pyplot as plt

def generate_dataset():
	x_batch = np.linspace(0, 2, 100)
	y_batch = 1.5 * x_batch + np.random.randn(*x_batch.shape) * 0.2 + 0.5
	return [1,2,3,4],[3,5,7,9] 

x_batch, y_batch = generate_dataset()

x = tf.placeholder(tf.float32, shape=(None, ), name='x')
y = tf.placeholder(tf.float32, shape=(None, ), name='y')
w = tf.Variable(np.random.normal(), name='W')
b = tf.Variable(np.random.normal(), name='b')
y_pred = tf.add(tf.multiply(w, x), b)
loss = tf.reduce_mean(tf.square(y_pred - y))
optimizer = tf.train.GradientDescentOptimizer(0.1)
train_op = optimizer.minimize(loss)
session=tf.Session()
session.run(tf.global_variables_initializer())
feed_dict = {x: x_batch, y: y_batch}
print("Value of W befor: "+ str(session.run(w) ))
print("Value of b before: "+ str(session.run(b)))
		
for i in range(1000):
	session.run(train_op, feed_dict)
        if i%100==0:
		print(i, "loss:", session.run(loss,feed_dict=feed_dict))

print('Predicting')
y_pred_batch = session.run(y_pred, {x : [100]})
print(y_pred_batch)
print(session.run(w))
print(session.run(b))
#plt.scatter(x_batch, y_batch)
#plt.plot(x_batch, y_pred_batch, color='red')
#plt.xlim(0, 2)
#plt.ylim(0, 2)
#plt.savefig('plot.png')
#plt.show()
