import tensorflow as tf
import numpy
import time
import matplotlib.pyplot as plt
import numpy as np

sess = tf.Session()

# Parameters
learning_rate = 0.1
training_epochs = 10000
batch_size = 1

# Network Parameters
n_input = 3 # Data input
n_hidden_1 = 5 # 1st layer num features
n_hidden_2 = 5 # 2nd layer num features
n_output = 1 # Data output

# tf Graph input
x = tf.placeholder("float", [None, 2], "a")
y = tf.placeholder("float", [None, 1], "b")
lab = tf.placeholder(tf.int32, [None], "h")

# Create model
def multilayer_perceptron_l2(_X, _weights, _biases):
    layer_1 = tf.nn.relu(tf.add(tf.matmul(_X, _weights['w1']), _biases['b1'])) #Hidden layer with RELU activation
    layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, _weights['w2']), _biases['b2'])) #Hidden layer with RELU activation
    pred = tf.matmul(layer_2, _weights['w3']) + _biases['b3']
    euclidean_mean = tf.reduce_mean(tf.nn.l2_loss(pred-y))
    cost = euclidean_mean
    return pred, cost

def multilayer_perceptron2_crossent(_X, _weights, _biases):
    layer_1 = tf.nn.relu(tf.add(tf.matmul(_X, _weights['w1']), _biases['b1'])) #Hidden layer with RELU activation
    layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, _weights['w2']), _biases['b2'])) #Hidden layer with RELU activation
    prob =  tf.matmul(layer_2, _weights['w32']) + _biases['b32']
    pred = tf.argmax(prob, 1)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(prob, lab, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    cost = cross_entropy_mean
    return pred, cost

# Store layers weight & bias
weights = {
    'w1' : tf.get_variable('W1', shape=[2, 5], dtype=tf.float32, initializer=tf.truncated_normal_initializer(1e-4)),
    'w2' : tf.get_variable('W2', shape=[5, 5], dtype=tf.float32, initializer=tf.truncated_normal_initializer(1e-4)),
    'w3' : tf.get_variable('W3', shape=[5, 1], dtype=tf.float32, initializer=tf.truncated_normal_initializer(1e-4)),
    'w32' : tf.get_variable('W32', shape=[5, 2], dtype=tf.float32, initializer=tf.truncated_normal_initializer(1e-4))
}
biases = {
    'b1' : tf.get_variable('b1', shape=[5], dtype=tf.float32, initializer=tf.truncated_normal_initializer(1e-4)),
    'b2' : tf.get_variable('b2', shape=[5], dtype=tf.float32, initializer=tf.truncated_normal_initializer(1e-4)),
    'b3' : tf.get_variable('b3', shape=[1], dtype=tf.float32, initializer=tf.truncated_normal_initializer(1e-4)),
    'b32' : tf.get_variable('b32', shape=[2], dtype=tf.float32, initializer=tf.truncated_normal_initializer(1e-4))
}

# Construct model
if False:
    pred, cost = multilayer_perceptron_l2(x, weights, biases)
else:
    pred, cost = multilayer_perceptron2_crossent(x, weights, biases)

# Define loss and optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost) # Adam Optimizer

# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
sess.run(init)

# Training Data
train_X = np.load('pts.npy')
train_label = np.load('labels.npy')
train_Y = np.expand_dims(train_label.astype(np.float32), 1)

# Training cycle
start = time.clock()
for epoch in range(training_epochs):
    # Fit training using batch data
    c, _ = sess.run([cost, optimizer], feed_dict={x: train_X, y: train_Y, lab:train_label})

    if epoch % 10 == 0:
        format_str = 'epoch %d, loss = %.2f '
        print format_str % (epoch, c)

end = time.clock()

print end - start #2.5 seconds -> 400 epochs per second 
print "Optimization Finished!"

# See result
gx = np.linspace(-1, 1, 50)
gy = np.linspace(-1, 1, 50)
xv, yv = np.meshgrid(gx, gy)
query_pts = np.concatenate([xv.reshape((-1,1)), yv.reshape((-1,1))], axis=1)

prob = np.zeros((query_pts.shape[0]), dtype=np.float32)
for r in xrange(query_pts.shape[0]):
    prob[r] = sess.run(pred, feed_dict={x:query_pts[[r],:]})

decision = prob<0.5
plt.plot(query_pts[decision,0], query_pts[decision,1], '.', markersize=20.0, markerfacecolor=[1, 0, 0, 0.2], markeredgewidth=0.0)
plt.plot(query_pts[np.logical_not(decision),0], query_pts[np.logical_not(decision),1], '.', markersize=20.0, markerfacecolor=[0, 1, 0, 0.2], markeredgewidth=0.0)

plt.scatter(train_X[train_label==0,0], train_X[train_label==0,1], s=100, facecolor=[1, 0, 0])
plt.scatter(train_X[train_label==1,0], train_X[train_label==1,1], s=100, facecolor=[1, 1, 0])
plt.show()