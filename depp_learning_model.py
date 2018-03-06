
import numpy as np
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

def cosine_similarity(a, b):
  c = tf.sqrt(tf.reduce_sum(tf.multiply(a,a),axis=1))
  d = tf.sqrt(tf.reduce_sum(tf.multiply(b,b),axis=1))
  e = tf.reduce_sum(tf.multiply(a,b),axis=1)
  f = tf.multiply(c,d)
  r = tf.divide(e,f)
  return r

q_indices_array = []
p_indices_array = []
n_indices_array = []


file_object = open("train_file_deep_learning.txt", "r")
for i, line in enumerate(file_object):
  line = line.replace(' ', '')
  line = line.replace('\n', '')
  arr = line.split(',')
  line =  [ [0, int(x)] for x in line.split(",") ]
  if(i % 7 == 0):
    q_indices_array.append(line)
  elif(i % 7 == 1):
    p_indices_array.append(line)
  else:
    n_indices_array.append(line)

        

print(len(q_indices_array))
print(len(p_indices_array))
print(len(n_indices_array))
# Parameters
training_epochs = 20
display_step = 1

# tf Graph Input
q_indices = tf.placeholder("int64")
q_values = tf.placeholder("int64")
x_q = tf.SparseTensor(q_indices, tf.cast(q_values, tf.float32), dense_shape=[1, 54872])

p_indices = tf.placeholder("int64")
p_values = tf.placeholder("int64")
x_p = tf.SparseTensor(p_indices, tf.cast(p_values, tf.float32), dense_shape=[1, 54872])

n1_indices = tf.placeholder("int64")
n1_values = tf.placeholder("int64")
x_n1 = tf.SparseTensor(n1_indices, tf.cast(n1_values, tf.float32), dense_shape=[1, 54872])

n2_indices = tf.placeholder("int64")
n2_values = tf.placeholder("int64")
x_n2 = tf.SparseTensor(n2_indices, tf.cast(n2_values, tf.float32), dense_shape=[1, 54872])

n3_indices = tf.placeholder("int64")
n3_values = tf.placeholder("int64")
x_n3 = tf.SparseTensor(n3_indices, tf.cast(n3_values, tf.float32), dense_shape=[1, 54872])

n4_indices = tf.placeholder("int64")
n4_values = tf.placeholder("int64")
x_n4 = tf.SparseTensor(n4_indices, tf.cast(n4_values, tf.float32), dense_shape=[1, 54872])

n5_indices = tf.placeholder("int64")
n5_values = tf.placeholder("int64")
x_n5 = tf.SparseTensor(n5_indices, tf.cast(n5_values, tf.float32), dense_shape=[1, 54872])

W_1 = tf.Variable(tf.truncated_normal([54872, 300], mean=0, stddev= 0.1, dtype=tf.float32), name = "W1")

l1_q  = tf.sparse_tensor_dense_matmul(x_q, W_1)
l1_p  = tf.sparse_tensor_dense_matmul(x_p, W_1)
l1_n1 = tf.sparse_tensor_dense_matmul(x_n1, W_1)
l1_n2 = tf.sparse_tensor_dense_matmul(x_n2, W_1)
l1_n3 = tf.sparse_tensor_dense_matmul(x_n3, W_1)
l1_n4 = tf.sparse_tensor_dense_matmul(x_n4, W_1)
l1_n5 = tf.sparse_tensor_dense_matmul(x_n5, W_1)

W_2 = tf.Variable(tf.zeros([300, 100], tf.float32), name = "W2")
b_2 = tf.Variable(tf.truncated_normal([1, 100], mean=0, stddev= 0.1, dtype=tf.float32), name = "b2")

l2_q  = tf.tanh(tf.matmul(l1_q, W_2) + b_2)
l2_p  = tf.tanh(tf.matmul(l1_p, W_2) + b_2)
l2_n1 = tf.tanh(tf.matmul(l1_n1, W_2) + b_2)
l2_n2 = tf.tanh(tf.matmul(l1_n2, W_2) + b_2)
l2_n3 = tf.tanh(tf.matmul(l1_n3, W_2) + b_2)
l2_n4 = tf.tanh(tf.matmul(l1_n4, W_2) + b_2)
l2_n5 = tf.tanh(tf.matmul(l1_n5, W_2) + b_2)

# W_3 = tf.Variable(tf.zeros([300, 300], tf.float32))
# b_3 = tf.Variable(tf.truncated_normal([1, 300], mean=0, stddev= 0.1, dtype=tf.float32))

# l3_q  = tf.tanh(tf.matmul(l2_q, W_3) + b_3)
# l3_p  = tf.tanh(tf.matmul(l2_p, W_3) + b_3)
# l3_n1 = tf.tanh(tf.matmul(l2_n1, W_3) + b_3)
# l3_n2 = tf.tanh(tf.matmul(l2_n2, W_3) + b_3)
# l3_n3 = tf.tanh(tf.matmul(l2_n3, W_3) + b_3)
# l3_n4 = tf.tanh(tf.matmul(l2_n4, W_3) + b_3)
# l3_n5 = tf.tanh(tf.matmul(l2_n5, W_3) + b_3)

W_4 = tf.Variable(tf.truncated_normal([100, 10], mean=0, stddev= 0.1, dtype=tf.float32), name="W4")
b_4 = tf.Variable(tf.truncated_normal([1, 10], mean=0, stddev= 0.1, dtype=tf.float32), name="b4")

y_q  = tf.tanh(tf.matmul(l2_q, W_4) + b_4)
y_p  = tf.tanh(tf.matmul(l2_p, W_4) + b_4)
y_n1 = tf.tanh(tf.matmul(l2_n1, W_4) + b_4)
y_n2 = tf.tanh(tf.matmul(l2_n2, W_4) + b_4)
y_n3 = tf.tanh(tf.matmul(l2_n3, W_4) + b_4)
y_n4 = tf.tanh(tf.matmul(l2_n4, W_4) + b_4)
y_n5 = tf.tanh(tf.matmul(l2_n5, W_4) + b_4)

r_p  = cosine_similarity(y_q, y_p)
r_n1 = cosine_similarity(y_q, y_n1)
r_n2 = cosine_similarity(y_q, y_n2)
r_n3 = cosine_similarity(y_q, y_n3)
r_n4 = cosine_similarity(y_q, y_n4)
r_n5 = cosine_similarity(y_q, y_n5)

sum_r = tf.exp(r_p) + tf.exp(r_n1) + tf.exp(r_n2) + tf.exp(r_n3) + tf.exp(r_n4) + tf.exp(r_n5)

prob_p_doc = tf.div(tf.exp(r_p), sum_r)
logloss = -tf.reduce_sum(tf.log(prob_p_doc))
optimizer = tf.train.AdamOptimizer(0.0001).minimize(logloss)


# Initializing the variables
init = tf.global_variables_initializer()

saver = tf.train.Saver()
# Launch the graph
with tf.Session() as sess:
  sess.run(init)
  # Training cycle
  for epoch in range(training_epochs):
    i = 0
    ll_overall = 0
    while(i < 26333):
      buckets_q = []
      for j in range(len(q_indices_array[i])):
        buckets_q.append(1)
      buckets_p = []
      for j in range(len(p_indices_array[i])):
        buckets_p.append(1)
      buckets_n1 = []
      for j in range(len(n_indices_array[i * 5])):
        buckets_n1.append(1)
      buckets_n2 = []
      for j in range(len(n_indices_array[i * 5 + 1])):
        buckets_n2.append(1)
      buckets_n3 = []
      for j in range(len(n_indices_array[i * 5 + 2])):
        buckets_n3.append(1)
      buckets_n4 = []
      for j in range(len(n_indices_array[i * 5 + 3])):
        buckets_n4.append(1)
      buckets_n5 = []
      for j in range(len(n_indices_array[i * 5 + 4])):
        buckets_n5.append(1)

      _, ll = sess.run([optimizer, logloss], feed_dict={q_indices: q_indices_array[i],
                                                      q_values: buckets_q,
                                                      p_indices: p_indices_array[i],
                                                      p_values: buckets_p,
                                                      n1_indices: n_indices_array[i * 5],
                                                      n1_values: buckets_n1,
                                                      n2_indices: n_indices_array[i * 5 + 1],
                                                      n2_values: buckets_n2,
                                                      n3_indices: n_indices_array[i * 5 + 2],
                                                      n3_values: buckets_n3,
                                                      n4_indices: n_indices_array[i * 5 + 3],
                                                      n4_values: buckets_n4,
                                                      n5_indices: n_indices_array[i * 5 + 4],
                                                      n5_values: buckets_n5})
      ll_overall += ll
      i += 1
    ll_overall /= 26333
    saver.save(sess, './saved_model', global_step=epoch)
    print(ll_overall)

