import tensorflow as tf
from collections import defaultdict

def cosine_similarity(x1, x2):
	x1_val = tf.sqrt(tf.reduce_sum(tf.matmul(x1,tf.transpose(x1)),axis=1))
	x2_val = tf.sqrt(tf.reduce_sum(tf.matmul(x2,tf.transpose(x2)),axis=1))
	denom =  tf.multiply(x1_val,x2_val)
	num = tf.reduce_sum(tf.multiply(x1,x2),axis=1)
	return tf.div(num,denom)


q_indices = tf.placeholder("int64")
q_values = tf.placeholder("int64")
x_q = tf.SparseTensor(q_indices, tf.cast(q_values, tf.float32), dense_shape=[1, 54872])

r_indices = tf.placeholder("int64")
r_values = tf.placeholder("int64")
x_r = tf.SparseTensor(r_indices, tf.cast(r_values, tf.float32), dense_shape=[1, 54872])

W_1 = tf.placeholder("float32")

l1_q  = tf.sparse_tensor_dense_matmul(x_q, W_1)
l1_r  = tf.sparse_tensor_dense_matmul(x_r, W_1)

W_2 = tf.placeholder("float32")
b_2 = tf.placeholder("float32")

l2_q  = tf.tanh(tf.matmul(l1_q, W_2) + b_2)
l2_r  = tf.tanh(tf.matmul(l1_r, W_2) + b_2)


W_4 = tf.placeholder("float32")
b_4 = tf.placeholder("float32")

y_q  = tf.tanh(tf.matmul(l2_q, W_4) + b_4)
y_r  = tf.tanh(tf.matmul(l2_r, W_4) + b_4)

similarity = cosine_similarity(y_q, y_r)[0]

init = tf.global_variables_initializer()

with tf.Session() as sess:
	rcids_array = []
	file_object = open("parents.txt", "r")
	for i, line in enumerate(file_object):
		line = line.replace(' ', '')
		line = line.replace('\n', '')
		arr = line.split(',')
		line =  [ [0, int(x)] for x in line.split(",") ]
		rcids_array.append(line)

	test_bcans_array = []
	file_object = open("test_charachter_trigraphs.txt", "r")
	for i, line in enumerate(file_object):
		line = line.replace(' ', '')
		line = line.replace('\n', '')
		arr = line.split(',')
		line =  [ [0, int(x)] for x in line.split(",") ]
		test_bcans_array.append(line)

	
	sess_2 = tf.Session()
	saver = tf.train.import_meta_graph('saved_model-19.meta')
	saver.restore(sess_2,tf.train.latest_checkpoint('./'))


	graph = tf.get_default_graph()
	W1 = sess_2.run('W1:0')
	W2 = sess_2.run('W2:0')
	b2 = sess_2.run('b2:0')
	W4 = sess_2.run('W4:0')
	b4 = sess_2.run('b4:0')
	big_k = 100

	test_bcans_names = []
	test_bcans_actual_rcids = []
	test_bcan_name_file = open("test_feb13th.txt", "r")
	for i, line in enumerate(test_bcan_name_file):
		line = line.replace('\n', '')
		bcan_name, actual_rcid_name = line.split('^')
		test_bcans_names.append(bcan_name)
		test_bcans_actual_rcids.append(actual_rcid_name)

	rcid_names = []
	word_to_rcid = defaultdict(list)
	rcid_name_file = open("base_parents.txt", "r")
	for i, line in enumerate(rcid_name_file):
		if(i > 0):
			line = line.replace('\n', '')
			rcid_name = line.split('^')[1]
			rcid_names.append(rcid_name)
			if(rcid_name != ""):
				rcid_words = rcid_name.split(" ")
				for w in rcid_words:
					word_to_rcid[w].append(i - 1)		

	stop_words = []

	output_file  = open("deep_learning_output.txt", "w")
	for bcan_index, test_bcan in enumerate(test_bcans_array):
		if(bcan_index % 100 == 0):
			print(bcan_index)
		bcan_words = test_bcans_names[bcan_index].split(" ")
		should_compute_rcids = []
		for w_b in bcan_words:
			if(w_b not in stop_words):
				if(word_to_rcid[w_b] is not None):
					for res in word_to_rcid[w_b]:
						should_compute_rcids.append(res)
		should_compute_rcids = set(should_compute_rcids)
		output_file.write(test_bcans_names[bcan_index] + "\n")
		output_file.write(test_bcans_actual_rcids[bcan_index] + "\n")
		output_file.write("********" + "\n")

		cosine_dict = dict()
		for rcid_index in should_compute_rcids:
			rcid_i = rcids_array[rcid_index]
			buckets_q = []
			for j in range(len(test_bcan)):
				buckets_q.append(1)
			buckets_r = []
			for j in range(len(rcid_i)):
				buckets_r.append(1)
			cosine_dict[rcid_index] = sess.run(similarity, feed_dict={q_indices: test_bcan,q_values: buckets_q,r_indices: rcid_i,r_values: buckets_r,W_1: W1, W_2: W2, b_2: b2, W_4: W4,b_4: b4})
		a1_sorted_keys = sorted(cosine_dict, key=cosine_dict.get, reverse=True)

		small_k = 0
		for r in a1_sorted_keys:
			if(small_k < big_k):
				output_file.write(rcid_names[r] + "\n")
				small_k += 1
		output_file.write("^^^^^^^^" + "\n")
output_file.close()