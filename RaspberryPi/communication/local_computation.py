import numpy as np
import tensorflow as tf

filename = "../models/" + "frozen.pb" 
dics = {0:"running",1:"jumping",2:"sitting",3:"walking",4:"lying"}

while True:
	f = open("../data/subject1.txt", "r")
	for line_ in f:
		line_ = line_.strip().split(' ')
		line_ = [np.float32(i) for i in line_]
		with tf.Session() as sess:
			with tf.gfile.GFile(filename, "r") as f:
				graph_def = tf.GraphDef()
				graph_def.ParseFromString(f.read())
			new_input = tf.placeholder(tf.float32, [None, 52])

			output = tf.import_graph_def(
				graph_def,
				input_map={"input/input_x:0": new_input},
				return_elements = ['cloud_1/output:0']
			)

		sensor_output = np.reshape(line_, (1,52))
		results = sess.run(output, feed_dict={new_input: sensor_output})
		out = sess.run(tf.argmax(results[0], 1))
		print(dics[out[0]])
	f.close()