import sys
sys.path.append('../')

from utils.utils import *
from utils.preliminaries import *
from data_reading.read_data import *
from lib.hierarchical_neural_networks import *
from smart_watch import *

from sklearn.preprocessing import normalize

def get_output(name, x, keepprob, connection_num, features_index=None):
    if name == "FullyConnectedMLP":
        output = LocalSensorNetwork("MLP", x, [256, 256, 100, 3],  keep_prob=keepprob).build_layers()

    elif name == "HierarchyAwareMLP":
        sensor_outputs = []
        for key, value in features_index.iteritems():
            with tf.variable_scope(key):
                sensor_output = LocalSensorNetwork(key, x[:,min(value):max(value)+1], [256, connection_num], keep_prob = keepprob)
                sensor_outputs.append(sensor_output)

        cloud = CloudNetwork("cloud", [256, 100, 3], keep_prob=keepprob)
        output = cloud.connect(sensor_outputs)
    return output

def main(log_dir, arch , train_data, train_labels, \
	test_data, test_labels, \
	validation_data, validation_labels, \
	l2, \
	keepprob, \
	connection_num, \
	starter_learning_rate, \
	subject):

	tf.reset_default_graph()   
	n_features = train_data.shape[1]
	with tf.name_scope('input'):
	    x = tf.placeholder(tf.float32, [None, train_data.shape[1]])
	    y_ = tf.placeholder(tf.int32, [None, 3])
	    keep_prob = tf.placeholder(tf.float32)

	output = get_output(arch, x, keep_prob, connection_num)

	training_epochs = 100
	batch_size = 256
	with tf.name_scope('cross_entropy'):

	    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=output))   
	    l2_loss = sum(
	        tf.nn.l2_loss(tf_var)
	            for tf_var in tf.trainable_variables()
	                if not ("noreg" in tf_var.name or "bias" in tf_var.name))

	    total_loss = cross_entropy + l2 * l2_loss

	global_step = tf.Variable(0, trainable=False)
	learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
	                                       1000, 0.9, staircase=True)

	with tf.name_scope('adam_optimizer'):
	    train_step = tf.train.AdamOptimizer(learning_rate).minimize(total_loss, global_step=global_step)
	    #train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(total_loss, global_step=global_step)


	with tf.name_scope('accuracy'):
	    correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(y_, 1))
	    correct_prediction = tf.cast(correct_prediction, tf.float32)
	    accuracy = tf.reduce_mean(correct_prediction)

	saver = tf.train.Saver()
	train_cross_entropy_writer = tf.summary.FileWriter(log_dir  + "/train")
	test_cross_entropy_writer = tf.summary.FileWriter(log_dir  + "/test")
	validation_cross_entropy_writer = tf.summary.FileWriter(log_dir + "/validation")
	tf.summary.scalar("cross_entropy", cross_entropy)
	tf.summary.scalar("total_loss", total_loss)
	tf.summary.scalar("accuracy", accuracy)
	write_op = tf.summary.merge_all()

	checkpoint_file = os.path.join(log_dir + "/model_checkpoints", 'model.ckpt')
	with tf.Session() as sess:
	    sess.run(tf.global_variables_initializer())
	    batch_count = train_data.shape[0] / batch_size
	    validation_loss_last_epoch = None
	    last_test_cross_entropy = None

	    patience = 5
	    for epoch in range(training_epochs):
			print epoch
			# number of batches in one epoch
			idxs = np.random.permutation(train_data.shape[0]) #shuffled ordering
			X_random = train_data[idxs]
			Y_random = train_labels[idxs]

			for i in range(batch_count):
			    train_data_batch = X_random[i * batch_size: (i+1) * batch_size,:]
			    train_label_batch = Y_random[i * batch_size: (i+1) * batch_size]
			    _ = sess.run([train_step], feed_dict={x: train_data_batch, y_: train_label_batch, keep_prob: keepprob})

			summary = sess.run(write_op, feed_dict={x: train_data, y_: train_labels, keep_prob: 1.0})
			train_cross_entropy_writer.add_summary(summary, epoch)
			train_cross_entropy_writer.flush()

			summary = sess.run(write_op, feed_dict={x: test_data, y_: test_labels, keep_prob: 1.0})
			test_cross_entropy_writer.add_summary(summary, epoch)
			test_cross_entropy_writer.flush()

			summary = sess.run(write_op, feed_dict={x: validation_data, y_: validation_labels, keep_prob: 1.0})
			validation_cross_entropy_writer.add_summary(summary, epoch)
			validation_cross_entropy_writer.flush()

			test_acc = sess.run(accuracy,
				feed_dict={x: test_data, y_: test_labels, keep_prob: 1.0})
			train_acc = sess.run(accuracy,
				feed_dict={x: train_data, y_: train_labels, keep_prob: 1.0})

			print train_acc
			print test_acc
           
            if epoch % 20 == 0:
                saver.save(sess, checkpoint_file, global_step=epoch)
            
            '''
            val_loss = sess.run(total_loss,
                feed_dict={x: validation_data, y_: validation_labels, keep_prob: 1.0})
            
            if validation_loss_last_epoch == None:
                validation_loss_last_epoch = val_loss
            else:
                if val_loss < validation_loss_last_epoch:
                    patience = 5
                    validation_loss_last_epoch = val_loss
                    saver.save(sess, checkpoint_file, global_step=epoch)
                else:
                    patience = patience - 1
                    if patience == 0:
                        break
            '''

if __name__ == '__main__':
	argv = sys.argv[1:]
	    
	task = "basic_activities" 
	task_mapping = {"basic_activities": label_mapping, "kitchen_activities": kitchen_label_mapping}

	RawData  = RawDataDigester("../../data/05-14-2018/MQTT_Messages.txt")	
	watch_df = Smartwatch(RawData.get_watch_data()).toDataFrame()

	label_pd = read_labels("../../data/05-14-2018/labels.txt")

	watch_df = watch_df[(watch_df.TimeStamp >= label_pd['TimeStamp'].iloc[0]) & (watch_df.TimeStamp <= label_pd['TimeStamp'].iloc[-1])].reset_index(drop=True)
	watch_label_df = watch_df.join(label_pd.set_index('TimeStamp'), on='TimeStamp', how='outer')

	watch_label_df = watch_label_df.replace({task: task_mapping[task]})

	X = watch_label_df[['SmartWatchHeartRate','SmartWatchAcc_X','SmartWatchAcc_Y','SmartWatchAcc_Z', \
						'SmartWatchGyro_X', 'SmartWatchGyro_Y', 'SmartWatchGyro_Z']].astype('float').as_matrix()
	y = watch_label_df[task].astype('float').as_matrix()
	
	#X = normalize(X, axis=0, norm='max')
 	X = (X - X.mean(axis=0)) / X.std(axis=0)

	feature_extractor = FeatureExtractor(X, y)
	X, y = feature_extractor.get_extracted_features()

	train_data, test_data, train_labels, test_labels = train_test_split(X,y, test_size=0.1)

	print train_data.shape

	# for testing
	validation_data = test_data
	validation_labels = test_labels

	train_labels = train_labels.astype(int)
	test_labels = test_labels.astype(int)
	validation_labels = validation_labels.astype(int)
	# convert to one-hot encoding
	train_labels = np.eye(8)[train_labels].reshape(train_labels.shape[0], 8)
	test_labels = np.eye(8)[test_labels].reshape(test_labels.shape[0], 8)
	validation_labels = np.eye(8)[validation_labels].reshape(validation_labels.shape[0], 8)


	NetWorkGrid = {'l2_reg': [0.0],
	    'keep_prob': [1.0],
	    'step_size': [1e-3],
	    'connection_nums': [6]}

	log_dir = "../output/"
	if "FullyConnectedMLP" in argv:
	    try:
	        os.makedirs(log_dir + "FullyConnectedMLP/")
	    except OSError as e:
	        if e.errno != errno.EEXIST:
	            raise

	    log_dir = log_dir + "FullyConnectedMLP/"
	    counter = 0
	    for l2 in NetWorkGrid['l2_reg']:
	        for kp in NetWorkGrid['keep_prob']:
	            for step in NetWorkGrid['step_size']:

	                try:
	                    os.makedirs(log_dir + "/" + str(counter))
	                except OSError as e:
	                    if e.errno != errno.EEXIST:
	                        raise
	                log_dir_ = log_dir + "/" + str(counter)
	                params_ = {'l2_reg': l2, 'keep_prob': kp, "step_size": step}
	                f = log_dir_ + "/params.json"
	                with open(f, 'wb') as fh:
	                    json.dump(params_, fh) 

	                main(log_dir_, "FullyConnectedMLP" , train_data, train_labels, \
	                    test_data, test_labels, \
	                    validation_data, validation_labels, \
	                    l2, \
	                    kp, \
	                    None, \
	                    step, None)
	                counter = counter + 1


	elif "HierarchyAwareMLP" in argv:

	    try:
	        os.makedirs(log_dir + "HierarchyAwareMLP/")
	    except OSError as e:
	        if e.errno != errno.EEXIST:
	            raise

	    log_dir = log_dir + "HierarchyAwareMLP/"
	    counter = 0
	    for l2 in NetWorkGrid['l2_reg']:
	        for kp in NetWorkGrid['keep_prob']:
	            for step in NetWorkGrid['step_size']:
	                for connection_num in NetWorkGrid['connection_nums']:
	                    try:
	                        os.makedirs(log_dir +  "/" + str(counter))
	                    except OSError as e:
	                        if e.errno != errno.EEXIST:
	                            raise
	                    log_dir_ = log_dir + "/" + str(counter)
	                    params_ = {'l2_reg': l2, 'keep_prob': kp, "step_size": step, "connection_num" : connection_num}
	                    f = log_dir_ + "/params.json"
	                    with open(f, 'wb') as fh:
	                        json.dump(params_, fh) 

	                    main(log_dir_, "HierarchyAwareMLP" , train_data, train_labels, \
	                        test_data, test_labels, \
	                        validation_data, validation_labels, \
	                        l2, \
	                        kp, \
	                        connection_num, \
	                        step, None)

	                    counter = counter + 1
	