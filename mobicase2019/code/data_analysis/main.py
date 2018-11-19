import sys
sys.path.append('../')

from utils.utils import *
from utils.preliminaries import *
from build import get_preprocessed_data
from lib.hierarchical_neural_networks import *

def task_difficulties(labels, predicted_labels):
    cnf_matrix = confusion_matrix(labels, predicted_labels)
    #print cnf_matrix
    return cnf_matrix


def get_output(name, x, keepprob, connection_num, classes, features_index=None):
    if name == "FullyConnectedMLP":
        output = LocalSensorNetwork("MLP", x, [256, 256, 100, classes],  keep_prob=keepprob).build_layers()

    elif name == "HierarchyAwareMLP":
        sensor_outputs = []
        for key, value in features_index.iteritems():
            with tf.variable_scope(key):
                sensor_output = LocalSensorNetwork(key, x[:,min(value):max(value)+1], [256, connection_num], keep_prob = keepprob)
                sensor_outputs.append(sensor_output)

        cloud = CloudNetwork("cloud", [256, 100, classes], keep_prob=keepprob)
        output = cloud.connect(sensor_outputs)
    return output


def NeuralNets(log_dir, arch , train_data, train_labels, \
        test_data, test_labels, \
        validation_data, validation_labels, \
        l2, \
        keepprob, \
        connection_num, \
        starter_learning_rate, \
        subject, epoch, batch_size):

    tf.reset_default_graph()   
    n_features = train_data.shape[1]
    classes = int(test_labels.max())+1
    # convert to one-hot vector
    train_labels = train_labels.astype(int)
    test_labels = test_labels.astype(int)
    validation_labels = validation_labels.astype(int)

    test_labels_classes = test_labels
    train_labels = np.eye(classes)[train_labels].reshape(train_labels.shape[0], classes)
    test_labels = np.eye(classes)[test_labels].reshape(test_labels.shape[0], classes)
    validation_labels = np.eye(classes)[validation_labels].reshape(validation_labels.shape[0], classes)


    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [None, train_data.shape[1]])
        y_ = tf.placeholder(tf.int32, [None, classes])
        keep_prob = tf.placeholder(tf.float32)

    output = get_output(arch, x, keep_prob, connection_num, classes)

    training_epochs = epoch
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

    train_accuracy = []
    test_accuracy = []

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        batch_count = train_data.shape[0] / batch_size
        validation_loss_last_epoch = None
        last_test_cross_entropy = None

        patience = 5
        for epoch in range(training_epochs):
            print epoch
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

            validation_acc= sess.run(accuracy,
                feed_dict={x: validation_data, 
                y_: validation_labels, keep_prob: 1.0})

            print "Train Accuracy: {}".format(train_acc)
            print "Test Accuracy: {}".format(test_acc)
            print "Validation Accuracy: {}".format(validation_acc)

           
            train_accuracy.append(train_acc)
            test_accuracy.append(test_acc)

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

        # get confusion matrix
        predicted_labels = sess.run(tf.argmax(output, 1),
            feed_dict={x: validation_data, keep_prob: 1.0})
        cfn_matrix = task_difficulties(test_labels_classes, predicted_labels)
        pretty_print_cfn_matrix(cfn_matrix)
        return train_accuracy, test_accuracy, cfn_matrix

def XGB(train_X, train_y, test_X, test_y):
    print "Start classification"
    xg = xgboost.XGBClassifier()
    xg.fit(train_X, train_y)
    print "Train accuracy"
    print xg.score(train_X, train_y)
    print "Test accuracy"
    print xg.score(test_X, test_y)


def logit(train_X, train_y, test_X, test_y):
    pass


def pretty_print_cfn_matrix(cfn_matrix):
    values = pd.DataFrame(cfn_matrix)
    cols = [x[1] for x in sorted(LABEL_ENCODING2NAME.items())]
    values.columns = cols
    values["name"] = cols
    values = values.set_index("name")
    print values

if __name__=="__main__":
    anthony_data, yunhui_data = get_preprocessed_data()
    
    l2 = 1e-3
    kp = 0.30
    step = 1e-3
    epoch = 5
    batch_size = 256
    log_dir = "../output/NeuralNets/"

    # drop = "humidity"
    # #relevant_cols = ["in_kitchen","in_dining_room","in_living_room"]
    # relevant_cols = filter(lambda x: drop in x, anthony_data.columns)
    # print relevant_cols
    # anthony_data = anthony_data.drop(relevant_cols, axis="columns")
    # yunhui_data = yunhui_data.drop(relevant_cols, axis="columns")

    train_X  = yunhui_data.drop(['label'], axis=1).values[:-300,:]
    train_y = yunhui_data['label'].values[:-300]

    validation_split = np.random.binomial(1, 0.80, size=(anthony_data.shape[0],))
    test_X  = anthony_data.drop(
        ['label'], axis=1).loc[validation_split == 0,:].values
    test_y = anthony_data['label'][validation_split == 0].values
    validation_X  = anthony_data.drop(
        ['label'], axis=1).loc[validation_split == 1,:].values
    validation_y = anthony_data['label'][validation_split == 1].values

    #XGB(train_X, train_y, test_X, test_y)

    train_acc, test_acc, cfn_matrix = NeuralNets(
        log_dir, "FullyConnectedMLP" , train_X , train_y,
        test_X, test_y,
        test_X, test_y,
        l2,
        kp,
        None,
        step, None, epoch, batch_size)


    all_sensors = ["location", "metasense", "tv_plug", "teapot_plug", \
    "pressuremat",  "cabinet1", "cabinet2", "drawer1",  "drawer2", \
    "fridge", "watch"]

    for idx in range(len(all_sensors)):
        print "without " + all_sensors[idx]

        sensors_without_one = all_sensors[:idx] + all_sensors[(idx + 1):]
        anthony_data, yunhui_data = get_preprocessed_data(sensors_without_one)
        train_X  = anthony_data.drop(['label'], axis=1).as_matrix()
        train_y = anthony_data['label'].as_matrix()

        test_X  = yunhui_data.drop(['label'], axis=1).as_matrix()
        test_y = yunhui_data['label'].as_matrix()

        l2 = 0.0
        kp = 1.0
        step = 1e-3
        epoch = 80
        batch_size = 256
        log_dir = "../output/NeuralNets/"

        NeuralNets(log_dir, "FullyConnectedMLP" , train_X , train_y, \
            test_X, test_y, \
            test_X, test_y, \
            l2, \
            kp, \
            None, \
            step, None, epoch, batch_size)
