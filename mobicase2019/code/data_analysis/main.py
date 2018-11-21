import sys
sys.path.append('../')

import scipy.stats as stats

from utils.utils import *
from utils.preliminaries import *
from build import get_preprocessed_data
from lib.hierarchical_neural_networks import *

# get freezed tensorflow model
def freeze_graph(sess, dir_, sensors, variable_list):

    models = sensors + ['kitchen', 'smartthings', 'livingroom', 'smart_watch', 'cloud']
    for model in models:
        variable_name = model + "_output"
        for idx, var in enumerate(variable_list):
            if variable_name in var:
                variable_name = var
                break

        frozen_graph_def = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, [variable_name])
        with tf.gfile.GFile(dir_ + model + "_frozen.pb", "wb") as f:
            f.write(frozen_graph_def.SerializeToString())

def task_difficulties(labels, predicted_labels):
    cnf_matrix = confusion_matrix(labels, predicted_labels)
    #print cnf_matrix
    return cnf_matrix

def get_output(arch,
    teapot_plug_x, \
    pressuremat_x, \
    metasense_x, \
    cabinet1_x, \
    cabinet2_x, \
    drawer1_x, \
    drawer2_x, \
    fridge_x, \
    tv_plug_x, \
    location_x, \
    watch_x, \
    keep_prob, level_1_connection_num, level_2_connection_num, classes, features_index = None):

    if arch == "FullyConnectedMLP":
        output = LocalSensorNetwork("MLP", x, [128, 64, classes],  keep_prob=keep_prob).build_layers()

    elif arch == "HierarchyAwareMLP":
        cloud = CloudNetwork("cloud", [128, 64, classes], keep_prob=keep_prob)

        kitchen = CloudNetwork("kitchen", [64, level_2_connection_num], keep_prob=keep_prob)
        livingroom = CloudNetwork("livingroom", [64, level_2_connection_num], keep_prob=keep_prob)
        smartthings = CloudNetwork("smartthings", [64, level_2_connection_num], keep_prob=keep_prob)
        smart_watch = CloudNetwork("smart_watch", [64, level_2_connection_num], keep_prob=keep_prob)
   
        kitchen_sensors = ["teapot_plug", "pressuremat", "metasense"]
        smartthings_sensors = ['cabinet1', 'cabinet2', 'drawer1', 'drawer2', 'fridge']
        livingroom_sensors = ['tv_plug']
        smart_watch_sensors = ['location', 'watch']

        kitchen_input = []
        livingroom_input = []
        smartingthings_input = []
        smartwatch_input = []

        sensors_x = [teapot_plug_x, pressuremat_x, metasense_x, cabinet1_x, cabinet2_x, drawer1_x, drawer2_x, fridge_x, tv_plug_x, location_x, watch_x]

        for idx, (key, value) in enumerate(features_index.iteritems()):

            with tf.variable_scope(key):
                sensor_output = LocalSensorNetwork(key, sensors_x[idx], [64, level_1_connection_num], keep_prob = keep_prob)

                if key in kitchen_sensors:
                    kitchen_input.append(sensor_output)
                elif key in livingroom_sensors:
                    livingroom_input.append(sensor_output)
                elif key in smartthings_sensors:
                    smartingthings_input.append(sensor_output)
                elif key in smart_watch_sensors:
                    smartwatch_input.append(sensor_output)

        kitchen_output = kitchen.connect(kitchen_input)  
        livingroom_output = livingroom.connect(livingroom_input)  
        smartthings_output = smartthings.connect(smartingthings_input)  
        smartwatch_output = smart_watch.connect(smartwatch_input)  

        output = cloud.connect([kitchen_output, livingroom_output, smartthings_output, smartwatch_output])
    return output


def NeuralNets(sensors, log_dir, arch , train_data, train_labels, \
        test_data, test_labels, \
        validation_data, validation_labels, \
        l2, \
        keepprob, \
        level_1_connection_num, \
        level_2_connection_num, \
        starter_learning_rate, \
        subject, epoch, batch_size, features_index, verbose = False):

    tf.reset_default_graph()   
    n_features = train_data.shape[1]
    classes = int(test_labels.max())+1

    train_data =  partition_features(train_data, features_index)
    test_data =  partition_features(test_data, features_index)
    validation_data =  partition_features(validation_data, features_index)


    # convert to one-hot vector
    train_labels = train_labels.astype(int)
    test_labels = test_labels.astype(int)
    validation_labels = validation_labels.astype(int)

    test_labels_classes = test_labels
    train_labels = np.eye(classes)[train_labels].reshape(train_labels.shape[0], classes)
    test_labels = np.eye(classes)[test_labels].reshape(test_labels.shape[0], classes)
    validation_labels = np.eye(classes)[validation_labels].reshape(validation_labels.shape[0], classes)

    with tf.name_scope('input'):
        teapot_plug_x = tf.placeholder(tf.float32, [None, train_data[0].shape[1]], "teapot_plug")
        pressuremat_x = tf.placeholder(tf.float32, [None, train_data[1].shape[1]], "pressuremat")
        metasense_x = tf.placeholder(tf.float32, [None, train_data[2].shape[1]], "metasense")
        cabinet1_x = tf.placeholder(tf.float32, [None, train_data[3].shape[1]], "cabinet1")
        cabinet2_x = tf.placeholder(tf.float32, [None, train_data[4].shape[1]], "cabinet2")
        drawer1_x = tf.placeholder(tf.float32, [None, train_data[5].shape[1]], "drawer1")
        drawer2_x = tf.placeholder(tf.float32, [None, train_data[6].shape[1]], 'drawer2')
        fridge_x = tf.placeholder(tf.float32, [None, train_data[7].shape[1]], "fridge")
        tv_plug_x = tf.placeholder(tf.float32, [None, train_data[8].shape[1]], "tv_plug")
        location_x = tf.placeholder(tf.float32, [None, train_data[9].shape[1]], "location")
        watch_x = tf.placeholder(tf.float32, [None, train_data[10].shape[1]], "watch")

        y_ = tf.placeholder(tf.int32, [None, classes])
        keep_prob = tf.placeholder(tf.float32, name="keepprob")

    output = get_output(arch, 
        teapot_plug_x, \
        pressuremat_x, \
        metasense_x, \
        cabinet1_x, \
        cabinet2_x, \
        drawer1_x, \
        drawer2_x, \
        fridge_x, \
        tv_plug_x, \
        location_x, \
        watch_x, \
        keep_prob, level_1_connection_num, level_2_connection_num, classes, features_index)

    variable_list = [n.name for n in tf.get_default_graph().as_graph_def().node]


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
    validation_accuracy = []

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        batch_count = train_data[0].shape[0] / batch_size
        validation_loss_last_epoch = None
        last_test_cross_entropy = None

        patience = 5
        for epoch in range(training_epochs):
            if verbose: print epoch

            idxs = np.random.permutation(train_data[0].shape[0]) #shuffled ordering
            teapot_plug_x_random = train_data[0][idxs, :]
            pressuremat_x_random =  train_data[1][idxs, :]
            metasense_x_random =  train_data[2][idxs, :]
            cabinet1_x_random =  train_data[3][idxs, :]
            cabinet2_x_random =  train_data[4][idxs, :]
            drawer1_x_random =  train_data[5][idxs, :]
            drawer2_x_random =  train_data[6][idxs, :]
            fridge_x_random =  train_data[7][idxs, :]
            tv_plug_x_random =  train_data[8][idxs, :]
            location_x_random =  train_data[9][idxs, :]
            watch_x_random =  train_data[10][idxs, :]

            Y_random = train_labels[idxs]

            for i in range(batch_count):

                teapot_plug_train_data_batch = teapot_plug_x_random[i * batch_size: (i+1) * batch_size,:]
                pressuremat_train_data_batch = pressuremat_x_random[i * batch_size: (i+1) * batch_size,:]
                metasense_train_data_batch = metasense_x_random[i * batch_size: (i+1) * batch_size,:]
                cabinet1_train_data_batch = cabinet1_x_random[i * batch_size: (i+1) * batch_size,:]
                cabinet2_train_data_batch = cabinet2_x_random[i * batch_size: (i+1) * batch_size,:]
                drawer1_train_data_batch = drawer1_x_random[i * batch_size: (i+1) * batch_size,:]
                drawer2_train_data_batch = drawer2_x_random[i * batch_size: (i+1) * batch_size,:]
                fridge_train_data_batch = fridge_x_random[i * batch_size: (i+1) * batch_size,:]
                tv_plug_data_batch = tv_plug_x_random[i * batch_size: (i+1) * batch_size,:]  
                location_train_data_batch = location_x_random[i * batch_size: (i+1) * batch_size,:]
                watch_train_data_batch = watch_x_random[i * batch_size: (i+1) * batch_size,:]

                train_label_batch = Y_random[i * batch_size: (i+1) * batch_size]

                _ = sess.run([train_step], feed_dict={
                    teapot_plug_x: teapot_plug_train_data_batch, \
                    pressuremat_x: pressuremat_train_data_batch, \
                    metasense_x: metasense_train_data_batch, \
                    cabinet1_x: cabinet1_train_data_batch, \
                    cabinet2_x: cabinet2_train_data_batch, \
                    drawer1_x: drawer1_train_data_batch, \
                    drawer2_x:  drawer2_train_data_batch, \
                    fridge_x: fridge_train_data_batch, \
                    tv_plug_x: tv_plug_data_batch, \
                    location_x:  location_train_data_batch, \
                    watch_x:  watch_train_data_batch, \
                    y_: train_label_batch, keep_prob: keepprob})


            summary = sess.run(write_op, feed_dict={
                teapot_plug_x: train_data[0], \
                pressuremat_x: train_data[1], \
                metasense_x: train_data[2], \
                cabinet1_x: train_data[3], \
                cabinet2_x: train_data[4], \
                drawer1_x: train_data[5], \
                drawer2_x: train_data[6], \
                fridge_x: train_data[7], \
                tv_plug_x: train_data[8], \
                location_x: train_data[9], \
                watch_x: train_data[10], \
                y_: train_labels, keep_prob: 1.0})

            train_cross_entropy_writer.add_summary(summary, epoch)
            train_cross_entropy_writer.flush()


            summary = sess.run(write_op, feed_dict={
                teapot_plug_x: test_data[0], \
                pressuremat_x: test_data[1], \
                metasense_x: test_data[2], \
                cabinet1_x: test_data[3], \
                cabinet2_x: test_data[4], \
                drawer1_x: test_data[5], \
                drawer2_x: test_data[6], \
                fridge_x: test_data[7], \
                tv_plug_x: test_data[8], \
                location_x: test_data[9], \
                watch_x: test_data[10], \
                y_: test_labels, keep_prob: 1.0})
            test_cross_entropy_writer.add_summary(summary, epoch)
            test_cross_entropy_writer.flush()


            summary = sess.run(write_op, feed_dict={
                teapot_plug_x: validation_data[0], \
                pressuremat_x: validation_data[1], \
                metasense_x: validation_data[2], \
                cabinet1_x: validation_data[3], \
                cabinet2_x: validation_data[4], \
                drawer1_x: validation_data[5], \
                drawer2_x: validation_data[6], \
                fridge_x: validation_data[7], \
                tv_plug_x: validation_data[8], \
                location_x: validation_data[9], \
                watch_x: validation_data[10], \
                y_: validation_labels, keep_prob: 1.0})

            validation_cross_entropy_writer.add_summary(summary, epoch)
            validation_cross_entropy_writer.flush()

            test_acc = sess.run(accuracy,
                feed_dict={
                teapot_plug_x: test_data[0], \
                pressuremat_x: test_data[1], \
                metasense_x: test_data[2], \
                cabinet1_x: test_data[3], \
                cabinet2_x: test_data[4], \
                drawer1_x: test_data[5], \
                drawer2_x: test_data[6], \
                fridge_x: test_data[7], \
                tv_plug_x: test_data[8], \
                location_x: test_data[9], \
                watch_x: test_data[10], \
                y_: test_labels, keep_prob: 1.0})

            train_acc = sess.run(accuracy,
                feed_dict={
                teapot_plug_x: train_data[0], \
                pressuremat_x: train_data[1], \
                metasense_x: train_data[2], \
                cabinet1_x: train_data[3], \
                cabinet2_x: train_data[4], \
                drawer1_x: train_data[5], \
                drawer2_x: train_data[6], \
                fridge_x: train_data[7], \
                tv_plug_x: train_data[8], \
                location_x: train_data[9], \
                watch_x: train_data[10], \
                y_: train_labels, keep_prob: 1.0})

            validation_acc= sess.run(accuracy,
                feed_dict={
                teapot_plug_x: validation_data[0], \
                pressuremat_x: validation_data[1], \
                metasense_x: validation_data[2], \
                cabinet1_x: validation_data[3], \
                cabinet2_x: validation_data[4], \
                drawer1_x: validation_data[5], \
                drawer2_x: validation_data[6], \
                fridge_x: validation_data[7], \
                tv_plug_x: validation_data[8], \
                location_x: validation_data[9], \
                watch_x: validation_data[10], \
                y_: validation_labels, keep_prob: 1.0})

            if verbose:
                print "Train Accuracy: {}".format(train_acc)
                print "Test Accuracy: {}".format(test_acc)
                print "Validation Accuracy: {}".format(validation_acc)
           
            train_accuracy.append(train_acc)
            test_accuracy.append(test_acc)
            validation_accuracy.append(validation_acc)

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

        # freeze the model
        saved_models_log =  log_dir + "saved_models/"
        try:
            os.makedirs(saved_models_log)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

        freeze_graph(sess, saved_models_log, sensors,  variable_list)

        # get confusion matrix
        predicted_labels = sess.run(tf.argmax(output, 1),
            feed_dict={
            teapot_plug_x: test_data[0], \
            pressuremat_x: test_data[1], \
            metasense_x: test_data[2], \
            cabinet1_x: test_data[3], \
            cabinet2_x: test_data[4], \
            drawer1_x: test_data[5], \
            drawer2_x: test_data[6], \
            fridge_x: test_data[7], \
            tv_plug_x: test_data[8], \
            location_x: test_data[9], \
            watch_x: test_data[10], \
            keep_prob: 1.0})

        #wtest_labels_actual = windowed_prediction(test_labels_classes, 3)
        #wtest_labels_predicted = windowed_prediction(predicted_labels, 3)
        cfn_matrix = task_difficulties(test_labels_classes, predicted_labels)
        cfn_matrix = pretty_print_cfn_matrix(cfn_matrix)
        if verbose:
            print cfn_matrix
            print "FINAL ACCURACY: {}".format(
                np.trace(cfn_matrix.values) / cfn_matrix.values.sum().astype(np.float64))
        return train_accuracy, test_accuracy, validation_accuracy, cfn_matrix


def windowed_prediction(labels, window_size):
    return [stats.mode(labels[ix-window_size:ix]).mode[0] 
        for ix in range(window_size,labels.shape[0])]


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
    return values

def partition_features(train_data, features_index):
    sensor_data_list = []
    for key, item in features_index.iteritems():
        sensor_data_list.append(train_data[:, item])

    return sensor_data_list

if __name__=="__main__":    
    #anthony_data, yunhui_data, sensors = get_preprocessed_data(exclude_sensors=['airbeam'])

    yunhui_data = pd.read_hdf("../../temp/data_processed.h5", "yunhui")
    anthony_data = pd.read_hdf("../../temp/data_processed.h5", "anthony")

    with open("../../temp/sensors.txt") as fh:
        sensors = eval(fh.read())

    clf = "HierarchyAwareMLP"

    # get feature index for each sensor
    features =  anthony_data.columns.tolist()[1:]
    sensors = sensors[:-1]
    features_index = collections.OrderedDict()
    for sensor in sensors:
        features_index[sensor] = []
        for idx, feature in enumerate(features):
            if sensor in feature:
                features_index[sensor].append(idx)
    print features_index


    l2_grid = [1e-8]
    kp_grid = [0.50]

    step = 1e-3
    
    # connect sensors to room
    level_1_connection_num = 2

    # connect room to the cloud
    level_2_connection_num = 4

    epoch = 1
    batch_size = 256
    log_dir = "../output/NeuralNets/" + clf + "/"

    try:
        os.makedirs(log_dir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    train_data = anthony_data
    test_data = yunhui_data

    train_X  = train_data.drop(['label'], axis=1).values[:-300,:]
    train_y = train_data['label'].values[:-300]

    '''
    validation_split = np.random.binomial(1, 0.80, size=(test_data.shape[0],))
    test_X  = test_data.drop(
        ['label'], axis=1).loc[validation_split == 0,:].values
    test_y = test_data['label'][validation_split == 0].values
    validation_X  = test_data.drop(
        ['label'], axis=1).loc[validation_split == 1,:].values
    validation_y = test_data['label'][validation_split == 1].values
    '''
    test_X  = test_data.drop(
        ['label'], axis=1).values
    test_y = test_data['label'].values

    validation_X = test_X
    validation_y = test_y

    
    results = []
    for l2 in l2_grid:
        for kp in kp_grid:
            train_acc, test_acc, validation_acc, cfn_matrix = NeuralNets(
                sensors,
                log_dir, clf , train_X , train_y,
                test_X, test_y,
                validation_X, validation_y,
                l2,
                kp,
                level_1_connection_num,
                level_2_connection_num,
                step, None, epoch, batch_size, features_index, True)

            results.append(
                (train_acc, test_acc, validation_acc, cfn_matrix, l2, kp))

    results = sorted(results, key=lambda x: max(x[2]))
    best_results = results[-1]
    best_l2 = best_results[4]
    best_kp = best_results[5]
    print "BEST ACCURACY: {}".format(best_results[1][np.argmax(best_results[2])])
    print "L2: {}".format(best_results[4])
    print "KP: {}".format(best_results[5])
    print "CONFUSION: "
    print pretty_print_cfn_matrix(best_results[3])
    