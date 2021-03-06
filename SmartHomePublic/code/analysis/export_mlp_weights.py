import sys
sys.path.append('../')

from preliminaries.preliminaries import *

def clip(x):
    return np.clip(x, -8., 8.)

def paranoid_log(x, eps=1e-8):
    return np.log(x+eps)

def get_sparse_weights(log_sigma2, w, thresh=3.0):
    log_alpha = clip(log_sigma2 - paranoid_log(np.square(w)))
    select_mask = np.less(log_alpha, thresh).astype(np.float32)
    
    return w*select_mask

def get_actual_weights(model_dir, sensor_name, graph_def):
    """
        Code for getting the weights of the trained models

        Args:
            model_dir: the directory that saves the trained models
            sensor_name: the name of the sensor
            graph_def: the current tensorflow computation graph 
        Returns:
    """
    # obtain all the nodes in the frozen model
    graph_nodes = [n for n in graph_def.node if n.op == 'Const']
    saved_models_log =  model_dir + "saved_sparse_weights/" + sensor_name
 
    try:
        os.makedirs(saved_models_log)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    # find the weights in frozen model and export into text files
    log_sigma2 = None
    for n in graph_nodes:
        param =  tensor_util.MakeNdarray(n.attr['value'].tensor)
        if param.shape is not ():
            name = n.name.split("/")

            if sensor_name is not "cloud":
                if "log_sigma" in str(name[-1]):
                    log_sigma2 = param
                else:
                    if "w" in str(name[-1]):
                        param = get_sparse_weights(log_sigma2, param)

                    np.savetxt(saved_models_log + "/" + name[-1] + "_weight_values.txt", param) 
            else:
                if "log_sigma" not in str(name[-1]):
                    np.savetxt(saved_models_log + "/" + name[-1] + "_weight_values.txt", param) 

def load_frozen_graph(model_dir, sensor_name, sensor_input, variable_list):
    """
        Code for loading single trained model

        Args:
            model_dir: the directory that saves the trained models
            sensor_name: the name of the sensor
            sensor_input: the data collected from the current sensor
            variable_list: the list for looking up variable names in the original network 

        Returns: the output of the trained model given the sensor_input
    """

    filename = model_dir + sensor_name +  "_frozen.pb"

    with tf.Session() as sess:
        with tf.gfile.GFile(filename, "r") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        new_input = tf.placeholder(tf.float32, [None, sensor_input.shape[1]], sensor_name)
        keep_prob = tf.placeholder(tf.float32)
        phase = tf.placeholder(tf.bool)

        variable_name = sensor_name + "_output"
        for idx, var in enumerate(variable_list):
            if variable_name in var:
                variable_name = var
                break

        if sensor_name is not "cloud":

            output = tf.import_graph_def(
                graph_def,
                input_map={"input/" + sensor_name+":0": new_input, "input/phase:0": phase},
                return_elements = [variable_name+":0"]
            )
        else:
            output = tf.import_graph_def(
                graph_def,
                input_map={"input/" + sensor_name+":0": new_input, "input/keepprob:0": keep_prob},
                return_elements = [variable_name+":0"]
            )

        get_actual_weights(model_dir, sensor_name, graph_def)

        if type(sensor_input) is not np.ndarray:
            sensor_input = sensor_input.eval()


        results = sess.run(output, feed_dict={new_input: sensor_input, phase : False, keep_prob: 1.0})

        return results

def hierarchical_inference(model_dir, test_data, test_labels, sensors, features_index, variable_list):
    """
        Code for hierarchical inference

        Args:
            model_dir: the directory that saves the trained models
            test_data: test dataset saved in numpy array
            test_labels: test labels saved in numpy array
            sensors: the sensor used for testing
            features_index: index for the features of each sensor
            variable_list: the list for looking up variable names in the original network 

        Returns:
    """

    kitchen_sensors = ["teapot_plug", "pressuremat", "metasense"]
    smartthings_sensors = ['cabinet1', 'cabinet2', 'drawer1', 'drawer2', 'fridge']
    livingroom_sensors = ['tv_plug']
    smart_watch_sensors = ['watch']
    ble_location_sensors = ['location']

    classes = int(test_labels.max())+1
    test_labels = test_labels.astype(int)
    test_labels = np.eye(classes)[test_labels].reshape(test_labels.shape[0], classes)

    local_hidden_layers = [16, 32, 64, 128]
    for local_hidden_layer in local_hidden_layers:
        sub_model_dir = model_dir + "local_hidden_layer_" + str(local_hidden_layer) + "/"

        kitchen_input = []
        livingroom_input = []
        smartthings_input = []
        smartwatch_input = []
        ble_location_input = []  

        for idx, sensor in enumerate(sensors):
            if sensor not in smartthings_sensors and sensor not in smart_watch_sensors and sensor not in ble_location_sensors:
                sensor_output = load_frozen_graph(sub_model_dir, sensor, test_data[idx], variable_list)
                sensor_output = sensor_output[0]
            else:
                sensor_output = test_data[idx]
                sensor_output = tf.convert_to_tensor(sensor_output, np.float32)

            if sensor in kitchen_sensors:
                kitchen_input.append(sensor_output)
            elif sensor in smartthings_sensors:
                smartthings_input.append(sensor_output)
            elif sensor in livingroom_sensors:
                livingroom_input.append(sensor_output)
            elif sensor in smart_watch_sensors:
                smartwatch_input.append(sensor_output)
            elif sensor in ble_location_sensors:
                ble_location_input.append(sensor_output)

        kitchen_input = tf.concat(kitchen_input, axis=1)
        livingroom_input = tf.concat(livingroom_input, axis=1)
        smartthings_input = tf.concat(smartthings_input, axis=1)
        smartwatch_input = tf.concat(smartwatch_input, axis=1)
        ble_location_input = tf.concat(ble_location_input, axis=1)


        level2_input = [kitchen_input, livingroom_input, smartthings_input, smartwatch_input, ble_location_input ]
        level2_model = ['kitchen', 'livingroom', 'smartthings', 'smart_watch', "ble_location"]
        cloud_input = []

        for idx, sensor in enumerate(level2_model):
            sensor_output = load_frozen_graph(sub_model_dir, sensor, level2_input[idx], variable_list)
            sensor_output = sensor_output[0]
            cloud_input.append(sensor_output)

        cloud_input = tf.concat(cloud_input, axis=1)

        output = load_frozen_graph(sub_model_dir, "cloud", cloud_input, variable_list)
        output = output[0]

        correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(test_labels, 1))
        correct_prediction = tf.cast(correct_prediction, tf.float32)
        accuracy = tf.reduce_mean(correct_prediction)

        with tf.Session() as sess:
            print sess.run(accuracy)

def partition_features(train_data, features_index):
    sensor_data_list = []
    for key, item in features_index.iteritems():
        sensor_data_list.append(train_data[:, item])

    return sensor_data_list

if __name__=="__main__":     

    subject2_data = pd.read_hdf("../../temp/data_processed_centered.h5", "subject2")

    metasense_vars = filter(
        lambda x: "metasense_S" in x, subject2_data.columns)

    subject2_data = subject2_data.drop(metasense_vars, axis="columns")
    with open("../../temp/sensors.txt") as fh:
        sensors = eval(fh.read())

    sensors = sensors[:-1]
    # drop the "cooking" category due to measurement error
    subject2_data = subject2_data.loc[subject2_data["label"] != 1.0,:]

    with open("../../temp/sensors.txt") as fh:
        sensors = eval(fh.read())
    sensors = sensors[:-1]
     # get feature index for each sensor
    features =  subject2_data.columns.tolist()[1:]
    features_index = collections.OrderedDict()
    for sensor in sensors:
        features_index[sensor] = []
        for idx, feature in enumerate(features):
            if sensor in feature:
                features_index[sensor].append(idx)


    with open("../../temp/tensorflow_variable_list.txt") as fh:
        variable_list = eval(fh.read())
    subject2_data = subject2_data.values[100:-200,:]

    test_data = subject2_data
    '''
    validation_split = np.random.binomial(1, 0.80, size=(test_data.shape[0],))
    test_X  = test_data.drop(
        ['label'], axis=1).loc[validation_split == 0,:].values
    test_y = test_data['label'][validation_split == 0].values
    '''

    test_X = test_data[:,1:]
    test_y = test_data[:,0]

    model_dir = "../../output/NeuralNets/HierarchyAwareMLP/saved_sparse_models/"
    test_X =  partition_features(test_X, features_index)

    hierarchical_inference(model_dir, test_X, test_y, sensors, features_index, variable_list)
    