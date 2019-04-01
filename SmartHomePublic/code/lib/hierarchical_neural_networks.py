"""
Hirarchical Neural Networks
"""

import tensorflow as tf
import numpy as np
import variational_dropout as vd

class  Network(object):
    """
        The Network class 

    """
    def __init__(self, name, hidden_layers, activation_fn=tf.nn.relu, initializer=tf.contrib.layers.xavier_initializer, keep_prob = 0.5, sparse=False, phase=False):
        """
            Initilize network

            Args:
                name: name of the network
                hidden_layers: a 1-D tensor indicates the number of hidden neurons in each layer, the last element is the number of output
                activation_fn: activation function 
                initializer: initialization methods
                sparse: sparsify network of not
            Returns:
        """

        self.name = name
        self.hidden_layers = hidden_layers    
        self.activation_fn = activation_fn
        self.initializer = initializer
        self.keep_prob = keep_prob
        self.sparse = sparse
        self.phase = phase
        self.thresh = 3
        self.output = None

    def one_fully_connected_layer(self, x, n_hidden, layer_idx, activation_fn=None):
        """
            Builds one_fully_connected_layer

            Args:
                x: an input tensor with the dimensions (N_examples, N_features)
                n_hidden: number of hidden units
                layer_idx: the layer index of the fully connected layer that is currently being built

            Returns:
                an output tensor with the dimensions (N_examples, n_hidden)
        """

        with tf.variable_scope(self.name):
            n_input = int(x.shape[1])
            w = tf.get_variable("w_"+str(layer_idx), [n_input, n_hidden],
                    initializer=self.initializer())

            b = tf.get_variable("bias_"+str(layer_idx), [n_hidden,],
                    initializer=tf.constant_initializer(0.))

            if activation_fn is not None:
                layer = activation_fn(tf.matmul(x, w) + b)
            else:
                layer = self.activation_fn(tf.matmul(x, w) + b)

            l2_loss  = tf.nn.l2_loss(w, name= "l2_loss" + str(layer_idx))
            return  tf.nn.dropout(layer, self.keep_prob)

    def build_layers(self):
        """
            Builds a stack of fully connected layers

            Args:

            Returns:
                an output tensor with the dimensions (N_examples, hidden_layers[-1])
        """

        output = self.x
        for layer_idx, n_hidden in enumerate(self.hidden_layers[:-1]):
            if self.sparse:
                output = vd.fully_connected(output, self.phase, n_hidden, layer_idx, scope=self.name)
            else:
                output = self.one_fully_connected_layer(output, n_hidden, layer_idx, tf.nn.relu)

        with tf.variable_scope(self.name):

            if self.sparse:
                output = vd.fully_connected(output, self.phase, self.hidden_layers[-1], "output", activation_fn = tf.identity, scope=self.name)
            else:
                output = self.one_fully_connected_layer(output, self.hidden_layers[-1], "output", tf.identity)
            output = tf.identity(output, name=self.name + "_output")

            '''
            n_input = int(output.shape[1])
            w = tf.get_variable("w_output", [n_input, self.hidden_layers[-1]],
                    initializer=self.initializer())

            b = tf.get_variable("bias_output", [self.hidden_layers[-1],],
                    initializer=tf.constant_initializer(0.))
        
            log_sigma2 = vd.log_sigma2_variable([n_input, self.hidden_layers[-1]], "output")
            log_alpha =  vd.get_log_alpha(log_sigma2, w, name="output")

            # at test time,we just mask
            select_mask = tf.cast(tf.less(log_alpha, self.thresh), tf.float32)

            # choose between adding noise, or applying mask, depending on phase
            output = tf.cond(self.phase, lambda: fc_noisy(output, log_alpha, w, b), lambda: fc_masked(output, select_mask, w, b))
            output = tf.identity(output, name=self.name + "_output")

            self.output = output
            '''            
            return output

    def get_output(self):
        return self.output

class LocalSensorNetwork(Network):
    """
        The LocalSensorNetwork class 
    """
    def __init__(self, name, x, hidden_layers, activation_fn = tf.nn.relu, initializer=tf.contrib.layers.xavier_initializer, keep_prob = 1.0, sparse=True, phase=False):
        """
            Initilize network

            Args:
                name: name of the network
                x: an input tensor with the dimensions (N_examples, N_features)
                hidden_layers: a 1-D tensor indicates the number of hidden neurons in each layer, the last element is the number of output
                activation_fn: activation function 
                initializer: initialization methods

            Returns:
        """
        super(LocalSensorNetwork, self).__init__(name, hidden_layers, activation_fn, initializer, keep_prob, sparse, phase)
        self.x = x

class CloudNetwork(Network):
    """
        The CloudNetwork class

        Examples
        --------
        sensor1 = LocalSensorNetwork("sensor1", input_sensor_1, [128,8])
        sensor2 = LocalSensorNetwork("sensor2", input_sensor_2, [256,16])

        cloud = CloudNetwork("cloud", [256,10])
        model = cloud.connect([sensor1, sensor2])
        --------

    """
    def __init__(self, name, hidden_layers, activation_fn=tf.nn.relu, initializer=tf.contrib.layers.xavier_initializer, keep_prob = 0.5, sparse=False, phase=False):
         super(CloudNetwork, self).__init__(name, hidden_layers, activation_fn, initializer, keep_prob, sparse, phase)
         self.name = name
    
    def connect(self, sensors = [], method = "concat"):
        """
            Connect the output of LocalSensorNetworks to CloudNetwork

            Args:
                sensors: a list of LocalSensorNetworks instance

                method : {'inner_product', 'concat'}
                    Specifies how to connect LocalSensorNetworks with the CloudNetwork

            Returns:
                an output tensor with the dimensions (N_examples, hidden_layers[-1])
        """
        outputs = []
        for sensor_idx, sensor in enumerate(sensors):
            with tf.variable_scope("connect_sensor_" + str(sensor_idx)):
                if isinstance(sensor, LocalSensorNetwork):
                    sensor_output = sensor.build_layers()
                else:
                    sensor_output = sensor

                n_input = int(sensor_output.shape[1])
                # output one feature to the could
                if method == "inner_product":
                    w = tf.get_variable("weight_"+str(sensor_idx), [n_input, 1],
                        initializer=self.initializer())
                    output = tf.matmul(sensor_output, w)
                    outputs.append(output)
                # concat the outputs of local sensors
                elif method == "concat":
                    outputs.append(sensor_output)

        self.x = tf.concat(outputs, axis=1, name= "input/" + self.name)
     
        return self.build_layers()

def gather_l2_loss(graph):
    node_defs = [n for n in graph.as_graph_def().node if 'l2_loss' in n.name]
    tensors = [graph.get_tensor_by_name(n.name+":0") for n in node_defs]
    return tensors


