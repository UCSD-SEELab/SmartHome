import sys
sys.path.append('../')

import scipy.stats as stats

from preliminaries.preliminaries import *
from lib.hierarchical_neural_networks import *
import lib.variational_dropout as vd

import random

class MyDataFlow():
    def __init__(self, mode):
        self.mode = mode
        if self.mode == "raw_data":
            self.files_path = "/Desktop/hierarchical_learning_datasets/PAMAP2_Dataset/Protocol/"
            self.files = {
                    'data': ['subject101.dat'] #, 'subject102.dat','subject104.dat', 'subject105.dat', 'subject106.dat', 'subject107.dat', 'subject108.dat']
                }
            def readPamap2(files, train_or_test):
                # Only include the following activities
                activities = ['1','2','3','4','5']
                data = readPamap2Files(files['data'], activities)
                return data
                
            def readPamap2Files(filelist, activities):
                data = []
                labels = []
                for i, filename in enumerate(filelist):
                    print('Reading file %d of %d' % (i+1, len(filelist)))
                    with open(self.files_path + filename, 'r') as f:
           
                        df = pd.read_csv(f, delimiter=' ',  header = None)
                        df = df.dropna()
                        df = df[df[1].isin(activities)]
                        df = df._get_numeric_data()
                        df = df.as_matrix()
                        data_ = df[:,2:]
                        label_ = df[:,1]

                        data.append(np.asarray(data_))
                        labels.append(np.asarray(label_, dtype=int)-1)

                return {'inputs': data, 'targets': labels}
            self.data = readPamap2(self.files, 'data')
            self.data_list, self.label_list = self.data['inputs'], self.data['targets']

        elif self.mode == "extracted_data":

            def read_data(pathName, sensors):
                print pathName
                data_dict = {}
                labels = []
                for sensor in sensors:
                    data_dict[sensor] = []
                    file_ = pathName + sensor + "_data.txt"
                    with open(file_) as f:
                        for line in f:
                            line = line.strip().split(' ')
                            line = [np.float32(i) for i in line]
                            data_dict[sensor].append(line)
                file_ = pathName + "basic_activity_labels_data.txt"
                with open(file_) as f:
                    for line in f:
                        line = line[:-1]
                        line = int(np.float32(line)) 
                        labels.append(line)
                train_data = [] 
                hand = data_dict['hand']
                chest = data_dict['chest']
                ankle = data_dict['ankle']

                for i in range(len(hand)):
                    tmp = []
                    tmp = hand[i]+chest[i]+ankle[i]
                    train_data.append(tmp)

                train_data = np.asarray(train_data)
                train_labels = np.asarray(labels)

                idx1 = np.nonzero(train_labels != 99)
                train_data = train_data[idx1[0]]
                train_labels = train_labels[idx1[0]]
                train_labels = train_labels - 1

                return  train_data, train_labels

            path_ = "/home/henry/Desktop/PAMAP2_Dataset/"

            sensors = ['hand', 'chest', 'ankle']
            train_path1 = path_ + 'ext_features/subject101_ext_features/'
            train_path2 = path_ + 'ext_features/subject102_ext_features/'
            #train_path3 = '/Users/henry/Desktop/ucsd/seelab/Hirarchical_ML/iot/code/PAMAP2_parser/subject103_ext_features/'
            train_path4 = path_ + 'ext_features/subject104_ext_features/'
            train_path5 = path_ + 'ext_features/subject105_ext_features/'
            train_path6 = path_ + 'ext_features/subject106_ext_features/'
            train_path7 = path_ + 'ext_features/subject107_ext_features/'
            train_path8 = path_ + 'ext_features/subject108_ext_features/'

            train_data1, train_labels1 = read_data(train_path1, sensors)
            train_data2, train_labels2 = read_data(train_path2, sensors)
            #train_data3, train_labels3 = read_data(train_path3, sensors)
            train_data4, train_labels4 = read_data(train_path4, sensors)
            train_data5, train_labels5 = read_data(train_path5, sensors)
            train_data6, train_labels6 = read_data(train_path6, sensors)
            train_data7, train_labels7 = read_data(train_path7, sensors)
            train_data8, train_labels8 = read_data(train_path8, sensors)
            self.data_list = [train_data1, train_data2, train_data4, train_data5, train_data6, train_data7, train_data8]
            self.label_list = [train_labels1, train_labels2, train_labels4, train_labels5, train_labels6, train_labels7, train_labels8]
        
    def get_data(self):
        return self.data_list, self.label_list 
        
#cross validation
def pamap2_cv(subjects, i, data_list, label_list):

    test_data = data_list[i]
    test_labels = label_list[i]

    valiation_idx = random.randint(0, len(subjects)-1)

    while valiation_idx == i:
         valiation_idx = random.randint(0, len(subjects)-1)

    validation_data = data_list[valiation_idx]
    validation_labels = label_list[valiation_idx]

    train_data_list = data_list[:]
    train_label_list = label_list[:]

    train_data_list.remove(test_data)
    train_data_list.remove(validation_data)
    train_label_list.remove(test_labels)
    train_label_list.remove(validation_labels)

    train_data = np.concatenate(train_data_list)
    train_labels = np.concatenate(train_label_list)

    train_labels = train_labels.reshape(train_labels.shape[0], 1)
    test_labels = test_labels.reshape(test_labels.shape[0], 1)
    validation_labels = validation_labels.reshape(validation_labels.shape[0], 1)

    idx1 = np.nonzero(train_labels != 99)
    train_data = train_data[idx1[0]]
    train_labels = train_labels[idx1[0]]

    idx2 = np.nonzero(test_labels != 99)
    test_data = test_data[idx2[0]]
    test_labels = test_labels[idx2[0]]
    
    idx3 = np.nonzero(validation_labels != 99)
    validation_data = validation_data[idx3[0]]
    validation_labels = validation_labels[idx3[0]]      

    one_hot_train_labels = np.eye(5)[train_labels].reshape(train_labels.shape[0], 5)
    one_hot_test_labels = np.eye(5)[test_labels].reshape(test_labels.shape[0], 5)
    one_hot_validation_labels = np.eye(5)[validation_labels].reshape(validation_labels.shape[0], 5)
    
    return i, train_data, one_hot_train_labels, test_data, one_hot_test_labels, validation_data, one_hot_validation_labels

def get_output(arch, x, keepprob, connection_nums, phase, thresh):
    if arch == "FullyConnectedMLP":
        output = LocalSensorNetwork("MLP", x, [256, 5],  keep_prob=keepprob).build_layers()

    elif arch == "HierarchyAwareMLP":
        hand = LocalSensorNetwork("hand", x[:,0:12], [64, connection_num], keep_prob=1.0,  sparse=True, phase=phase, thresh=thresh)
        ankle = LocalSensorNetwork("ankle", x[:,12:24], [64, connection_num],  keep_prob=1.0,  sparse=True, phase=phase, thresh=thresh)
        chest = LocalSensorNetwork("chest", x[:,24:], [64, connection_num],  keep_prob=1.0,  sparse=True, phase=phase, thresh=thresh)
        cloud = CloudNetwork("cloud", [256, 5],  keep_prob = keepprob,  sparse=False, phase=phase)
        output = cloud.connect([hand, chest, ankle])
        
    return output

def main(arch , train_data, train_labels, \
    test_data, test_labels, \
    validation_data, validation_labels, \
    l2, \
    keepprob, \
    connection_num, \
    starter_learning_rate, \
    subject, batch_size, training_epochs, thresh):
  
    tf.reset_default_graph()
    tf.set_random_seed(0)

    n_features = train_data.shape[1]
    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [None, 36])
        y_ = tf.placeholder(tf.int32, [None, 5])
        keep_prob = tf.placeholder(tf.float32)
        phase = tf.placeholder(tf.bool, name="phase")

    output = get_output(arch, x, keep_prob, connection_num,  phase, thresh)

    with tf.name_scope('cross_entropy'):

        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=output))   
        
        l2_loss = sum(
            tf.nn.l2_loss(tf_var)
                for tf_var in tf.trainable_variables()
                    if not ("noreg" in tf_var.name or "bias" in tf_var.name))

        # prior DKL part of the ELBO
        log_alphas = vd.gather_logalphas(tf.get_default_graph())
        divergences = [vd.dkl_qp(la) for la in log_alphas]
      
        N = float(train_data[0].shape[0])
        dkl = tf.reduce_sum(tf.stack(divergences))

        # combine to form the ELBO
        total_loss = cross_entropy + l2 * l2_loss + (1./N)*dkl
  
    
    with tf.name_scope('sparseness'):
        sparse = vd.sparseness(log_alphas, thresh)
    

    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                           1000, 0.9, staircase=True)

    with tf.name_scope('sgd'):
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(total_loss, global_step=global_step)
        #train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(total_loss, global_step=global_step)

    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(y_, 1))
        correct_prediction = tf.cast(correct_prediction, tf.float32)
        accuracy = tf.reduce_mean(correct_prediction)

    train_accuracy = []
    test_accuracy = []
    validation_accuracy = []

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
                _ = sess.run([train_step], feed_dict={x: train_data_batch, y_: train_label_batch, keep_prob: keepprob, phase: True})

            train_acc = sess.run(accuracy,
                feed_dict={x: train_data, y_: train_labels, keep_prob: 1.0, phase: False})

            test_acc, test_sparsity = sess.run((accuracy, sparse), 
                feed_dict={x: test_data, y_: test_labels, keep_prob: 1.0, phase: False})

            val_acc, val_sparsity = sess.run((accuracy, sparse), 
                feed_dict={x: validation_data, y_: validation_labels, keep_prob: 1.0, phase: False})

            print "Train Accuracy: {}".format(train_acc)
            print "Test Accuracy: {}, Test Sparsity: {}".format(test_acc, test_sparsity)
            print "Validation Accuracy: {}, Val Sparsity: {}".format(val_acc, val_sparsity)
            
            train_accuracy.append(train_acc)
            test_accuracy.append(test_acc)
            validation_accuracy.append(val_acc)

        return train_accuracy, test_accuracy, validation_accuracy

if __name__=="__main__":

    subjects = ["subject101", "subject102", "subject104", "subject105", "subject106", "subject107", "subject108"]
   
    data_list, label_list = MyDataFlow("extracted_data").get_data()
  
    NetWorkGrid = {'l2_reg': [1e-2],
        'keep_prob': [0.5],
        'step_size': [1e-2],
        'connection_nums': [6]
        }

    test_subject = 0
    training_epochs = 30
    batch_size = 64

    threshs = [8, 8.1 ,8.2, 8.3, 8.4, 8.5]

    results = []
    for l2 in NetWorkGrid['l2_reg']:
        for kp in NetWorkGrid['keep_prob']:
            for step in NetWorkGrid['step_size']:
                for thresh in threshs:
                    for connection_num in NetWorkGrid['connection_nums']:
                        print "Threshold: {}".format(thresh)
                            
                        subject, train_data, train_labels, test_data, test_labels, validation_data, validation_labels = pamap2_cv(subjects, test_subject, data_list, label_list)

                        train_acc, test_acc, validation_acc = main("HierarchyAwareMLP", train_data, train_labels, \
                            test_data, test_labels, \
                            validation_data, validation_labels, \
                            l2, \
                            kp, \
                            connection_num, \
                            step, subject, batch_size, training_epochs, thresh)

                        results.append((train_acc, test_acc, validation_acc, l2, kp))

    '''
    results = sorted(results, key=lambda x: max(x[1]))
    best_results = results[-1]
    best_l2 = best_results[3]
    best_kp = best_results[4]

    print "========================="
    print "RESULTS"
    print "========================="

    print "BEST ACCURACY: {}".format(best_results[1][np.argmax(best_results[2])])
    print "L2: {}".format(best_results[3])
    print "KP: {}".format(best_results[4]) 
    '''