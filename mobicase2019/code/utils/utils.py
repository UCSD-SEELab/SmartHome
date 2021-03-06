from preliminaries import *

kitchen_label_mapping = {'cleaning table': 0, 'clearing table':1, 'eating and drinking': 2, 'food pred': 3, \
 'making coffee': 4, 'microwave': 5, 'nothing': 6, 'putting away dishes': 7, \
 'serving food': 8, 'setting table': 9, 'stove': 10, 'washing dishes by dishwasher': 11, \
 'washing dishes by hand': 12}

label_mapping = {'standing': 0, 'sitting': 1, 'walking': 2}

drawer_mapping = {'open': 1, 'closed': 0}
motion_mapping = {'active': 1, 'inactive': 0}
tamper_mapping = {'detected': 1, 'clear': 0}

def toDateTime(s):
	dt = parser.parse(s)
	return dt

def sorted_by_time_stamp(log_tuples):
	return sorted(log_tuples, key=lambda x: x[0])     

def windowz(data, window_size, step):
    start = 0
    while start < len(data):
        yield start, start + window_size
        start += step

def checkInput():
    index = None
    while index == None:
        try:
            index = int(raw_input())
            return index
        except ValueError:
            print "Wrong choice, please input again." 
            index = None


def partition_features(train_data, features_index):
    sensor_data_list = []
    for key, item in features_index.iteritems():
        sensor_data_list.append(train_data[:, item])

    return sensor_data_list