import sys
sys.path.append('../')

import scipy.stats as stats

from utils.utils import *
from utils.preliminaries import *
from build import get_preprocessed_data

if __name__=="__main__":    
    #anthony_data, yunhui_data, sensors = get_preprocessed_data(exclude_sensors=['airbeam'])

    with open("../../temp/sensors.txt") as fh:
        sensors = eval(fh.read())

    clf = "DecisionTree"
    
    '''
    for sensor in sensors:
        print "drop " + sensor
    '''
    yunhui_data = pd.read_hdf("../../temp/data_processed.h5", "yunhui")
    anthony_data = pd.read_hdf("../../temp/data_processed.h5", "anthony")

    '''
    to_drop = filter(lambda x: sensor in x, anthony_data.columns)
    anthony_data = anthony_data.drop(to_drop, axis="columns")
    yunhui_data = yunhui_data.drop(to_drop, axis="columns")
    '''
    train_data = anthony_data
    test_data = yunhui_data

    train_X  = train_data.drop(['label'], axis=1).values[:-300,:]

    train_y = train_data['label'].values[:-300]

    validation_split = np.random.binomial(1, 0.80, size=(test_data.shape[0],))
    test_X  = test_data.drop(
        ['label'], axis=1).loc[validation_split == 0,:].values

    test_y = test_data['label'][validation_split == 0].values
    validation_X  = test_data.drop(
        ['label'], axis=1).loc[validation_split == 1,:].values
    validation_y = test_data['label'][validation_split == 1].values

    # count each label
    '''
    unique, counts = np.unique(train_y, return_counts=True)
    print dict(zip(unique, counts))
    '''

    log_dir = "../output/TreeModels/" + clf + "/"

    try:
        os.makedirs(log_dir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    if clf == "XGboost":
        try:
            os.makedirs(log_dir + "XGboost/")
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
        f = log_dir + "/XGboost/results.json"
        model = XGBClassifier()

        model.fit(train_X, train_y)
        print "Train acc: {}".format(model.score(train_X, train_y)) 
        print "Test acc: {}".format(model.score(test_X, test_y)) 
        #booster = model.get_booster()
        #plot_tree(booster)
        #plt.show()

    elif clf == "DecisionTree":
        min_impurity_decrease = [0]
        min_samples_split = [2]

        try:
            os.makedirs(log_dir + "DecisionTree/")
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
        f = log_dir + "/DecisionTree/results.json"

        for ms in min_samples_split:
            for mi in min_impurity_decrease:
                model = DecisionTreeClassifier(min_samples_split=ms, min_impurity_decrease=mi, random_state=0)
                json_ = {"min_impurity_decrease": mi, "min_samples_split": ms}
                model.fit(train_X, train_y)
                print "Train acc: {}".format(model.score(train_X, train_y)) 
                print "Test acc: {}".format(model.score(test_X, test_y)) 
                tree.export_graphviz(model, out_file='tree.dot')   
