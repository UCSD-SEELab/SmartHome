import sys
sys.path.append('../')

from utils.utils import *
from utils.preliminaries import *
from data_reading.read_data import *

class FeatureExtractor(object):
	def __init__(self, X, y, window_size = 200, step = 10):
		self.window_size = window_size
		self.step = step
		self.X = X
		self.y = y

		self.features = ["skewness", "mean", "std", "var", "median", "maxima", "minima","fft"]

	# Correlation between two axises
	# l1, l2: list of floats(for a single axis)
	def _gen_corr(self, l1, l2):
		return [np.corrcoef(l1, l2)[0, 1]]

	def segment(self):
		segments = np.zeros(((len(self.X) - self.window_size) / self.step + 1 , self.X.shape[1]*len(self.features)))
		labels = np.zeros(((len(self.X) - self.window_size) / self.step + 1, 1))

		i_segment = 0
		i_label = 0
		for (start, end) in windowz(self.X, self.window_size, self.step):
		    if(len(self.X[start:end]) == self.window_size):
				l = self.X[start:end]

				max_ = np.max(l, axis=0)
				min_ = np.min(l, axis=0)
				mean = np.mean(l, axis=0)
				stdev = np.std(l, axis=0)
				skew_ = skew(l, axis=0)
				median_ = np.median(l, axis=0)
				var_ = np.var(l,axis=0)

				s = 0.0
				ft_l = np.fft.fft(l, axis=0)
				for x in ft_l:
					s += np.abs(x)**2
				energy = s / len(ft_l)	

				segments[i_segment] = np.concatenate((var_, median_, skew_, max_, min_, mean, stdev, energy))
				labels[i_label] = self.y[end-1]
				i_label+=1
				i_segment+=1

		return segments, labels


	def get_extracted_features(self):
		return self.segment()


class MC10(object):
	def __init__(self, acc_path, ecg_path):
		self.acc_data = np.genfromtxt(acc_path, delimiter=',', skip_header=1, names=['t','x','y','z'])
		self.acc_date = [toDateTime(datetime.datetime.fromtimestamp(int(d) / 1000.0).strftime('%Y-%m-%d %H:%M:%S')) for d in self.acc_data['t']]

		self.elec_data = np.genfromtxt(ecg_path, delimiter=',', skip_header=1, names=['t','v'])
		self.elec_date = [toDateTime(datetime.datetime.fromtimestamp(int(d) / 1000.0).strftime('%Y-%m-%d %H:%M:%S')) for d in self.elec_data['t']]

	def toDataFrame(self):
		acc_df = pd.DataFrame(
			 {'TimeStamp': self.acc_date,
			  'MC10Acc_X': self.acc_data['x'],
			  'MC10Acc_Y': self.acc_data['y'],
			  'MC10Acc_Z': self.acc_data['z']
		    })

		elec_df = pd.DataFrame(
		    {'TimeStamp': self.elec_date,
			'MC10_ECG': self.elec_data['v'],
		    })
		
		mc10_df= acc_df.join(elec_df.set_index('TimeStamp'), on='TimeStamp', how='outer')
		mc10_df = mc10_df.reset_index(drop=True)
		return mc10_df


if __name__ == '__main__':
	mc_ = MC10('../../data/05-14-2018/sample_study_yunhui/Sample\ \Study/yunhui/medial_chest/d5la7x6c/2018-05-14T20-59-48-500Z/accel.csv', \
		'../../data/05-14-2018/sample_study_yunhui/Sample\ \Study/yunhui/medial_deltoid_right/d5la7xo3/2018-05-14T20-59-52-228Z/elec.csv')

	df = mc_.toDataFrame()
	
	label_pd = read_labels("../../data/5-14-2018/labels.txt")
	df = df[(df.TimeStamp >= label_pd['TimeStamp'].iloc[0]) & (df.TimeStamp <= label_pd['TimeStamp'].iloc[-1])].reset_index(drop=True)
	label_df = df.join(label_pd.set_index('TimeStamp'), on='TimeStamp', how='outer')

	label_df = label_df.replace({'basic_activities': label_mapping})
	
	X = label_df[['MC10Acc_X','MC10Acc_Y','MC10Acc_Z', 'MC10_ECG']].astype('float').as_matrix()
	y = label_df['basic_activities'].astype('float').as_matrix()


	feature_extractor = FeatureExtractor(X, y)
	X, y = feature_extractor.get_extracted_features()
	X_train, X_test, y_train, y_test = train_test_split(X, y)


	logreg = LogisticRegression()
	logreg.fit(X_train, y_train)
	print logreg.score(X_train, y_train)
	print logreg.score(X_test, y_test)

	X_train = X_train.astype('float')
	X_test = X_test.astype('float')

	xg = xgboost.XGBClassifier()
	xg.fit(X_train, y_train)
	print xg.score(X_train, y_train)
	print xg.score(X_test, y_test)	
	
	