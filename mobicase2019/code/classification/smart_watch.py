import sys
sys.path.append('../')

from utils.utils import *
from utils.preliminaries import *
from data_reading.read_data import *

class FeatureExtractor(object):
	def __init__(self, X, y, window_size = 5, step = 1):
		self.window_size = window_size
		self.step = step
		self.X = X
		self.y = y

		#self.features = ["skewness", "mean", "std", "var", "median", "maxima", "minima", "fft"]

		self.features = ["skewness",  "mean",  "std", "var",  "maxima", "minima", "fft"]

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
		

				segments[i_segment] = np.concatenate(( skew_, max_, min_, mean, stdev, var_, energy))
				labels[i_label] = self.y[end-1]
				i_label+=1
				i_segment+=1

		return segments, labels


	def get_extracted_features(self):
		return self.segment()

class Smartwatch():
	def __init__(self, logs):
		self.logs = logs
		self.acc_list = []
		self.hrm_list = []
		self.gyro_list = []

		for line in logs:
			payloads = line["Payload"].split(";")
			self.hrm_list.append((toDateTime(line["TimeStamp"][:19]), payloads[3]))
			self.acc_list.append((toDateTime(line["TimeStamp"][:19]), payloads[5], payloads[6], payloads[7]))
			self.gyro_list.append((toDateTime(line["TimeStamp"][:19]), payloads[9], payloads[10], payloads[11]))

		self.acc_list = sorted_by_time_stamp(self.acc_list)
		self.hrm_list = sorted_by_time_stamp(self.hrm_list)
		self.gyro_list = sorted_by_time_stamp(self.gyro_list)

		self.dates = [x[0] for x in self.hrm_list]
		self.hrm = [x[1] for x in self.hrm_list]

		self.acc_x = [x[1] for x in self.acc_list]
		self.acc_y = [x[2] for x in self.acc_list]
		self.acc_z = [x[3] for x in self.acc_list]

		self.gyro_x = [x[1] for x in self.gyro_list]
		self.gyro_y = [x[2] for x in self.gyro_list]
		self.gyro_z = [x[3] for x in self.gyro_list]

	def _print(self):	
		hrm_time = [x[0] for x in self.hrm_list]
		hrm = [x[1] for x in self.hrm_list]
		dates = matplotlib.dates.date2num(hrm_time)
		plt.plot_date(dates, hrm)
		plt.show()

	def toDataFrame(self):
		self.df = pd.DataFrame(
		    {'TimeStamp': self.dates,
			'SmartWatchHeartRate': self.hrm,
			'SmartWatchAcc_X': self.acc_x,
			'SmartWatchAcc_Y': self.acc_y,
			'SmartWatchAcc_Z': self.acc_z,
			'SmartWatchGyro_X': self.gyro_x,
			'SmartWatchGyro_Y': self.gyro_y,
			'SmartWatchGyro_Z': self.gyro_z
		    })
		return self.df

if __name__ == '__main__':

	task = "basic_activities"
	task_mapping = {"basic_activities": label_mapping, "kitchen_activities": kitchen_label_mapping}

	RawData  = RawDataDigester("../../data/05-14-2018/MQTT_Messages.txt")	
	watch_df = Smartwatch(RawData.get_watch_data()).toDataFrame()
	
	label_pd = read_labels("../../data/05-14-2018/labels.txt")

	watch_df = watch_df[(watch_df.TimeStamp >= label_pd['TimeStamp'].iloc[0]) & (watch_df.TimeStamp <= label_pd['TimeStamp'].iloc[-1])].reset_index(drop=True)
	watch_label_df = watch_df.join(label_pd.set_index('TimeStamp'), on='TimeStamp', how='outer')

	watch_label_df = watch_label_df.replace({task: task_mapping[task]})
	print watch_label_df.shape

	X = watch_label_df[['SmartWatchHeartRate','SmartWatchAcc_X','SmartWatchAcc_Y','SmartWatchAcc_Z', \
						'SmartWatchGyro_X', 'SmartWatchGyro_Y', 'SmartWatchGyro_Z']].astype('float').as_matrix()
	y = watch_label_df[task].astype('float').as_matrix()

	feature_extractor = FeatureExtractor(X, y)
	X, y = feature_extractor.get_extracted_features()
	X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.1)
	print X_train.shape
	print X_test.shape

	'''
	#print X_train
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

	clf = MLPClassifier(solver='sgd', alpha=1e-5,
                     hidden_layer_sizes=(256, 256, 3), max_iter=1000, random_state=1)
	
	clf.fit(X_train, y_train)
	print clf.score(X_train, y_train)
	print clf.score(X_test, y_test)	
	'''