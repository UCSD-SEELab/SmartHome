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

		self.features = ["mean", "std", "fft"]

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
				mean = np.mean(l, axis=0)
				stdev = np.std(l, axis=0)
				s = 0.0
				ft_l = np.fft.fft(l, axis=0)
				for x in ft_l:
					s += np.abs(x)**2
				energy = s / len(ft_l)	

				segments[i_segment] = np.concatenate((mean, stdev, energy))
				labels[i_label] = self.y[end-1]
				i_label+=1
				i_segment+=1

		return segments, labels


	def get_extracted_features(self):
		return self.segment()

class BLElocalization(object):
	def __init__(self, BLE1, BLE2, BLE3):
		self.BLE1 = BLE1
		self.BLE2 = BLE2
		self.BLE3 = BLE3

		self.BLE1_list = []
		self.BLE2_list = []
		self.BLE3_list = []

		for data in self.BLE1:
			payloads = ast.literal_eval(data["Payload"])
			self.BLE1_list.append((toDateTime(data["TimeStamp"][:19]), payloads["val"]))

		for data in self.BLE2:
			payloads = ast.literal_eval(data["Payload"])
			self.BLE2_list.append((toDateTime(data["TimeStamp"][:19]), payloads["val"]))

		for data in self.BLE3:
			payloads = ast.literal_eval(data["Payload"])
			self.BLE3_list.append((toDateTime(data["TimeStamp"][:19]), payloads["val"]))

		self.BLE1_list = sorted_by_time_stamp(self.BLE1_list)
		self.BLE2_list = sorted_by_time_stamp(self.BLE2_list)
		self.BLE3_list = sorted_by_time_stamp(self.BLE3_list)

		self.dates1 = [x[0] for x in self.BLE1_list]
		self.dates2 = [x[0] for x in self.BLE2_list]
		self.dates3 = [x[0] for x in self.BLE3_list]


		self.BLE1_val = [x[1] for x in self.BLE1_list]
		self.BLE2_val = [x[1] for x in self.BLE2_list]
		self.BLE3_val = [x[1] for x in self.BLE3_list]

	def _print(self):
		dates = matplotlib.dates.date2num(self.dates1)
		plt.plot_date(dates, self.BLE1_val)
		#plt.plot(range(len(self.BLE1_val)), self.BLE1_val)
		plt.show()

	def toDataFrame(self):
		ble1_df = pd.DataFrame(
		    {'TimeStamp': self.dates1,
			'BLE1': self.BLE1_val
		    })
		ble2_df = pd.DataFrame(
		    {'TimeStamp': self.dates2,
			'BLE2': self.BLE2_val
		    })
		ble3_df = pd.DataFrame(
		    {'TimeStamp': self.dates3,
			'BLE3': self.BLE3_val
		    })

		BLE_df = ble1_df.join(ble2_df.set_index('TimeStamp'), on='TimeStamp', how='outer')
		BLE_df = BLE_df.reset_index(drop=True)
		BLE_df = ble3_df.join(BLE_df.set_index('TimeStamp'), on='TimeStamp', how='outer')
		BLE_df = BLE_df.reset_index(drop=True)

		return BLE_df

if __name__ == '__main__':

	task = "kitchen_activities" 
	task_mapping = {"basic_activities": label_mapping, "kitchen_activities": kitchen_label_mapping}

	RawData  = RawDataDigester("../../data/05-14-2018/MQTT_Messages.txt")	
	ble1, ble2, ble3 = RawData.get_ble_data()
	ble = BLElocalization(ble1, ble2, ble3)
	ble_df = ble.toDataFrame()

	label_pd = read_labels("../../data/05-14-2018/labels.txt")
	ble_df = ble_df[(ble_df.TimeStamp >= label_pd['TimeStamp'].iloc[0]) & (ble_df.TimeStamp <= label_pd['TimeStamp'].iloc[-1])].reset_index(drop=True)
		
	ble_label_df = ble_df.join(label_pd.set_index('TimeStamp'), on='TimeStamp', how='outer')

	print ble_label_df
	ble_label_df = ble_label_df.fillna(method="bfill")
	ble_label_df = ble_label_df.fillna(0)
	print ble_label_df


	ble_label_df = ble_label_df.replace({task: task_mapping[task]})
	X = ble_label_df[['BLE1','BLE2','BLE3']].astype('float').as_matrix()
	y = ble_label_df[task].astype('float').as_matrix()

	feature_extractor = FeatureExtractor(X, y)
	X, y = feature_extractor.get_extracted_features()
	X_train, X_test, y_train, y_test = train_test_split(X,y)

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
	