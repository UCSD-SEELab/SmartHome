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

class Smartthings(object):
	def __init__(self, logs):
		self.logs = logs

		self.drawer_mapping = {'open': 1, 'closed': 0}
		self.motion_mapping = {'active': 1, 'inactive': 0}
		self.tamper_mapping = {'detected': 1, 'clear': 0}

		self.sensor_list = {
			"smartthings/Aeotec MultiSensor 6/humidity": [], \
			"smartthings/Aeotec MultiSensor 6/illuminance": [], \
			"smartthings/Aeotec MultiSensor 6/motion": [], \
			"smartthings/Aeotec MultiSensor 6/tamper": [], \
			"smartthings/Aeotec MultiSensor 6/temperature": [], \
			"smartthings/Basin Left Drawer/contact": [], \
			"smartthings/Bottom Left Drawer/contact": [], \
			"smartthings/Bottom Left Drawer/temperature": [], \
			"smartthings/Bottom Right Drawer/contact": [], \
			"smartthings/Bottom Right Drawer/temperature": [], \
			"smartthings/Cabinet Drawer/contact": [], \
			"smartthings/Cabinet Drawer/temperature": [], \
			"smartthings/Dishwasher Above Drawer/contact": [], \
			"smartthings/Dishwasher Above Drawer/temperature": [], \
			"smartthings/Fridge/acceleration": [], \
			"smartthings/Fridge/contact": [], \
			"smartthings/Fridge/temperature": [], \
			"smartthings/Fridge/threeAxis": [], \
			"smartthings/Motion Sensor/motion": [], \
			"smartthings/Motion Sensor/temperature": [], \
			"smartthings/Stove Lower Drawer/contact": [], \
			"smartthings/Stove Lower Drawer/temperature": [], \
			"smartthings/Top Left Drawer/contact": [], \
			"smartthings/Top Left Drawer/temperature": [], \
			"smartthings/Top Right Drawer/contact": [], \
			"smartthings/Top Right Drawer/temperature": []}

		for key, value in self.sensor_list.iteritems():
			for log in self.logs:
				if log["Topic"] == key:
					self.sensor_list[key].append((toDateTime(log["TimeStamp"][:19]), log["Payload"]))

		for key, value in self.sensor_list.iteritems():
			self.sensor_list[key] = sorted_by_time_stamp(self.sensor_list[key])


	def toDataFrame(self):
		smartthings_df = pd.DataFrame({})

		for key, value in self.sensor_list.iteritems():
			dates = [x[0] for x in value]
			vals = [x[1] for x in value]
			key_df = pd.DataFrame(
				 {'TimeStamp': dates,
				  key: vals,
			    })
			if smartthings_df.empty == True:
				smartthings_df = key_df
			else:
				smartthings_df = smartthings_df.join(key_df.set_index('TimeStamp'), on='TimeStamp', how='outer')
				smartthings_df = smartthings_df.reset_index(drop=True)
		smartthings_df = smartthings_df.sort_values('TimeStamp').reset_index(drop=True)
		
		smartthings_df = smartthings_df.replace({"smartthings/Aeotec MultiSensor 6/motion": self.motion_mapping})
		smartthings_df = smartthings_df.replace({"smartthings/Motion Sensor/motion": self.motion_mapping})
		smartthings_df = smartthings_df.replace({"smartthings/Aeotec MultiSensor 6/tamper": self.tamper_mapping})
		smartthings_df = smartthings_df.replace({"smartthings/Fridge/contact": self.drawer_mapping})
	
		smartthings_df = smartthings_df.replace({"smartthings/Basin Left Drawer/contact": self.drawer_mapping})
		smartthings_df = smartthings_df.replace({"smartthings/Cabinet Drawer/contact": self.drawer_mapping})
		smartthings_df = smartthings_df.replace({"smartthings/Bottom Left Drawer/contact": self.drawer_mapping})
		smartthings_df = smartthings_df.replace({"smartthings/Bottom Right Drawer/contact": self.drawer_mapping})
		smartthings_df = smartthings_df.replace({"smartthings/Top Left Drawer/contact": self.drawer_mapping})
		smartthings_df = smartthings_df.replace({"smartthings/Top Right Drawer/contact": self.drawer_mapping})
		smartthings_df = smartthings_df.replace({"smartthings/Dishwasher Above Drawer/contact": self.drawer_mapping})

		return smartthings_df
	
	def _print(self):
		pass
		#print self.sensor_list["smartthings/Fridge/contact"]

		#for key, value in self.sensor_list.iteritems():
		#	print value

	def _keys(self):
		return self.sensor_list.keys()

if __name__ == '__main__':
	task = "kitchen_activities" 
	task_mapping = {"basic_activities": label_mapping, "kitchen_activities": kitchen_label_mapping}

	RawData  = RawDataDigester("../../data/05-14-2018/MQTT_Messages.txt")	
	smartthings_data = RawData.get_smartthings_data()
	smartthings = Smartthings(smartthings_data)
	smartthings_df = smartthings.toDataFrame()
	print smartthings_df

	label_pd = read_labels("../../data/05-14-2018/labels.txt")
	smartthings_df = smartthings_df[(smartthings_df.TimeStamp >= label_pd['TimeStamp'].iloc[0]) & (smartthings_df.TimeStamp <= label_pd['TimeStamp'].iloc[-1])].reset_index(drop=True)
		
	smartthings_label_df = smartthings_df.join(label_pd.set_index('TimeStamp'), on='TimeStamp', how='outer')
	print smartthings_label_df
	
	smartthings_label_df = smartthings_label_df.fillna(method="bfill")
	smartthings_label_df = smartthings_label_df.fillna(0)
	smartthings_label_df = smartthings_label_df.replace({task: task_mapping[task]})

	X = smartthings_label_df[smartthings._keys()].astype('float').as_matrix()
	y = smartthings_label_df[task].astype('float').as_matrix()

	#feature_extractor = FeatureExtractor(X, y)
	#X, y = feature_extractor.get_extracted_features()
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
	