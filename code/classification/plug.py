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

class Plugs(object):
	def __init__(self, plug1, plug2, plug3):
		self.plug1 = plug1
		self.plug2 = plug2
		self.plug3 = plug3

		self.plug_1_voltage_list = []
		self.plug_1_current_list = []

		self.plug_2_voltage_list = []
		self.plug_2_current_list = []

		self.plug_3_voltage_list = []
		self.plug_3_current_list = []

		for data in self.plug1:
			payloads = ast.literal_eval(data['Payload'])
			self.plug_1_voltage_list.append((toDateTime(data["TimeStamp"][:19]), payloads["voltage"]))
			self.plug_1_current_list.append((toDateTime(data["TimeStamp"][:19]), payloads["current"]))

		for data in self.plug2:
			payloads = ast.literal_eval(data["Payload"])
			self.plug_2_voltage_list.append((toDateTime(data["TimeStamp"][:19]), payloads["voltage"]))
			self.plug_2_current_list.append((toDateTime(data["TimeStamp"][:19]), payloads["current"]))

		for data in self.plug3:
			payloads = ast.literal_eval(data["Payload"])
			self.plug_3_voltage_list.append((toDateTime(data["TimeStamp"][:19]), payloads["voltage"]))
			self.plug_3_current_list.append((toDateTime(data["TimeStamp"][:19]), payloads["current"]))

		self.plug_1_voltage_list = sorted_by_time_stamp(self.plug_1_voltage_list)
		self.plug_1_current_list = sorted_by_time_stamp(self.plug_1_current_list)

		self.plug_2_voltage_list = sorted_by_time_stamp(self.plug_2_voltage_list)
		self.plug_2_current_list = sorted_by_time_stamp(self.plug_2_current_list)

		self.plug_3_voltage_list = sorted_by_time_stamp(self.plug_3_voltage_list)
		self.plug_3_current_list = sorted_by_time_stamp(self.plug_3_current_list)

		self.dates1 = [x[0] for x in self.plug_1_current_list]
		self.dates2 = [x[0] for x in self.plug_2_current_list]
		self.dates3 = [x[0] for x in self.plug_3_current_list]

		self.plug_1_vol = [x[1] for x in self.plug_1_voltage_list]
		self.plug_1_cur = [x[1] for x in self.plug_1_current_list]

		self.plug_2_vol = [x[1] for x in self.plug_2_voltage_list]
		self.plug_2_cur = [x[1] for x in self.plug_2_current_list]

		self.plug_3_vol = [x[1] for x in self.plug_3_voltage_list]
		self.plug_3_cur = [x[1] for x in self.plug_3_current_list]

	def toDataFrame(self):
		plug1_df = pd.DataFrame(
		    {'TimeStamp': self.dates1,
			'plug_1_current': self.plug_1_cur,
			'plug_1_voltage': self.plug_1_vol
		    })

		plug2_df = pd.DataFrame(
		{'TimeStamp': self.dates2,
		'plug_2_current': self.plug_2_cur,
		'plug_2_voltage': self.plug_2_vol
		})

		plug3_df = pd.DataFrame(
		    {'TimeStamp': self.dates3,
			'plug_3_current': self.plug_3_cur,
			'plug_3_voltage': self.plug_3_vol
		    })

		plug_df = plug1_df.join(plug2_df.set_index('TimeStamp'), on='TimeStamp', how='outer')
		plug_df = plug_df.reset_index(drop=True)
		plug_df = plug3_df.join(plug_df.set_index('TimeStamp'), on='TimeStamp', how='outer')
		plug_df = plug_df.reset_index(drop=True)

		self.plug_df = plug_df
		return self.plug_df

	def _print_plug_1(self):
		plug_1_current_time = [x[0] for x in self.plug_1_current_list]
		plug_1_current = [x[1] for x in self.plug_1_current_list]
		dates = matplotlib.dates.date2num(plug_1_current_time)
		print dates
		plt.plot_date(dates, plug_1_current)
		plt.show()

	def plot_current(self):
		self.toDataFrame()

		plt.title("Action rating vs Comedy rating - by year", loc='center')
		plt.plot_date(self.plug_df['TimeStamp'], self.plug_df['plug_1_current'])
		plt.plot_date(self.plug_df['TimeStamp'], self.plug_df['plug_2_current'])
		plt.plot_date(self.plug_df['TimeStamp'], self.plug_df['plug_3_current'])

		plt.show()


if __name__ == '__main__':

	task = "basic_activities" 
	task_mapping = {"basic_activities": label_mapping, "kitchen_activities": kitchen_label_mapping}

	RawData  = RawDataDigester("../../data/05-14-2018/MQTT_Messages.txt")	
	plug1, plug2, plug3 = RawData.get_pir_data()
	plug = Plugs(plug1, plug2, plug3)
	plug.plot_current()


	plug_df = plug.toDataFrame()

	print  plug_df['plug_1_current'][plug_df['plug_1_current'].notnull()]

	label_pd = read_labels("../../data/05-14-2018/labels.txt")
	plug_df = plug_df[(plug_df.TimeStamp >= label_pd['TimeStamp'].iloc[0]) & (plug_df.TimeStamp <= label_pd['TimeStamp'].iloc[-1])].reset_index(drop=True)
		
	plug_label_df = plug_df.join(label_pd.set_index('TimeStamp'), on='TimeStamp', how='outer')
	print  plug_label_df['plug_1_current'][plug_label_df['plug_1_current'].notnull()]
	

	plug_label_df = plug_label_df.replace({task: task_mapping[task]})

	X = plug_label_df[['SmartplugHeartRate','SmartplugAcc_X','SmartplugAcc_Y','SmartplugAcc_Z', \
						'SmartplugGyro_X', 'SmartplugGyro_Y', 'SmartplugGyro_Z']].astype('float').as_matrix()

	y = plug_label_df[task].astype('float').as_matrix()

	feature_extractor = FeatureExtractor(X, y)
	X, y = feature_extractor.get_extracted_features()
	X_train, X_test, y_train, y_test = train_test_split(X,y)
	print y_train

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
	