import sys
sys.path.append('../')

from utils.utils import *
from utils.preliminaries import *
from data_reading.read_data import *
import json

class FeatureExtractor(object):
	def __init__(self, X, y, window_size = 5, step = 1):
		self.window_size = window_size
		self.step = step
		self.X = X
		self.y = y

		self.features = ["mean", "std"]

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

				segments[i_segment] = np.concatenate((mean, stdev))
				labels[i_label] = self.y[end-1]
				i_label+=1
				i_segment+=1

		return segments, labels

	def get_extracted_features(self):
		return self.segment()

class PIR(object):
	def __init__(self, pir1, pir2, pir3):

		self.pir1 = []
		self.pir2 = []
		self.pir3 = []

		for line in pir1:
			payloads =  json.loads(line["Payload"])
			self.pir1.append((toDateTime(payloads['timestamp']), payloads["values"]))

		for line in pir2:
			payloads =  json.loads(line["Payload"])
			self.pir2.append((toDateTime(payloads['timestamp']), payloads["values"]))

		for line in pir3:
			try:
				payloads = eval(line['Payload'])
				self.pir3.append((toDateTime(line['TimeStamp'][:-7]), payloads["posy"], payloads["angle1"], payloads["posx"], payloads["angle2"]))
			except:
				pass

		self.pir1 = sorted_by_time_stamp(self.pir1)
		self.pir2 = sorted_by_time_stamp(self.pir2)
		self.pir3 = sorted_by_time_stamp(self.pir3)


		self.dates1 = [x[0] for x in self.pir1]
		self.dates2 = [x[0] for x in self.pir2]
		self.dates3 = [x[0] for x in self.pir3]

		self.pir1 = [x[1] for x in self.pir1]
		self.pir2 = [x[1] for x in self.pir2]

		self.posy = [x[1] for x in self.pir3]
		self.angle1 = [x[2] for x in self.pir3]
		self.posx = [x[3] for x in self.pir3]
		self.angle2 = [x[4] for x in self.pir3]


	def _print(self):
		f = open("./pir_data.txt", "w+")
		for l in self.pir1:
			f.write(str(l))
			f.write("\n")
		f.close()

	def toDataFrame(self):
		'''
		pir1_df = pd.DataFrame(
		    {'TimeStamp': self.dates1,
			'pir1': self.pir1
		    })

		pir2_df = pd.DataFrame(
		    {'TimeStamp': self.dates2,
			'pir2': self.pir2
		    })
		'''
		pir3_df = pd.DataFrame(
		    {'TimeStamp': self.dates3,
		     'posx': self.posx,
		     'posy': self.posy,
		     'angle1': self.angle1,
		     'angle2': self.angle2,
		    })

		'''
		pir_df = pir1_df.join(pir2_df.set_index('TimeStamp'), on='TimeStamp', how='outer')
		pir_df = pir_df.reset_index(drop=True)
		pir_df = pir3_df.join(pir_df.set_index('TimeStamp'), on='TimeStamp', how='outer')
		pir_df = pir_df.reset_index(drop=True)
		'''
		return pir3_df

if __name__ == '__main__':

	task = "basic_activities" 
	task_mapping = {"basic_activities": label_mapping, "kitchen_activities": kitchen_label_mapping}

	RawData  = RawDataDigester("../../data/05-14-2018/MQTT_Messages.txt")	
	pir1, pir2, pir3 = RawData.get_pir_data()

	pir = PIR(pir1, pir2, pir3)
	pir_df = pir.toDataFrame()

	label_pd = read_labels("../../data/05-14-2018/labels.txt")
	pir_df = pir_df[(pir_df.TimeStamp >= label_pd['TimeStamp'].iloc[0]) & (pir_df.TimeStamp <= label_pd['TimeStamp'].iloc[-1])].reset_index(drop=True)
	pir_label_df = pir_df.join(label_pd.set_index('TimeStamp'), on='TimeStamp', how='outer')
	pir_label_df = pir_label_df.replace({task: task_mapping[task]})

	pir_label_df = pir_label_df.fillna(method="bfill")
	pir_label_df = pir_label_df.fillna(0)

	X = pir_label_df[['posx','posy', 'angle1', 'angle2']].astype('float').as_matrix()
	y = pir_label_df[task].astype('float').as_matrix()

	feature_extractor = FeatureExtractor(X, y)
	X, y = feature_extractor.get_extracted_features()
	X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.1)

	print X_train.shape

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

