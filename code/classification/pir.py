from utils import *
from preliminaries import *
from read_data import *

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


class PIR(object):
	def __init__(self, pir1, pir2, pir3):
		self.pir1 = pir1
		self.pir2 = pir2
		self.pir3 = pir3
		print self.pir1

	def _print(self):
		f = open("./pir_data.txt", "w+")
		for l in self.pir1:
			f.write(str(l))
			f.write("\n")
		f.close()



if __name__ == '__main__':


	task = "basic_activities" 
	task_mapping = {"basic_activities": label_mapping, "kitchen_activities": kitchen_label_mapping}

	RawData  = RawDataDigester("../data/MQTT_Messages.txt")	
	pir1, pir2, pir3 = RawData.get_pir_data()
	pir = PIR(pir1, pir2, pir3)
	
	'''
	pir._print


	pir_df = pir.toDataFrame()

	print  pir_df['pir_1_current'][pir_df['pir_1_current'].notnull()]

	label_pd = read_labels("../data/labels.txt")
	pir_df = pir_df[(pir_df.TimeStamp >= label_pd['TimeStamp'].iloc[0]) & (pir_df.TimeStamp <= label_pd['TimeStamp'].iloc[-1])].reset_index(drop=True)
		
	pir_label_df = pir_df.join(label_pd.set_index('TimeStamp'), on='TimeStamp', how='outer')
	print  pir_label_df['pir_1_current'][pir_label_df['pir_1_current'].notnull()]
	

	pir_label_df = pir_label_df.replace({task: task_mapping[task]})

	X = pir_label_df[['SmartpirHeartRate','SmartpirAcc_X','SmartpirAcc_Y','SmartpirAcc_Z', \
						'SmartpirGyro_X', 'SmartpirGyro_Y', 'SmartpirGyro_Z']].astype('float').as_matrix()

	y = pir_label_df[task].astype('float').as_matrix()

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
	'''