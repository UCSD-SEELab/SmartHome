import sys
sys.path.append('../')

from utils.utils import *
from utils.preliminaries import *

# **** Processing procedure for raw data ****
class RawDataDigester(object):
	def __init__(self, path):
		f = open(path, "r")
		self.data = defaultdict(list)
		self.labels = []

		for line in f:
			line = line.strip().split("#Message:#")
			topic = line[0].split('#')[1]
			if topic == 'watch':
				message = line[1].split(";")
			else:
				try:
					message =  eval(line[1])
				except NameError:
					message = line[1]
			
			if topic == 'labels':
				self.labels.append(message)
			else:
				if topic.lower() == "smartthings":
						self.data["smartthings"].append(message)
				else:
						self.data[topic].append(message)

	def get_watch_data(self):
		return self.data['watch']

	def get_pir_data(self):
		return self.data['pir/raw/1'], self.data['pir/raw/2'], self.data['pir/angular_locations']

	def get_plugs_data(self):
		return self.data['plug1'], self.data['plug2'], self.data['plug3']

	def get_ble_data(self):
		return self.data['rssi1'], self.data['rssi2'], self.data['rssi3']

	def get_smartthings_data(self):
		return self.data['smartthings']

	def get_pressuremat_data(self):
		return self.data['PressureMat/raw']

	def get_labels(self):
		return self.labels

'''
def read_labels(file):
	labels = open(file, "r")
	date_list = []
	basic_activities_list = []
	kitchen_activities_list = []

	for line in labels:
		line = line.strip().split(" ", 3)
		date_list.append(toDateTime(line[0]+ " " + line[1]))
		basic_activities_list.append(line[2])
		kitchen_activities_list.append(line[3])

	label_pd = pd.DataFrame(
		 {'TimeStamp': date_list,
		  'basic_activities': basic_activities_list,
		  'kitchen_activities': kitchen_activities_list
	    })

	label_pd = label_pd.iloc[np.repeat(np.arange(len(label_pd)), 3)]
	for row_index in range(label_pd.shape[0]):
		if row_index % 3 == 1:
			label_pd['TimeStamp'].iloc[row_index] = label_pd['TimeStamp'].iloc[row_index] + datetime.timedelta(seconds = 1)
		elif row_index % 3 == 2:
			label_pd['TimeStamp'].iloc[row_index] = label_pd['TimeStamp'].iloc[row_index] + datetime.timedelta(seconds = 2)
	label_pd = label_pd.reset_index(drop=True)

	return label_pd
'''

if __name__ == '__main__':
	p = "../../data/MQTT_Messages_Yunhui_11-15-18.txt"

	d = RawDataDigester(p)
	labels = d.get_labels()
	print len(labels)
