import sys
sys.path.append('../')

from preliminaries.preliminaries import *

# **** Processing procedure for raw data ****
class RawDataDigester(object):
    def __init__(self, path):
        with open(path) as fh:
            lines = fh.readlines()

        self.data = defaultdict(list)
        self.labels = []

        # need to figure out the first timestamp
        curr_timestamp = None
        for line in lines:
            if line[:11] == "Timestamp:#":
                line = line[31:]
            line = line.strip().split("#Message:#")
            topic = line[0].split('#')[1]

            if topic == 'watch':
                message = line[1].split(";")
                curr_timestamp = message[-1]

            if curr_timestamp is not None:
                break


        parse_errors = []
        counter = 0
        for line in lines:
            if line[:11] == "Timestamp:#":
                line = line[31:]
            line = line.strip().split("#Message:#")
            topic = line[0].split('#')[1]
            try:
                if len(topic) == 0 or topic[-1] == "/":
                    parse_errors.append("#Message:#".join(line))
                    continue

                if topic == 'watch':
                    message = line[1].split(";")
                    curr_timestamp = message[-1]
                else:
                    try:
                        message =  eval(line[1])
                    except NameError:
                        message = line[1]
                    try:
                        message["timestamp"] = curr_timestamp
                    except TypeError:
                        message = {"timestamp": curr_timestamp, "message": message}
                
                if topic == 'labels':
                    self.labels.append(message)
                else:
                    self.data[topic].append(message)
            except Exception as e:
                print "PARSE ERROR IN: {}".format(line)
                raise e
            counter += 1

        if len(parse_errors) > 0:
            print "WARNING: {} MESSAGES COULD NOT BE PARSED".format(len(parse_errors))
            for msg in parse_errors:
                print msg
        print "PARSED: {} MESSAGES".format(counter)

    def get_watch_data(self):
        return self.data['watch']

    def get_pir_data(self):
        return self.data['pir/raw/1'], self.data['pir/raw/2']

    def get_plugs_data(self):
        return self.data['plug1'], self.data['plug2'], self.data['plug3'], self.data["tv_plug"], self.data["teapot_plug"]

    def get_kitchen_data(self):
        return self.data["Kitchen"]

    def get_ble_data(self):
        return self.data['rssi1'], self.data['rssi2'], self.data['rssi3']

    def get_airbeam_data(self):
        return self.data["AirBeam-8042/raw"]

    def get_crk_data(self):
        return self.data["crk"]

    def get_metasense_data(self):
        return self.data["MetaSense-E7E1/raw"]

    def get_smartthings_data(self):
        return self.data['smartthings']

    def get_bulb_data(self):
        return self.data["bulb"], self.data["kitchen_bulb"]

    def get_pressuremat_data(self):
        return self.data['PressureMat/raw']

    def list_topics(self):
        return self.data.keys()

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
