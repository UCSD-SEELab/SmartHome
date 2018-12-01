import sys
sys.path.append('../')

from preliminaries.preliminaries import *
from read_data import *

"""
preclean.py - this program extracts the raw MQTT message streams and converts
them into Pandas data frames. The class RawDataDigester is used to extract the
raw data into Python data structures. The return value of this class is
a dictionary whose keys are the names of sensor topics, and whose values are
lists containing the raw snesor messages.
"""

def main():
    clean_raw_data("../../data/MQTT_Messages_subject1_11_16_18.txt", "subject1")
    clean_raw_data("../../data/MQTT_Messages_subject2_11-15-18.txt", "subject2")

def clean_raw_data(path, subject=""):
    # Extract the raw sensor messages
    raw_data = RawDataDigester(path)

    # extract the labels. We only need to keep the start time of each
    # activity because end time can be determined as when the next
    # activity started.
    labels = process_labels(
            raw_data
        ).groupby(
            "label", as_index=False
        ).first().set_index("timestamp")

    # Process the various raw data streams. This mostly just
    # takes the raw data and converts it into a Pandas data frame
    watch_data = process_watch_data(raw_data)
    tv_plug, teapot_plug = process_plug_data(raw_data)
    airbeam_data = process_airbeam_data(raw_data)
    metasense_data = process_metasense_data(raw_data)
    crk_data = process_crk_data(raw_data)
    pir_data = process_pir_data(raw_data)

    pressuremat_data = process_pressuremat_data(raw_data)
    contact_data = process_contact_data(raw_data)
    misc_smartthings_data = process_misc_smartthings_data(raw_data)

    # create H5 dataset store for subject data
    out_path = "../../temp/{}_data.h5".format(subject)
    hdf_opts = {"complib": "blosc", "complevel": 9}
    labels.to_hdf(out_path, "labels", **hdf_opts)
    watch_data.to_hdf(out_path, "watch", **hdf_opts)
    tv_plug.to_hdf(out_path, "tv_plug", **hdf_opts)
    teapot_plug.to_hdf(out_path, "teapot_plug", **hdf_opts)
    airbeam_data.to_hdf(out_path, "airbeam", **hdf_opts)
    metasense_data.to_hdf(out_path, "metasense", **hdf_opts)
    pressuremat_data.to_hdf(out_path, "pressuremat", **hdf_opts)
    crk_data.to_hdf(out_path, "location", **hdf_opts)
    pir_data.to_hdf(out_path, "pir1", **hdf_opts)

    for name, data in contact_data.iteritems():
        data.to_hdf(out_path, name, **hdf_opts)

    for name, data in misc_smartthings_data.iteritems():
        data.to_hdf(out_path, name, **hdf_opts)


def process_labels(raw_data):
    labels = raw_data.get_labels()

    data = {
        "label": [x["activity"] for x in labels if bool(x["isActive"])], 
        "timestamp": [process_watch_ts(
            x["timestamp"]) for x in labels if bool(x["isActive"])]
    }

    return pd.DataFrame(data)


def process_pir_data(raw_data):
    pir = raw_data.get_pir_data()[0]
    vals = [extract_pir_values(x["message"]) for x in pir]
    mat = np.concatenate(vals).reshape(-1,vals[0].size)

    clean_data = pd.DataFrame(mat)
    clean_data.columns = map(lambda x: "pir1_{}".format(x), range(mat.shape[1]))
    clean_data["timestamp"] = [process_watch_ts(x["timestamp"]) for x in pir]

    return clean_data.set_index("timestamp")

def extract_pir_values(msg):
    return np.array(json.loads(msg)["values"])


def process_watch_data(raw_data, save_stub=""):
    watch = raw_data.get_watch_data()

    data = {"step": [], "heart_rate_bpm": [],
            "accel_X": [], "accel_Y": [], "accel_Z": [],
            "gyro_X": [], "gyro_Y": [], "gyro_Z": [], 
            "timestamp": []}

    for parsed in watch:
        data["step"].append(float(parsed[1]))
        data["heart_rate_bpm"].append(float(parsed[3]))
        data["accel_X"].append(float(parsed[5]))
        data["accel_Y"].append(float(parsed[6]))
        data["accel_Z"].append(float(parsed[7]))
        data["gyro_X"].append(float(parsed[9]))
        data["gyro_Y"].append(float(parsed[10]))
        data["gyro_Z"].append(float(parsed[11]))
        data["timestamp"].append(process_watch_ts(parsed[13]))

    clean_data = pd.DataFrame(data).set_index("timestamp").sort_index()

    return clean_data


def process_watch_ts(val):
    try:
        return datetime.datetime.strptime(val[6:], "%H:%M:%S:%f")
    except ValueError as e:
        print "Parsing Error: " + val
        raise e


def process_plug_data(raw_data):
    # apparently these features are junk
    #plug1 = unpack_features(raw_data.get_plugs_data()[0])
    #plug2 = unpack_features(raw_data.get_plugs_data()[1])

    tv_plug = unpack_features(raw_data.get_plugs_data()[3])
    teapot_plug = unpack_features(raw_data.get_plugs_data()[4])

    return tv_plug, teapot_plug

def unpack_features(messages, dtypes=None, default_dtype=np.float64):
    if dtypes is None:
        dtypes = {}
    
    data = {x: [] for x in messages[0].keys()}
    for item in messages:
        for name, value in item.iteritems():
            if name == "timestamp":
                data[name].append(process_watch_ts(value))
            elif name in dtypes:
                data[name].append(dtypes[name](value))
            else:
                data[name].append(default_dtype(value))

    processed = pd.DataFrame(data)
    return processed.set_index("timestamp") if "timestamp" in data else processed


def process_airbeam_data(raw_data, save_stub=""):
    clean_data = unpack_features(raw_data.get_airbeam_data())

    return clean_data


def process_metasense_data(raw_data, save_stub=""):
    clean_data = unpack_features(raw_data.get_metasense_data())

    return clean_data


def process_bulb_data(raw_data, save_stub=""):
    bulb1 = unpack_features(raw_data.get_bulb_data()[0])
    kitchen_bulb = unpack_features(raw_data.get_bulb_data()[1])

    return bulb1, kitchen_bulb


def process_crk_data(raw_data, save_stub=""):
    crk = raw_data.data["crk"]
    messages = map(lambda x: parse_rssi_message(x["message"]), crk)
    data = {
        "kitchen1_crk": [x["rssi1"] for x in messages],
        "kitchen2_crk": [x["rssi2"] for x in messages],
        "dining_room_crk": [x["rssi3"] for x in messages],
        "living_room1_crk": [x["rssi4"] for x in messages],
        "living_room2_crk": [x["rssi5"] for x in messages],
        "timestamp": [process_watch_ts(x["timestamp"]) for x in crk] 
    }

    crk = pd.DataFrame(data).set_index("timestamp")
    return crk


def parse_rssi_message(msg):
    cleaned = msg.replace("{","").replace("}","").split(",")
    data = {}
    for item in cleaned:
        name, value = item.split(": ")
        data[name.replace(" ","")] = float(value)
    return data


def process_pressuremat_data(raw_data, save_stub=""):
    pressure_mat = raw_data.get_pressuremat_data()

    rownames = filter(lambda x: "timestamp" not in x, pressure_mat[0].keys())
    data = {"pressuremat_sum": [], "timestamp": []}

    for item in pressure_mat:
        accum = 0.0
        for r in rownames:
            accum += np.sum(item[r])
        data["pressuremat_sum"].append(accum)
        data["timestamp"].append(process_watch_ts(item["timestamp"]))

    clean_data = pd.DataFrame(data).set_index("timestamp")
    return clean_data


def process_contact_data(raw_data):
    contact_sensor_vars = {"smartthings/Cabinet 1/contact": "cabinet1",
                           "smartthings/Cabinet 2/contact": "cabinet2",
                           "smartthings/Drawer 1/contact": "drawer1",
                           "smartthings/Drawer 2/contact": "drawer2",
                           "smartthings/Fridge/contact": "fridge",
                           "smartthings/Pantry/contact": "pantry"}

    contact_data = {}
    for data_stream_name, clean_name in contact_sensor_vars.iteritems():
        stream = raw_data.data[data_stream_name]
        varname = "{}_contact".format(clean_name)
        data = {varname: [], "timestamp": []}
        for item in stream:
            data[varname].append(1 if "open" in str(item["message"]) else 0)
            data["timestamp"].append(process_watch_ts(item["timestamp"]))

        contact_data[varname] = pd.DataFrame(data).set_index("timestamp")

    return contact_data


def process_misc_smartthings_data(raw_data):
    return {
        "dining_room_motion": process_active_stream(
            raw_data, "Diningroom MultiSensor 6/motion", "dining_room_motion"),
        "living_room_motion": process_active_stream(
            raw_data, "Living Room Motion Sensor/motion", "living_room_motion"),
        "kitchen_door_acceleration": process_active_stream(
            raw_data, "Kitchen Door/acceleration", "kitchen_door_acceleration"),
        "corridor_motion": process_active_stream(
            raw_data, "Living Room Corridor Motion Sensor/motion",
            "living_room_corridor_motion")
    }


def process_active_stream(raw_data, name, varname):
    stream = raw_data.data["smartthings/{}".format(name)]
    data = {
        varname: [x["message"] == "active" for x in stream],
        "timestamp": [process_watch_ts(x["timestamp"]) for x in stream]
    }

    return pd.DataFrame(data)

if __name__=="__main__":
    main()
