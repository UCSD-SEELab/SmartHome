import sys
sys.path.append('../')

from utils.utils import *
from utils.preliminaries import *
from data_reading.read_data import *


def main():
    clean_raw_data("../../data/MQTT_Messages_Anthony_11_16_18.txt", "anthony")


def clean_raw_data(path, subject=""):
    raw_data = RawDataDigester(path)

    watch_data = process_watch_data(raw_data)
    tv_plug, teapot_plug = process_plug_data(raw_data)
    airbeam_data = process_airbeam_data(raw_data)
    metasense_data = process_metasense_data(raw_data)

    if subject != "anthony":
        bulb1, kitchen_bulb = process_bulb_data(raw_data)
        rssi2 = process_ble_data(raw_data)

    pressuremat_data = process_pressuremat_data(raw_data)
    contact_data = process_contact_data(raw_data)
    misc_smartthings_data = process_misc_smartthings_data(raw_data)

    # create H5 dataset store for subject data
    out_path = "../../temp/{}_data.h5".format(subject)
    hdf_opts = {"complib": "blosc", "complevel": 9}
    watch_data.to_hdf(out_path, "watch", **hdf_opts)
    tv_plug.to_hdf(out_path, "tv_plug", **hdf_opts)
    teapot_plug.to_hdf(out_path, "teapot_plug", **hdf_opts)
    airbeam_data.to_hdf(out_path, "airbeam", **hdf_opts)
    metasense_data.to_hdf(out_path, "metasense", **hdf_opts)
    pressuremat_data.to_hdf(out_path, "pressuremat", **hdf_opts)

    for name, data in contact_data.iteritems():
        data.to_hdf(out_path, name, **hdf_opts)

    for name, data in misc_smartthings_data.iteritems():
        data.to_hdf(out_path, name, **hdf_opts)

    if subject != "anthony":
        kitchen_bulb.to_hdf(out_path, "kitchen_bulb", **hdf_opts)
        rssi2.to_hdf(out_path, "rssi2")


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

    # unfortunately we need to hackily assign timestamps to this data
    # since the collection process was bad. Data was sampled only during
    # the duration of the appropriate activity so we will just assign sequential
    # timestamps uniformly over that duration

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


def safe_join(left, right, **kwargs):
    validate = kwargs.pop("validate", "warn")
    expected = kwargs.pop("expected", "none")

    left_index = set(left.index)
    right_index = set(right.index)

    rr = right_index - left_index
    ll = left_index - left_index

    if expected == "all":
        if len(rr) or len(ll):
            print "WARNING: INDEXES DO NOT OVERLAP"
    elif expected == "right":
        if len(rr):
            print "WARNING: RIGHT INDEX IS NOT SUBSET OF LEFT"
    elif expected == "left":
        if len(ll): 
            print "WARNING: LEFT INDEX IS NOT SUBSET OF RIGHT"

    nobs_left_before = left.shape[0]
    nobs_right_before = right.shape[0]

    both = left.join(right, **kwargs)

    nobs_lost_left = nobs_left_before - both.shape[0]
    nobs_lost_right = nobs_right_before - both.shape[0]

    warn_left = nobs_lost_left > 0
    warn_right = nobs_lost_right > 0

    if validate is None:
        return both
    if warn_left:
        msg = "WARNING: LEFT LOST {} OBSERVATIONS DURING MERGE"
        print msg.format(nobs_lost_left)
    if warn_right:
        msg = "WARNING: RIGHT LOST {} OBSERVATIONS DURING MERGE"
        print msg.format(nobs_lost_right)

    if validate == "error":
        raise StandardError("Merge Error!")

    return both


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


def process_ble_data(raw_data, save_stub=""):
    rssi2 = unpack_features(raw_data.get_ble_data()[1])

    return rssi2


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
            data[varname].append(0 if "open" in str(item["message"]) else 1)
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
        "living_room_motion": process_active_stream(
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
