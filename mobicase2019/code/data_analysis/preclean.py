import sys
sys.path.append('../')

from utils.utils import *
from utils.preliminaries import *
from data_reading.read_data import *


def main():
    pass


def clean_raw_data(path, save_stub=""):
    raw_data = RawDataDigester(path)

    # watch data is sampled at the 
    # highest frequency so we rectify our data to that
    watch_data = process_watch_data(raw_data)
    plug1, plug2, tv_plug, teapot_plug = process_plug_data(raw_data)
    airbeam_data = process_airbeam_data(raw_data)
    metasense_data = process_metasense_data(raw_data)
    bulb1, kitchen_bulb = process_bulb_data(raw_data)
    rssi2 = process_ble_data(raw_data)
    pressuremat_data = process_pressuremat_data(raw_data)
    contact_data = process_contact_data(raw_data)
    misc_smartthings_data = process_misc_smartthings_data(raw_data)


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
    if save_stub != "":
        clean_data.to_csv(
            "../../temp/processed_watch_data_{}.csv".format(save_stub))

    return clean_data


def process_watch_ts(val):
    try:
        return datetime.datetime.strptime(val[6:], "%H:%M:%S:%f")
    except ValueError as e:
        print "Parsing Error: " + val
        raise e


def process_plug_data(raw_data, save_stub=""):
    plug1 = unpack_features(raw_data.get_plugs_data()[0])
    plug2 = unpack_features(raw_data.get_plugs_data()[1])

    # NOTE: plug3 contains no data
    # These plugs sample at a lower rate
    tv_plug = unpack_features(raw_data.get_plugs_data()[3])
    teapot_plug = unpack_features(raw_data.get_plugs_data()[4])

    if save_stub != "":
        plugs12.to_csv(
            "../../temp/processed_plug12_data_{}.csv".format(save_stub))
        plugs_tv_tea.to_csv(
            "../../temp/processed_plugs_tv_tea_data_{}.csv".format(save_stub))

    return plug1, plug2, tv_plug, teapot_plug

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

    if save_stub != "":
        clean_data.to_csv(
            "../../temp/processed_plug_data_{}.csv".format(save_stub))
    return clean_data


def process_metasense_data(raw_data, save_stub=""):
    clean_data = unpack_features(raw_data.get_metasense_data())

    if save_stub != "":
        clean_data.to_csv(
            "../../temp/processed_metasense_data_{}.csv".format(save_stub))
    return clean_data


def process_bulb_data(raw_data, save_stub=""):
    bulb1 = unpack_features(raw_data.get_bulb_data[0])
    kitchen_bulb = unpack_features(raw_data.get_bulb_data[1])

    if save_stub != "":
        bulb1.to_csv(
            "../../temp/processed_bulb_data_{}.csv".format(save_stub))
        kitchen_bulb.to_csv(
            "../../temp/processed_kitchen_bulb_data_{}.csv".format(save_stub))

    return bulb1, kitchen_bulb


def process_ble_data(raw_data, save_stub=""):
    rssi2 = unpack_features(raw_data.get_ble_data()[1])

    if save_stub != "":
        rssi2.to_csv("../../temp/processed_rssi2_data_{}".format(save_stub))

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
        data["timestamp"] = process_watch_ts(item["timestamp"])

    clean_data = pd.DataFrame(data).set_index("timestamp")

    if save_stub != "":
        clean_data.to_csv(
            "../../temp/processed_pressuremat_data_{}.csv".format(save_stub))


def process_contact_data(raw_data, varname):
    contact_sensor_vars = {"smartthings/Cabinet 1/contact": "cabinet1",
                           "smartthings/Cabinet 2/contact": "cabinet2",
                           "smartthings/Drawer 1/contact": "drawer1",
                           "smartthings/Drawer 2/contact": "drawer2",
                           "smartthings/Fridge/contact": "fridge",
                           "smartthings/Pantry/contact": "pantry"}

    contact_data = {}
    for data_stream_name, clean_name in contect_sensor_vars.iteritems():
        stream = raw_data.data[data_stream_name]
        varname = "{}_contact".format(varname)
        data = {varname: [], "timestamp": []}
        for item in stream:
            data[varname].append(0 if "open" in item["message"] else 1)
            data["timestamp"].append(process_watch_ts(item["timestamp"]))

        contact_data[clean_name] = pd.DataFrame(data).set_index("timestamp")

    return contact_data


def process_misc_smartthings_data(raw_data, save_stub):
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
