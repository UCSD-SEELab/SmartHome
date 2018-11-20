import sys
import tables
import pywt
sys.path.append('../')

from tabulate import tabulate
from utils.utils import *
from utils.preliminaries import *

CONTINUOUS_FEATURE_EXTRACTORS = [np.mean, np.var]

def get_preprocessed_data(exclude_sensors=None, verbose=False, use_wavelets=False):
    anthony_data, sensors = build_data(
        "../../temp/anthony_data.h5", 30, "anthony", exclude_sensors, use_wavelets)
    yunhui_data, _ = build_data(
        "../../temp/yunhui_data.h5", 300, "yunhui", exclude_sensors, use_wavelets)

    if verbose:
        print "===============> BEFORE NORMALIZING <================="

        print "++++++++++++++++ ANTHONY ++++++++++++++++"
        print anthony_data.mean()
        print anthony_data.var()

        print "++++++++++++++++ YUNHUI +++++++++++++++++"
        print yunhui_data.mean()
        print yunhui_data.var()
    
    normalize_continuous_cols(anthony_data)
    normalize_continuous_cols(yunhui_data)

    if verbose:    
        print "===============> AFTER NORMALIZING <================="

        print "++++++++++++++++ ANTHONY ++++++++++++++++"
        print anthony_data.mean()
        print anthony_data.var()

        print "++++++++++++++++ YUNHUI +++++++++++++++++"
        print yunhui_data.mean()
        print yunhui_data.var()
    
        anthony_data.describe().to_csv("../../temp/anthony_stats.csv")
        yunhui_data.describe().to_csv("../../temp/yunhui_stats.csv")

    return anthony_data, yunhui_data, sensors

def build_data(path, window_size, subject, use_wavelets, exclude_sensors=None):
    watch = pd.read_hdf(path, "watch")
    labels = pd.read_hdf(path, "labels")
    tv_plug = pd.read_hdf(path, "tv_plug")
    teapot_plug = pd.read_hdf(path, "teapot_plug")
    pressuremat = pd.read_hdf(path, "pressuremat")
    metasense = pd.read_hdf(path, "metasense")
    airbeam = pd.read_hdf(path, "airbeam")
    location = pd.read_hdf(path, "location")
        
    '''
    dining_room_motion = pd.read_hdf(path, "dining_room_motion")
    living_room_motion = pd.read_hdf(path, "living_room_motion")
    kitchen_door_acceleration = pd.read_hdf(path, "kitchen_door_acceleration")
    corridor_motion = pd.read_hdf(path, "corridor_motion")
    '''

    cabinet1_contact = pd.read_hdf(path, "cabinet1_contact")
    cabinet2_contact = pd.read_hdf(path, "cabinet2_contact")
    drawer1_contact = pd.read_hdf(path, "drawer1_contact")
    drawer2_contact = pd.read_hdf(path, "drawer2_contact")
    fridge_contact = pd.read_hdf(path, "fridge_contact")

    #print path
    #print labels

    #dining_room_motion = pd.read_hdf(
    #    path, "dining_room_motion").set_index("timestamp")
    # living_room_motion = pd.read_hdf(
    #     path, "living_room_motion").set_index("timestamp")

    watch_coarse = process_watch(watch, window_size, use_wavelets)
    labels_coarse = process_labels(watch, labels, window_size)
    location_coarse = process_location_data(watch, location, window_size)
    metasense_coarse = coarsen_continuous_features(metasense, watch, window_size)
    tv_plug_coarse = coarsen_continuous_features(
        tv_plug["current"].to_frame(), watch, window_size)
    teapot_plug_coarse = coarsen_continuous_features(
        teapot_plug["current"].to_frame(), watch, window_size)
    pressuremat_coarse = coarsen_continuous_features(
        pressuremat, watch, window_size)
    airbeam_coarse = coarsen_continuous_features(
        airbeam, watch, window_size)


    cabinet1_coarse = process_binary_features(
        cabinet1_contact, watch, "cabinet1", window_size)

    cabinet1_coarse = process_binary_features(
        cabinet1_contact, watch, "cabinet1", window_size)
    cabinet2_coarse = process_binary_features(
        cabinet2_contact, watch, "cabinet2", window_size)
    drawer1_coarse = process_binary_features(
        drawer1_contact, watch, "drawer1", window_size)
    drawer2_coarse = process_binary_features(
        drawer2_contact, watch, "drawer2", window_size)
    fridge_coarse = process_binary_features(
        fridge_contact, watch, "fridge", window_size)

    all_sensors = collections.OrderedDict([
                    # kitchen
                   ("teapot_plug", teapot_plug_coarse), 
                   ("pressuremat", pressuremat_coarse), 
                   ("metasense", metasense_coarse),

                   # smartthings
                   ("cabinet1", cabinet1_coarse),
                   ("cabinet2", cabinet2_coarse),
                   ("drawer1", drawer1_coarse),  
                   ("drawer2", drawer2_coarse), 
                   ("fridge", fridge_coarse),
 
                   #living room: 
                   ("tv_plug", tv_plug_coarse), 

                   # smart watch
                   ("location", location_coarse), 
                   ("watch", watch_coarse),

                   # not used
                   ("airbeam", airbeam_coarse),
                   ])

    exclude_sensors = [] if exclude_sensors is None else exclude_sensors
    all_data = labels_coarse
    for sensor in all_sensors:
        if sensor in exclude_sensors:
            continue
        if sensor in ['tv_plug', "teapot_plug", "metasense", "airbeam"]:
            rsuffix = "_{}".format(sensor)
        else:
            rsuffix = ""
        data = all_sensors[sensor]
        data.columns = map(lambda x: "{}_{}".format(sensor, x), data.columns)
        all_data = all_data.join(data, rsuffix=rsuffix)

    return all_data, all_sensors.keys()


def flatten_multiindex(index):
    return ["{}_{}".format(x,y) for x,y in index.tolist()]


def process_labels(watch, labels, window_size):
    obs_before = watch.shape[0]
    both = watch.loc[:,"step"].to_frame().join(
            labels, how="left"
        ).drop("step", axis="columns").fillna(method="ffill")
    assert obs_before == both.shape[0], "Merge Error"

    both = both.dropna()
    both["label_numeric"] = -1
    for label in LABEL_ENCODING:
        both.loc[both["label"] == label,"label_numeric"] = LABEL_ENCODING[label]

    both = both.drop("label", axis="columns")
    labels_coarse = both.rolling(window_size).mean().dropna()
    labels_coarse["label"] = labels_coarse["label_numeric"].round()
    labels_coarse = labels_coarse.drop("label_numeric", axis="columns")

    labels_coarse = labels_coarse.reset_index()
    labels_coarse["change"] = labels_coarse["label"].diff() != 0
    labels_coarse.loc[:,"change"][-1] = False
    labels_coarse["change_time"] = labels_coarse["timestamp"]
    labels_coarse.loc[labels_coarse["change"] == False,"change_time"] = np.nan
    labels_coarse = labels_coarse.fillna(method="ffill")
    labels_coarse["elapsed"] = labels_coarse["timestamp"] - labels_coarse["change_time"]
    to_keep = labels_coarse["elapsed"] > pd.Timedelta("30 seconds")

    labels_coarse = labels_coarse.loc[to_keep,["timestamp","label"]].set_index(
        "timestamp")

    return labels_coarse


def process_location_data(watch, location, window_size):
    location["closest"] = location.apply(
        lambda x: np.argmax(x.values), axis="columns")
    location = location.groupby(level=0)["closest"].first().to_frame()

    obs_before = watch.shape[0]
    both = watch.loc[:,"step"].to_frame().join(
            location, how="left"
        ).drop("step", axis="columns").fillna(method="ffill")
    assert obs_before == both.shape[0], "Merge Error"

    location_coarse = both.rolling(window_size).mean().dropna()
    location_coarse["closest"] = location_coarse["closest"].round()

    location_coarse["in_kitchen"] = location_coarse["closest"] < 2
    location_coarse["in_dining_room"] = location_coarse["closest"] == 2
    location_coarse["in_living_room"] = location_coarse["closest"] > 2

    varnames = ["in_kitchen","in_dining_room","in_living_room"]
    return location_coarse.loc[:,varnames]


def process_watch(watch, window_size, use_wavelet_transform=False):
    if use_wavelet_transform:
        # compute a highband and lowband wavelet decomposition
        # and extract features on the bands independently

        accel = process_wavelet_transform(watch, "accel")
        gyro = process_wavelet_transform(watch, "gyro")
    else:
        accel = process_accel_gyro(
            watch.loc[:,["accel_X","accel_Y","accel_Z"]], window_size, "_accel")
        gyro = process_accel_gyro(
            watch.loc[:,["gyro_X","gyro_Y","gyro_Z"]], window_size, "_gyro")

    accel_energy = compute_energy(
        watch.loc[:,["accel_X","accel_Y","accel_Z"]], window_size, "_accel")
    gyro_energy = compute_energy(
        watch.loc[:,["gyro_X","gyro_Y","gyro_Z"]], window_size, "_gyro")

    return accel.join(gyro).join(accel_energy).join(gyro_energy)


def process_wavelet_transform(watch, stub):
    dwtX = pydwt(watch["{}_X".format(stub)], "haar")
    dwtY = pydwt(watch["{}_Y".format(stub)], "haar")
    hwtZ = pydwt(watch["{}_Z".format(stub)], "haar")

    lowband = pd.DataFrame({
        "timestamp": watch.index,
        "{}_X".format(stub): dwtX[0],
        "{}_Y".format(stub): dwtY[0],
        "{}_Z".format(stub): dwtZ[0]
    }).set_index("timestamp")

    highband = pd.DataFrame({
        "timestamp": watch.index,
        "{}_X".format(stub): dwtX[1],
        "{}_Y".format(stub): dwtY[1],
        "{}_Z".format(stub): dwtZ[1]
    }).set_index("timestamp")

    ll = process_accelerometer(
        accel_lowband, window_size, "_{}_lowband".format(stub))
    hh = process_accelerometer(
        accel_highband, window_size, "_{}_highband".format(stub))
    return ll.join(hh)

def process_accel_gyro(accel, window_size, stub=""):
    A = accel.values
    data = {
        "timestamp": [],
        "mean_x": [],
        "mean_y": [],
        "mean_z": [],
        "var_x": [],
        "var_y": [],
        "var_z": [],
        "corr_xy": [],
        "corr_xz": [],
        "corr_yz": []
    }

    for ix in range(window_size, accel.shape[0]):
        ww = A[ix-window_size:ix,:]
        data["timestamp"].append(accel.index[ix])

        mu = ww.mean(axis=0)
        sigma = ww.var(axis=0)
        data["mean_x"].append(mu[0])
        data["mean_y"].append(mu[1])
        data["mean_z"].append(mu[2])
        data["var_x"].append(sigma[0])
        data["var_y"].append(sigma[1])
        data["var_z"].append(sigma[2])
        data["corr_xy"].append(np.corrcoef(ww[:,0], ww[:,1])[1,0])
        data["corr_xz"].append(np.corrcoef(ww[:,0], ww[:,2])[1,0])
        data["corr_yz"].append(np.corrcoef(ww[:,1], ww[:,2])[1,0])
    
    clean_data = pd.DataFrame(data).set_index("timestamp")
    clean_data.columns = map(lambda x: "{}{}".format(x, stub), clean_data.columns)
    return clean_data


def compute_energy(data, window_size, stub):
    A = data.values
    out = {
        "timestamp": [],
        "energy": []
    }

    for ix in range(window_size, data.shape[0]):
        ww = A[ix-window_size:ix,:]
        out["timestamp"].append(data.index[ix])
        TT = np.fft.fft(ww, axis=0)
        out["energy"].append(np.abs(TT).sum()*(1.0/window_size))

    clean_data = pd.DataFrame(out).set_index("timestamp")
    clean_data.columns = map(lambda x: "{}{}".format(x, stub), clean_data.columns)
    return clean_data


def coarsen_continuous_features(data, watch, window_size, fill_method="ffill"):
    data_grouped = data.groupby(level=0).mean()

    obs_before = watch.shape[0]
    both = watch.loc[:,"step"].to_frame().join(
            data_grouped, how="left"
        ).drop("step", axis="columns").fillna(method=fill_method)
    assert obs_before == both.shape[0], "Merge Error"

    features = [np.min, np.max, np.mean, np.var]
    data_coarsened = both.rolling(
        window_size).agg(CONTINUOUS_FEATURE_EXTRACTORS).dropna()
    data_coarsened.columns = flatten_multiindex(data_coarsened.columns)
    return data_coarsened

def process_binary_features(contact, watch, varname, window_size):
    contact = contact.groupby(level=0).first()
    obs_before = watch.shape[0]
    both = watch.loc[:,"step"].to_frame().join(
            contact, how="left"
        ).drop("step", axis="columns").fillna(
            method="ffill"
        ).fillna(0).sort_index()
    assert obs_before - both.shape[0] == 0, "Merge Error"

    both_coarsened = both.rolling(
        window_size).max().dropna().sort_index().reset_index()

    # now compute the number of seconds since the last "open" event
    both_coarsened["last_open"] = both_coarsened["timestamp"]
    where = both_coarsened["{}_contact".format(varname)] == 0
    both_coarsened.loc[where, "last_open"] = np.nan
    both_coarsened["last_open"] = both_coarsened["last_open"].fillna(
        method="ffill").fillna(method="bfill")
    both_coarsened["elapsed"] = (
        both_coarsened["timestamp"].sub(both_coarsened["last_open"])
    )
    both_coarsened = both_coarsened.set_index("timestamp")

    both_coarsened["{}_1min".format(varname)] = (
        np.logical_and(both_coarsened["elapsed"] <= pd.Timedelta("1 min"),
            both_coarsened["elapsed"] >= pd.Timedelta("0 min"))).astype(np.int64)
    both_coarsened["{}_5min".format(varname)] = (
        np.logical_and(both_coarsened["elapsed"] <= pd.Timedelta("5 min"),
            both_coarsened["elapsed"] >= pd.Timedelta("0 min"))).astype(np.int64)
    both_coarsened["{}_10min".format(varname)] = (
        np.logical_and(both_coarsened["elapsed"] <= pd.Timedelta("10 min"),
            both_coarsened["elapsed"] >= pd.Timedelta("0 min"))).astype(np.int64)
    varnames = map(lambda x: x.format(varname), ["{}_1min","{}_5min","{}_10min"])
    return both_coarsened.loc[:,varnames]


def normalize_continuous_cols(data):
    for col in data.columns:
        if col == "label" or data[col].dtype != np.float64:
            continue
        data[col] = (data[col] - data[col].mean()) / data[col].std() 

if __name__=="__main__":
    anthony_data, yunhui_data, _ = get_preprocessed_data(exclude_sensors=['airbeam'])
    anthony_data.to_hdf("../../temp/anthony_data_processed.h5")
    yunhui_data.to_hdf("../../temp/yunhui_data_processed.h5")

    anthony_data, yunhui_data, _ = get_preprocessed_data(exclude_sensors=['airbeam'], True)
    anthony_data.to_hdf("../../temp/anthony_data_processed_wavelets.h5")
    yunhui_data.to_hdf("../../temp/yunhui_data_processed_wavelets.h5")
