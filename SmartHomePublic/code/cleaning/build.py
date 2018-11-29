import sys
import tables
import pywt
sys.path.append('../')

from tabulate import tabulate
from preliminaries.preliminaries import *
from scipy.stats import mode

CONTINUOUS_FEATURE_EXTRACTORS = [np.mean, np.var]

def main():
    # flags for both datasets
    exclude_sensors = ["airbeam"]
    use_wavelets = False

    subject1_data, sensors = build_data(
        "../../temp/subject1_data.h5", 30, "subject1", 
        use_wavelets, exclude_sensors=exclude_sensors, write_dists="train", 
        exclude_transitions=True)
    subject2_data, _ = build_data(
        "../../temp/subject2_data.h5", 30, "subject2", 
        use_wavelets, exclude_sensors=exclude_sensors, 
        exclude_transitions=False)

    print "===============> BEFORE NORMALIZING <================="

    print "++++++++++++++++ subject1 ++++++++++++++++"
    print subject1_data.mean()
    print subject1_data.var()

    print "++++++++++++++++ subject2 +++++++++++++++++"
    print subject2_data.mean()
    print subject2_data.var()
    
    mu, sigma = normalize_continuous_cols(subject1_data)
    normalize_continuous_cols(subject2_data, mu, sigma)
  
    print "===============> AFTER NORMALIZING <================="

    print "++++++++++++++++ subject1 ++++++++++++++++"
    print subject1_data.mean()
    print subject1_data.var()

    print "++++++++++++++++ subject2 +++++++++++++++++"
    print subject2_data.mean()
    print subject2_data.var()

    subject1_data.describe().to_csv("../../temp/subject1_stats.csv")
    subject2_data.describe().to_csv("../../temp/subject2_stats.csv")

    subject1_data.to_hdf("../../temp/data_processed.h5", "subject1")
    subject2_data.to_hdf("../../temp/data_processed.h5", "subject2")

    return subject1_data, subject2_data, sensors

def build_data(path, window_size, subject, use_wavelets, 
               write_dists=None, exclude_sensors=None, 
               exclude_transitions=False):
    
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

    #dining_room_motion = pd.read_hdf(
    #    path, "dining_room_motion").set_index("timestamp")
    # living_room_motion = pd.read_hdf(
    #     path, "living_room_motion").set_index("timestamp")

    watch_coarse = process_watch(watch, window_size, use_wavelets)
    labels_coarse = process_labels(watch, labels, window_size, exclude_transitions)
    location_coarse = process_location_data(watch, location, window_size)
    metasense = preprocess_metasense(metasense)
    metasense_coarse = coarsen_continuous_features(metasense, watch, 3)
    tv_plug_coarse = coarsen_continuous_features(
        tv_plug["current"].to_frame(), watch, 3)
    teapot_plug_coarse = coarsen_continuous_features(
        teapot_plug["current"].to_frame(), watch, 3)
    pressuremat_coarse = coarsen_continuous_features(
        pressuremat, watch, 3)
    airbeam_coarse = coarsen_continuous_features(
        airbeam, watch, 3)


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

        # also export means and variances for relevant features
        if write_dists is not None:
            dists = pd.concat((data.mean(), data.std()), axis=1)
            dists.columns = ["mean", "variance"]
            save_path = "../output/{}_{}_distributions.csv"
            dists.to_csv(save_path.format(sensor, write_dists))

    with open("../../temp/sensors.txt", "w") as fh:
        fh.write(str(all_sensors.keys()))

    return all_data, all_sensors.keys()


def flatten_multiindex(index):
    return ["{}_{}".format(x,y) for x,y in index.tolist()]


def process_labels(watch, labels, window_size, exclude_transitions=False):
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
    if exclude_transitions:
        to_keep = labels_coarse["elapsed"] > pd.Timedelta("30 seconds")
    else:
        to_keep = labels_coarse.index

    labels_coarse = labels_coarse.loc[to_keep,["timestamp","label"]].set_index(
        "timestamp")

    return labels_coarse


def process_location_data(watch, location, window_size):
    location = location.replace(0, -99999)
    location["kitchen"] = location.loc[:,["kitchen2_crk","kitchen1_crk"]].max(axis="columns")
    location["living_room"] = location.loc[:,["living_room1_crk","living_room2_crk"]].max(axis="columns")
    location = location.loc[:,["kitchen", "living_room", "dining_room_crk"]]
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

    location_coarse["in_kitchen"] = location_coarse["closest"] == 0
    location_coarse["in_living_room"] = location_coarse["closest"] == 1
    location_coarse["in_dining_room"] = location_coarse["closest"] == 2

    varnames = ["in_kitchen","in_living_room","in_dining_room"]
    return location_coarse.loc[:,varnames]


def process_watch(watch, window_size, use_wavelet_transform=False):
    print use_wavelet_transform
    if use_wavelet_transform:
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
    dwtX = pywt.dwt(watch["{}_X".format(stub)], "haar")
    dwtY = pywt.dwt(watch["{}_Y".format(stub)], "haar")
    dwtZ = pywt.dwt(watch["{}_Z".format(stub)], "haar")

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


def preprocess_metasense(data):
    # Rescale everything to deviations from mean over the first 30 readings
    chunk = data.head(30)

    varnames = ["CO2", "temperature", "pressure", "humidity"]
    for var in varnames:
        data[var] = data[var] - chunk[var].mean()
    return data


def coarsen_continuous_features(data, watch, window_size, fill_method="ffill"):
    data_grouped = data.groupby(level=0).mean().sort_index()
    data_coarsened = data_grouped.rolling(
        window_size).agg(CONTINUOUS_FEATURE_EXTRACTORS)
    data_coarsened.columns = flatten_multiindex(data_coarsened.columns)

    obs_before = watch.shape[0]
    both = watch.loc[:,"step"].to_frame().join(
            data_coarsened, how="left"
        ).drop("step", axis="columns").fillna(method=fill_method).dropna()
    return both


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


def normalize_continuous_cols(data, mu=None, sigma=None):
    mu_new = []
    sigma_new = []
    for ix, col in enumerate(data.columns):
        if mu is None:
            m = data[col].mean()
        else:
            m = mu[ix]
        if sigma is None:
            s = data[col].std()
        else:
            s = sigma[ix]
        mu_new.append(m)
        sigma_new.append(s)
        if col == "label" or data[col].dtype != np.float64:
            continue

        data[col] = (data[col] - m) / s 
    return mu_new, sigma_new

if __name__=="__main__":
    main()
