import sys
import tables
sys.path.append('../')

from tabulate import tabulate
from utils.utils import *
from utils.preliminaries import *

CONTINUOUS_FEATURE_EXTRACTORS = [np.min, np.max, np.mean, np.var]

def main():
    anthony_data = build_data("../../temp/anthony_data.h5", 30, "anthony")
    yunhui_data = build_data("../../temp/yunhui_data.h5", 300, "yunhui")

    print "===============> BEFORE NORMALIZING <================="

    print "++++++++++++++++ ANTHONY ++++++++++++++++"
    print anthony_data.mean()
    print anthony_data.var()

    print "++++++++++++++++ YUNHUI +++++++++++++++++"
    print yunhui_data.mean()
    print yunhui_data.var()

    normalize_continuous_cols(anthony_data)
    normalize_continuous_cols(yunhui_data)

    print "===============> AFTER NORMALIZING <================="

    print "++++++++++++++++ ANTHONY ++++++++++++++++"
    print anthony_data.mean()
    print anthony_data.var()

    print "++++++++++++++++ YUNHUI +++++++++++++++++"
    print yunhui_data.mean()
    print yunhui_data.var()

    anthony_data.describe().to_csv("../../temp/anthony_stats.csv")
    yunhui_data.describe().to_csv("../../temp/yunhui_stats.csv")


def build_data(path, window_size, subject):
    watch = pd.read_hdf(path, "watch")
    labels = pd.read_hdf(path, "labels")
    tv_plug = pd.read_hdf(path, "tv_plug")
    teapot_plug = pd.read_hdf(path, "teapot_plug")
    pressuremat = pd.read_hdf(path, "pressuremat")
    metasense = pd.read_hdf(path, "metasense")
    location = pd.read_hdf(path, "location")

    cabinet1_contact = pd.read_hdf(path, "cabinet1_contact")
    cabinet2_contact = pd.read_hdf(path, "cabinet2_contact")
    drawer1_contact = pd.read_hdf(path, "drawer1_contact")
    drawer2_contact = pd.read_hdf(path, "drawer2_contact")
    fridge_contact = pd.read_hdf(path, "fridge_contact")

    print path
    print labels

    #dining_room_motion = pd.read_hdf(
    #    path, "dining_room_motion").set_index("timestamp")
    # living_room_motion = pd.read_hdf(
    #     path, "living_room_motion").set_index("timestamp")

    watch_coarse = process_watch(watch, window_size)
    labels_coarse = process_labels(watch, labels, window_size)
    location_coarse = process_location_data(watch, location, window_size)
    metasense_coarse = coarsen_continuous_features(metasense, watch, window_size)
    tv_plug_coarse = coarsen_continuous_features(
        tv_plug["current"].to_frame(), watch, window_size)
    teapot_plug_coarse = coarsen_continuous_features(
        teapot_plug["current"].to_frame(), watch, window_size)
    pressuremat_coarse = coarsen_continuous_features(
        pressuremat, watch, window_size)

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

    all_data = labels_coarse.join(
            watch_coarse
        ).join(
            location_coarse
        ).join(
            metasense_coarse
        ).join(
            tv_plug_coarse, rsuffix="_tv_plug"
        ).join(
            teapot_plug_coarse, rsuffix="_teapot_plug"
        ).join(
            pressuremat_coarse
        ).join(
            cabinet1_coarse
        ).join(
            cabinet2_coarse
        ).join(
            drawer1_coarse
        ).join(
            drawer2_coarse
        ).join(
            fridge_coarse
        )
    return all_data


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


def process_watch(watch, window_size):
    watch_coarsened = watch.rolling(
        window_size).agg(CONTINUOUS_FEATURE_EXTRACTORS).dropna()

    # flatten the annoying multi-index Pandas returns
    watch_coarsened.columns = flatten_multiindex(watch_coarsened.columns)
    return watch_coarsened


def flatten_multiindex(index):
    return ["{}_{}".format(x,y) for x,y in index.tolist()]


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
    main()
