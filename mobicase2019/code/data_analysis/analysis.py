import sys
import tables
sys.path.append('../')

from tabulate import tabulate
from utils.utils import *
from utils.preliminaries import *


def main():
    pass


def build_anthony_data():
    path = "../../temp/anthony_data.h5"
    
    watch = pd.read_hdf(path, "watch")
    tv_plug = pd.read_hdf(path, "tv_plug")
    teapot_plug = pd.read_hdf(path, "teapot_plug")
    pressuremat = pd.read_hdf(path, "pressuremat")
    metasense = pd.read_hdf(path, "metasense")
    # airbeam = pd.read_hdf(path, "airbeam")

    cabinet1_contact = pd.read_hdf(path, "cabinet1_contact")
    cabinet2_contact = pd.read_hdf(path, "cabinet2_contact")
    drawer1_contact = pd.read_hdf(path, "drawer1_contact")
    drawer2_contact = pd.read_hdf(path, "drawer2_contact")
    fridge_contact = pd.read_hdf(path, "fridge_contact")
    dining_room_motion = pd.read_hdf(
        path, "dining_room_motion").set_index("timestamp")
    living_room_motion = pd.read_hdf(
        path, "living_room_motion").set_index("timestamp")


def process_binary_features(contact, watch, varname):
    """
    This method computes the time since the last "active" message received.
    I.e. this computes the time since a door was opened or motion was 
    detected
    """

    obs_before = watch.shape[0]
    both = watch.loc[:,"step"].to_frame().join(
            contact, how="left"
        ).drop("step", axis="columns").fillna(0).sort_index().reset_index()
    assert obs_before - both.shape[0] == 0, "Merge Error"

    # now compute the number of seconds since the last "open" event
    both["last_open"] = both["timestamp"]
    both.loc[both["cabinet1_contact"] == 0, "last_open"] = np.nan
    both["last_open"] = both["last_open"].fillna(method="ffill")
    both[""]





