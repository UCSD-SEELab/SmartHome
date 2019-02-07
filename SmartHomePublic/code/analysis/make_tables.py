import os
import re
import json
import cStringIO
import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def main():
    files = filter(lambda x: "stats_subject" in x, os.listdir("../../output"))
    subjects = np.unique([x.split("_")[1] for x in files])
    hidden_units = np.unique([x.split("_")[2].replace(".json","") for x in files])
    
    plot_accuracy_hidden_units(subjects, hidden_units)
    make_confusion_matrices(subjects, hidden_units)


def plot_accuracy_hidden_units(subjects, hidden_units):
    results = {s: {"h": [], "a": []} for s in subjects}
    for s in subjects:
        for h in hidden_units:
            results[s]["h"].append(int(h))
            file = "../../output/stats_{}_{}.json".format(s, h)
            with open(file) as fh:
                results[s]["a"].append(get_accuracy(json.load(fh)))

    f = lambda x: pd.DataFrame(x).set_index("h").sort_index()
    out_frames = [(s,f(x)) for s, x in results.items()]
    data = pd.concat([x[1] for x in out_frames], axis=1)

    f = lambda x: "Subject {}".format(re.findall("\d", x)[0])
    data.columns = [f(x[0]) for x in out_frames]
    data = pd.melt(data.reset_index(),
                   id_vars="h", 
                   var_name="subject", 
                   value_name="accuracy")
    data = data.rename(columns={"h": "Hidden Units"})
    fct = sns.factorplot(x="subject", y="accuracy", 
        hue="Hidden Units", data=data, palette="dark", kind="bar")
    ax = fct.axes[0][0]
    ax.set_ylabel("Classification Accuracy")
    ax.set_xlabel("")
    ax.set_yticks(np.arange(0.0, 0.9, 0.1))
    path = "../../output/mlp_accuracy_hidden_comparison.png"
    plt.savefig(path, bbox_inches="tight")


def get_accuracy(data):
    return data["test_accuracy"][np.argmax(data["validation_accuracy"])]


def make_confusion_matrices(subjects, hidden_units):
    cfn_mats = []
    for s in subjects:
        for h in hidden_units:
            file = "../../output/stats_{}_{}.json".format(s, h)
            with open(file) as fh:
                data = json.load(fh)
            cfn = pd.DataFrame(data["cfn_matrix"])
            cfn.columns = data["cfn_matrix_labels"]
            cfn.index = data["cfn_matrix_labels"]
            cfn_mats.append((get_accuracy(data), cfn, (s, h)))
    
    cfn_mats = sorted(cfn_mats, key=lambda x: -x[0])

    with open("../../output/confusion_matrices.txt", "w") as fh:
        for acc, mat, params in cfn_mats:
            S = cStringIO.StringIO()
            mat.to_csv(S, sep="\t")
            header = "Subject: {}\nAccuracy: {}\nHidden Units: {}\n"
            fh.write(header.format(params[0], acc, params[1]))
            fh.write("\n")
            fh.write(S.getvalue())
            fh.write("\n\n")
            S.close()


if __name__=="__main__":
    main()
