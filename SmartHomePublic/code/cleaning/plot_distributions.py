import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = {}
data["subject1"] = pd.read_hdf("data_processed_centered.h5", "subject1") 
data["subject2"] = pd.read_hdf("data_processed_centered.h5", "subject2")
data["subject4"] = pd.read_hdf("data_processed_centered.h5", "subject4")
data["subject5"] = pd.read_hdf("data_processed_centered.h5", "subject5")
data["subject6"] = pd.read_hdf("data_processed_centered.h5", "subject6")

for s, d in data.items():
    d.index = (d.index - d.index.min())
    d = d.sort_index()

# compare distributions of some relevant variables
varnames = data["subject1"].columns
colors = ["darkgreen", "darkblue", "purple", "orange", "cyan"]
for v in varnames:
    print v
    plt.plot()

    x = []
    for ix, qq in enumerate(data.items()):
        s, q = qq
        toplot = np.random.choice(q.index, size=10000)
        gg = q.loc[toplot,v].sort_index()
        plt.plot(gg.index, gg.values,
            color=colors[ix], label=s)
        x.append(gg.values)

    # cov12 = np.corrcoef(x[0], x[1])[0][1]
    # cov13 = np.corrcoef(x[0], x[2])[0][1]
    # cov23 = np.corrcoef(x[1], x[2])[0][1]

    # subt = "CORR(1,2): {} - CORR(1,3): {} - CORR(2,3): {}".format(cov12, cov13, cov23)
    # plt.suptitle(v)
    plt.title(v)
    plt.legend()
    plt.savefig("plots/{}.png".format(v), bbox_inches="tight")
    plt.close()

s1 = pd.read_hdf("subject1_data.h5", "metasense")
s2 = pd.read_hdf("subject2_data.h5", "metasense")
s6 = pd.read_hdf("subject6_data.h5", "metasense")

for v in s1.columns:
    sns.distplot(trim_var(s1[v], "subject 1"), label="Subject 1")
    sns.distplot(trim_var(s2[v], "subject 2"), label="Subject 2")
    sns.distplot(trim_var(s6[v], "subject 3"), label="Subject 6")
    plt.legend()
    plt.show()

def trim_var(v, label):
    lb = v.quantile(0.05)
    ub = v.quantile(0.95)
    print label
    print v.describe()
    print "======================="
    return v[np.logical_and(v > lb, v < ub)]
