import os
import time

import numpy as np
import pandas as pd

from pythermalcomfort.clothing import clo_dynamic_ashrae
from pythermalcomfort.environment import v_relative
from pythermalcomfort.models import pmv_ppd_ashrae, pmv_ppd_iso
from pythermalcomfort.utilities import clo_individual_garments, met_typical_tasks

# input variables
tdb = 27  # dry bulb air temperature, [$^{\circ}$C]
tr = 25  # mean radiant temperature, [$^{\circ}$C]
v = 0.3  # average air speed, [m/s]
rh = 50  # relative humidity, [%]
activity = "Typing"  # participant's activity description
garments = ["Sweatpants", "T-shirt", "Shoes or sandals"]

met = met_typical_tasks[activity]  # activity met, [met]
icl = sum(
    [clo_individual_garments[item] for item in garments],
)  # calculate total clothing insulation

# calculate the relative air velocity
vr = v_relative(v=v, met=met)
# calculate the dynamic clothing insulation
clo = clo_dynamic_ashrae(clo=icl, met=met)

# calculate PMV in accordance with the ASHRAE 55 2020
results = pmv_ppd_iso(tdb=tdb, tr=tr, vr=vr, rh=rh, met=met, clo=clo)

# print the results
print(results)

# print PMV value
print(f"pmv={results['pmv']}, ppd={results['ppd']}%")

# for users who want to use the IP system
results_ip = pmv_ppd_iso(tdb=77, tr=77, vr=0.6, rh=50, met=1.1, clo=0.5, units="IP")
print(results_ip)

# If you want you can also pass pandas series or arrays as inputs
df = pd.read_csv(os.getcwd() + "/examples/template-SI.csv")

v_rel = v_relative(df["v"], df["met"])
clo_d = clo_dynamic_ashrae(df["clo"], df["met"])
results = pmv_ppd_ashrae(df["tdb"], df["tr"], v_rel, df["rh"], df["met"], clo_d, 0)
print(results)

df["vr"] = v_rel
df["clo_d"] = clo_d
df["pmv"] = results.pmv  # you can also use results["pmv"]
df["ppd"] = results["ppd"]

print(df.head())

# uncomment the following line if you want to save the data to .csv file
# df.to_csv('results.csv')

# This method is extremely fast and can perform a lot of calculations in very little time
iterations = 10000
tdb = np.empty(iterations)
tdb.fill(25)
tdb = tdb.tolist()
met = np.empty(iterations)
met.fill(1.5)
met = met.tolist()

v_rel = v_relative(0.1, met)
clo_d = clo_dynamic_ashrae(1, met)

# ASHRAE PMV
start = time.time()
pmv_ppd_ashrae(
    tdb=tdb,
    tr=23,
    vr=v_rel,
    rh=40,
    met=1.2,
    clo=clo_d,
)
end = time.time()
print(end - start)

# ISO PMV
start = time.time()
pmv_ppd_iso(tdb=tdb, tr=23, vr=v_rel, rh=40, met=1.2, clo=clo_d)
end = time.time()
print(end - start)
