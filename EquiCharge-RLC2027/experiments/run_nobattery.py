"""Run an audit module on the reference station WITHOUT its on-site battery.

    python -m experiments.run_nobattery audit_capacity

Runs experiments.<module>.main() with scarcity_station replaced by the battery-disabled
station from nobattery_station.py, writing into a temporary folder and then
saving each output as <name>_nobattery.json next to the released results. The released
battery-run files are never opened for writing.
"""
import glob
import importlib
import json
import os
import sys

import experiments.run_experiments as RE
from experiments.nobattery_station import scarcity_station_no_battery

RE.scarcity_station = scarcity_station_no_battery

name = sys.argv[1]
mod = importlib.import_module("experiments." + name)
final_dir = mod.RESULTS
tmp = os.path.join(final_dir, "_nobattery_tmp_" + name)
os.makedirs(tmp, exist_ok=True)
mod.RESULTS = tmp
mod.main()
for f in glob.glob(os.path.join(tmp, "*.json")):
    out = json.load(open(f))
    out["station"] = "reference site WITHOUT on-site battery"
    dst = os.path.join(final_dir, os.path.basename(f).replace(".json", "_nobattery.json"))
    json.dump(out, open(dst, "w"), indent=2)
    os.remove(f)
    print("wrote", dst)
os.rmdir(tmp)
