# %%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme("notebook")
sns.set_style("ticks")
sns.set_palette("Set2")

# %%
"""
Welcome to INI etc.

This week we will be working on the data collected from a large testing campaign on a Hawk aircraft at the LVV.

So far, two test campaigns have been completed {'LMS', 'NI'} the first campaign collects data from an LMS system in the frequency domian. The second collected time domain data from a NI system.

In order to make accessing and working with the Hawk data as straightforward as possible this week, we have provided a utility function (in python (sorry matlab)).

In order to run these notebooks (and to get access to the hawk_tools package) install the dependencies by:

pip install -r INI_hawk_requiremnts.txt 

that is included in the repo.
"""

# %%
from hawk_tools import get_hawk_data

# %%
"""
Get Hawk data has the following call signiture

get_hawk_data(
    test_camp, 
    test_id, 
    test_runs=None, 
    test_reps=None, 
    download_dir="./.hawk_data", 
    disk_only=False, 
    ask=True, 
    load_kwargs={}
)

Of these, test_camp refers to the test campaign {'LMS', 'NI'}, and test_ID refers to the test type in the corresponding spreadsheet (check the repo for these).

Other args are hopefully self explanitory but a breif description is included here.

test_camp: test campaign i.e. 'LMS' or 'NI'
test_id: test type i.e. 'BR_AR' for burst random amplitude ramp
test_runs: test runs to download i.e. 1 or [1,2] or None for all tests
test_reps: test repeats to download i.e. 2 or [1,2,4,8] or None for all reps
download_dir: download directory (caches files to here)
disk_only: bool, do not load into ram, function returns {}
ask: bool, disable user input check for large downloads
load_kwargs: see below. specify only some data ranges to be loaded into ram

Below are a few examples of using the function and interacting with the data.
"""


# %% Load a single rep of a single test

data = get_hawk_data("LMS", "BR_AR", 1, 1)
print(list(data.keys()))  # available datasets
print(list(data["BR_AR_1_1"].keys()))  # available sensors in the data
print(list(data["BR_AR_1_1"]["LTC-03"].keys()))  # fields per sensor

# %% pull out the FRF of a sngle sensor and plot

H = data["BR_AR_1_1"]["LTC-03"]["Frequency Response Function"]["Y_data"]["value"]
w = data["BR_AR_1_1"]["LTC-03"]["Frequency Response Function"]["X_data"]["value"]

H_units = data["BR_AR_1_1"]["LTC-03"]["Frequency Response Function"]["Y_data"]["units"]
w_units = data["BR_AR_1_1"]["LTC-03"]["Frequency Response Function"]["X_data"]["units"]

fig, axs = plt.subplots(2, 1, figsize=(8, 5))
axs[0].semilogy(w, np.abs(H))
axs[1].plot(w, np.angle(H))
axs[0].set_xlabel(r"$\omega$ ({})".format(w_units))
axs[1].set_xlabel(r"$\omega$ ({})".format(w_units))
axs[0].set_ylabel(r"$|H(\omega)|$ ({})".format(H_units))
axs[1].set_ylabel(r"$\angle H(\omega)$ ({})".format(H_units))
axs[0].set_xlim([0, 160])
axs[1].set_xlim([0, 160])
plt.tight_layout()
plt.show()

# %% load more repittions - note that the earlier download is cached
data = get_hawk_data("LMS", "BR_AR", 1, [1, 2, 3])
print(list(data.keys()))

# %% Load some NI data
data = get_hawk_data("NI", "RPH_AR", 1, [1, 2])
print(list(data.keys()))  # available datasets
print(list(data["RPH_AR_1_1"].keys()))  # available sensors in the data
print(list(data["RPH_AR_1_1"]["LTC-01"].keys()))  # fields per sensor
# Note the different fields available between test campaigns

"""
For more information on the sensor locations and the test runs, check out the spreadsheets available in the repo.

# Advanced usage

Optionally, the load_kwargs can be used to oly load certain signals from the data. A few examples are included below.
"""
