# %%
import numpy as np
from scipy.signal import csd
import matplotlib.pyplot as plt
import seaborn as sns
from hawk_tools import get_hawk_data
from pyma.oma import ssi

sns.set_theme("notebook")
sns.set_style("ticks")
sns.set_palette("Set2")

# %% intro

"""
Intro to the systyem ID part

LMA techniques

Frequency domain method: RFP
Time domain method: SSI
"""

# %% RFP

"""
RFP intro and maths
"""

# %% Load some data

# You should experiment with these:
sensor = "LTC-05"

# single test, single repeat, single sensor
data = get_hawk_data("LMS", "BR_AR", 1, 1)["BR_AR_1_1"][sensor]
# Try changing the sensor

H = data["Frequency Response Function"]["Y_data"]["value"]
w = data["Frequency Response Function"]["X_data"]["value"]

H_units = data["Frequency Response Function"]["Y_data"]["units"]

# plot the FRF
fig, axs = plt.subplots(2, 1, figsize=(8, 5))
axs[0].semilogy(w, np.abs(H))
axs[1].plot(w, np.angle(H))
axs[0].set_xlabel(r"$\omega$ (Hz)")
axs[1].set_xlabel(r"$\omega$ (Hz)")
axs[0].set_ylabel(r"$|H(\omega)|$ ({})".format(H_units))
axs[1].set_ylabel(r"$\angle H(\omega)$ ({})".format(H_units))
axs[0].set_xlim([0, 160])
axs[1].set_xlim([0, 160])
plt.tight_layout()
plt.show()

# %% Implement RFP


# Hermitian transpose
def HT(a):
    return a.conj().T


# Rational fraction polynomial model
def RFP(H, w, n_modes, oob_terms=0):
    # Specify the orders of our approximation
    m = (
        n_modes * 2 + 1 + oob_terms
    )  # number of coefficients in the numerator polynomial
    n = (
        n_modes * 2 + 1
    )  # number of coefficients in the denominator -1 because we fix b_n=1

    # complex frequency vector
    iw = 1j * w

    # Build monomial basis matricies
    Phi_a = iw[:, None] ** np.arange(m)
    Phi_b_all = iw[:, None] ** np.arange(n)
    Phi_b = Phi_b_all[:, :-1]  # ignore last column because bn=1

    # preallocate some calculations for speed
    Phi_bH = Phi_b * H[:, None]
    Hiwn = H * Phi_b_all[:, -1]
    D = -HT(Phi_a) @ (Phi_bH)

    # form the block matricies
    M = np.block([[HT(Phi_a) @ Phi_a, D], [D.T, HT(Phi_bH) @ (Phi_bH)]])
    x = np.block([HT(Phi_a) @ Hiwn, -HT(Phi_bH) @ Hiwn])

    # Solve and extract the coefficients of the polynomials
    AB = np.linalg.solve(np.real(M), np.real(x))
    a = AB[:m, None]
    b = np.append(AB[m:], 1)[:, None]

    # Generate the predicted FRF
    H_pred = (Phi_a @ a) / (Phi_b_all @ b)

    # Pull out the modal porperties
    roots_b = sorted(np.roots(np.flip(b[:, 0])))[
        ::-2
    ]  # remove every other becaus they are conj pairs
    wns = np.abs(roots_b)
    zetas = -np.real(roots_b) / wns
    return H_pred, wns, zetas


# %% fit the polynomials and plot

# you should experiment with these:
modes = 10
oob = 0
w_low = 5
w_high = 160

# fit the model
idx = np.logical_and(w > w_low, w < w_high)
H_pred, wns_pred, zetas_pred = RFP(H[idx], w[idx], n_modes=modes, oob_terms=oob)

# plot the FRF
fig, axs = plt.subplots(2, 1, figsize=(8, 5))
axs[0].semilogy(w, np.abs(H), label="True")
axs[0].semilogy(w[idx], np.abs(H_pred), label="Predicted")
axs[1].plot(w, np.angle(H), label="True")
axs[1].plot(w[idx], np.angle(H_pred), label="Predicted")
[axs[0].axvline(x, c="k", ls="--") for x in wns_pred]
axs[0].set_xlabel(r"$\omega$ (Hz)")
axs[1].set_xlabel(r"$\omega$ (Hz)")
axs[0].set_ylabel(r"$|H(\omega)|$ ({})".format(H_units))
axs[1].set_ylabel(r"$\angle H(\omega)$ ({})".format(H_units))
axs[0].set_xlim([0, 160])
axs[1].set_xlim([0, 160])
axs[0].legend(
    bbox_to_anchor=(0, 1.2, 1, 0),
    loc="upper left",
    ncols=2,
    mode="expand",
    borderaxespad=0,
    frameon=False,
)
plt.tight_layout()
plt.show()

# %%
"""
RFP requires a lot of fiddling with the parameters

nice option is to only apply RFP in discrete frequency intervals
"""

# %% RFP on discrete ranges

# You should experiment with these:
oob = 6
ranges = (  # (w_low, w_high), n_modes_in_range
    ((5, 9), 1),
    ((12, 14), 1),
    ((15, 19), 1),
    ((22, 24), 2),
    ((26, 30), 1),
    ((30.5, 31), 1),
    ((35, 37), 1),
    ((40.5, 44), 2),
    ((48.5, 54), 2),
    ((86, 90), 1),
    ((92, 100), 1),
    ((112, 118.5), 1),
    ((119.5, 122), 1),
    ((122, 125), 1),
    ((135, 138), 1),
    ((154, 162), 1),
)
# Challenge: see if you can ID the mode (?) at 70Hz


def get_wns(ranges, H, w, oob=0):
    zts = []
    wns = []
    H_pred = np.zeros_like(H) * np.nan
    for (low, high), n in ranges:
        idx = np.logical_and(w > low, w < high)
        Hp, wn, zs = RFP(H[idx], w[idx], n, oob_terms=oob)
        H_pred[idx] = np.squeeze(Hp)
        wns.extend(wn)
        zts.extend(zs)
    return H_pred, np.array(wns), np.array(zts)


# fit the model
H_pred, wns_pred, zetas_pred = get_wns(ranges, H, w, oob=oob)

# plot the FRF
fig, axs = plt.subplots(2, 1, figsize=(8, 5))
axs[0].semilogy(w, np.abs(H), label="True")
axs[0].semilogy(w, np.abs(H_pred), label="Predicted")
axs[1].plot(w, np.angle(H), label="True")
axs[1].plot(w, np.angle(H_pred), label="Predicted")
[axs[0].axvline(x, c="k", ls="--") for x in wns_pred]
axs[0].set_xlabel(r"$\omega$ (Hz)")
axs[1].set_xlabel(r"$\omega$ (Hz)")
axs[0].set_ylabel(r"$|H(\omega)|$ ({})".format(H_units))
axs[1].set_ylabel(r"$\angle H(\omega)$ ({})".format(H_units))
axs[0].set_xlim([0, 160])
axs[1].set_xlim([0, 160])
axs[0].legend(
    bbox_to_anchor=(0, 1.2, 1, 0),
    loc="upper left",
    ncols=2,
    mode="expand",
    borderaxespad=0,
    frameon=False,
)
plt.tight_layout()
plt.show()

# note that without computing the modeshapes we cannot reconstruct the entire FRF.

# %%
"""
Wrapping up on RFP method

    getting the modeshapes

    orthogonal polynomial bases

    polymax
    mlmm 
    etc.
"""

# %% SSI

"""
SSI maths goes here

pyMA shoutout
"""

# %% Load some data


# single test, single repeat, single sensor
data = get_hawk_data("NI", "RPH_AR", 1, 1)["RPH_AR_1_1"]

y = []
for key, sensor in data.items():
    if key[:3] in {"EXH", "FRC", "TRI", "Met"}:
        continue  # skip some sensor channels
    y.append(sensor["Measurement"]["value"])
y = np.array(y)
y_units = data["LTC-01"]["Measurement"]["units"]
dt = 1 / int(data["Meta"]["Acquisition Sample Rate"])

# plot the time series
plt.figure(figsize=(8, 5))
plt.plot(y.T)
plt.ylabel(f"Acceleration ({y_units})")
plt.show()
# note the 10 repeats in the series


# %% compte singular valued spectrum

# you should experiment with these:
nfft = 10000
sensor_idx = slice(None) # will be slow for all sensors

# Compute CPSD to (N/2, P, P) tensor
yc = y[sensor_idx, :]
cpsd = np.zeros(
    (int(nfft / 2) + 1, yc.shape[0], yc.shape[0]), dtype=complex
)
# This can take ~2 minutes to compute for all sensors
for i, sig1 in enumerate(yc):
    for j, sig2 in enumerate(yc):
        f, cpsd[:, i, j] = csd(sig1, sig2, fs=1 / dt, nperseg=nfft, noverlap=None)

_, SVS, _ = np.linalg.svd(cpsd)

#%% Plot singuilar valued spectrum

plt.figure(figsize=(8,5))
plt.semilogy(f, SVS[:,:5])
plt.ylabel('$|H|$')
plt.xlabel('$\omega$')
plt.xlim([0, 160])
plt.tight_layout()
plt.show()


# %% use pyMA to perform SSI

# you should experiment with these:
sensor_idx = slice(None) # will be slow for all sensors
decimate_factor = 10 # speed up by increasing this
max_order = 60
compute_orders = -1 # -1 => compute all orders

y_dc = y[sensor_idx, ::decimate_factor]
dt_decimated = dt * decimate_factor
opts = {
    "max_model_order": max_order,
    "model_order": compute_orders,
    "dt": dt_decimated,
}
alg = ssi.SSI(opts)
props = alg(y_dc)

# %% plot stabilisation diagram

plt.figure()
for i, order in enumerate(props):
    wns = order[0] / (2 * np.pi)
    wns = wns[order[1] > 0]
    plt.scatter(wns, [i] * len(wns), s=1, marker="x", c="k")
plt.gca().twinx().semilogy(f, SVS[:, :5], label='SVS')
plt.xlim([0, 160])
plt.show()


# %%

"""
SSI epilogue
"""


# %%

"""
Next steps / exercises

    - load the data on your machine
    - explore and make plots
    - 
"""
