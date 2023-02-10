# %%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from hawk_tools import get_hawk_data

sns.set_theme("notebook")
sns.set_style("ticks")
sns.set_palette("Set2")

# %% intro

'''
Intro to the SHM part

What is an SHM?

Eploring the data: PCA

Simple damage detection: Outlier analysis
'''

# %% PCA

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


ranges = (
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


def get_wns(ws, frf):
    wns = []
    for (low, high), n in ranges:
        idx = np.logical_and(ws > low, ws < high)
        _, wn, _ = RFP(frf[idx], ws[idx], n, oob_terms=6)
        wns.extend(wn)
    return np.array(wns)


BR_AMP_levels = [0.4, 0.8, 1.2, 1.6, 2]
DS_AMP_levels = [0.4, 1.2, 2, 0.4, 1.2, 2, 0.4, 1.2, 2]
DS_DMG_levels = [1, 1, 1, 2, 2, 2, 3, 3, 3]
all_series = [("BR_AR", np.arange(1, 6)), ("DS_CTE", np.arange(19, 28))]

load_opts = {
    "data": "Frequency Response Function",
    "meta": False,
    "attrs": False,
    "compress_x_axis": True,
}
data_dir = r"C:\Users\me1mcz\Downloads\hawk_data"
out = {}
# loop through test series
for series, runs in all_series:
    # loop through test runs
    for run in runs:
        run_data = get_hawk_data(
            "LMS", series, run, download_dir=data_dir, ask=0, load_kwargs=load_opts
        )
        # loop through test repeats
        for rep, rep_data in run_data.items():
            ws = rep_data["X_data"]["Frequency Response Function"]["value"]
            # loop through sensors
            wns = []
            for sensor, sensor_data in rep_data.items():
                # exclude some sensors and metadata
                if sensor[:3] in {"TRI", "EXH", "FRC", "Met", "X_d"}:
                    continue
                # exclude sensors closest to wing root (SNR too low)
                if int(sensor[-1]) < 2:
                    continue
                # compute wns
                frf = sensor_data["Frequency Response Function"]["Y_data"]["value"]
                wns.append(get_wns(ws, frf))

            # Store some metadata for visualisation
            dat = rep.split("_")
            num = int(dat[2]) - 1
            if dat[0] == "DS":
                num-=19#HACK
                dmg = DS_DMG_levels[num]
                amp = DS_AMP_levels[num]
            elif dat[0] == "BR":
                dmg = 0
                amp = BR_AMP_levels[num]
            av_wns = np.mean(wns, axis=0)
            out |= {rep: [av_wns, {"dmg": dmg, "amp": amp}]}
print("Total tests analysed: ", len(out))
print("Sensors analysed per test: ", len(wns))
wns, labs = zip(*(out.values()))
wns = np.array(wns)

# %% all the identified modes

U, s, V = np.linalg.svd(wns, full_matrices=0)
PCS = U @ np.diag(s)

ll = ["Baseline", "$M_a=256$", "$M_a=620$", "$M_a=898$"]
plt.figure(figsize=(8, 5))
for d in [0, 1, 2, 3]:
    ix = [l["dmg"] == d for l in labs]
    ss = [20*v["amp"] for i, v in zip(ix, labs) if i]
    plt.scatter(PCS[ix, 0], PCS[ix, 1], color=f'C{d}', label=ll[d], s=ss)
plt.legend(
    bbox_to_anchor=(0, 1.1, 1, 0),
    loc="upper left",
    ncols=4,
    mode="expand",
    borderaxespad=0,
    frameon=False,
)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.tight_layout()


# %% explore the outlier

plt.figure(figsize=(8, 5))
sns.boxplot(wns)
plt.xlabel('Identified peak index')
plt.ylabel(r'$\omega_n$')
plt.tight_layout()


# %% cropping out mode in range (92-100)

# remove the 10th mode from the PCA
wns_crop = np.delete(wns, [10], axis=1)

U, s, V = np.linalg.svd(wns_crop, full_matrices=0)
PCS = U @ np.diag(s)

ll = ["Baseline", "$M_a=256$", "$M_a=620$", "$M_a=898$"]
plt.figure(figsize=(8, 5))
for d in [0, 1, 2, 3]:
    ix = [l["dmg"] == d for l in labs]
    ss = [20*v["amp"] for i, v in zip(ix, labs) if i]
    plt.scatter(PCS[ix, 0], PCS[ix, 1], color=f'C{d}', label=ll[d], s=ss)
plt.legend(
    bbox_to_anchor=(0, 1.1, 1, 0),
    loc="upper left",
    ncols=4,
    mode="expand",
    borderaxespad=0,
    frameon=False,
)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.tight_layout()




# %% Outlier analysis (Matty tbc)