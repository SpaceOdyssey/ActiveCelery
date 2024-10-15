# from pylab import *
import random
from matplotlib import pyplot as plt
import itertools
import numpy as np
from pathlib import Path
import os
import uuid
from scipy.signal import welch
from scipy.linalg import cholesky, inv

# This script creates simulated data.

# Functions.
def periodogram(t, y, fs, noise):
    """
    Compute the periodogram.
    """
    pgram = np.empty(len(fs))
    for i in range(len(fs)):
        sine   = np.sin(2*np.pi*fs[i]*t)
        cosine = np.cos(2*np.pi*fs[i]*t)
        pgram[i] = sum(sine*y/noise)**2 + sum(cosine*y/noise)**2
    return pgram/len(fs)

# Data.
root = Path(os.environ.get("ACTIGRAPHY_PATH"))
print()

# Constant Inputs
n_days = 14
n_obs = 1000
freqs = [1.0, 3.0]  # cycles per day  # linspace(0.05, 0.15, 11)
periods = [1/f for f in freqs]
amps = [1.0, 1.0]
qualities = [25.0, 25.0]

# Verying Inputs.
noise_var_opt = np.logspace(-1, 1, 5)
s_opt = np.random.randint(0, 2**32 - 1, size = 5)

for noise_var, s in itertools.product(noise_var_opt, s_opt):
    run_id = str(uuid.uuid4())
    run_path = root.joinpath("Simulated", run_id)
    run_path.mkdir()

    # Save inputs.
    with open(run_path.joinpath("inputs.csv"), "w") as f:
        f.write("noise_var,%s\n" % noise_var)
        f.write("n_days,%s\n" % n_days)
        f.write("n_obs,%s\n" % n_obs)
        f.write("seed,%s\n" % s)

    with open(run_path.joinpath("component_inputs.csv"), "w") as f:
        f.write(f"amplitude,{','.join(map(str, amps))}\n")
        f.write(f"period,{','.join(map(str, periods))}\n")
        f.write(f"quality,{','.join(map(str, qualities))}\n")

    random.seed(s)
    t = np.sort(n_days*np.random.rand(n_obs))

    [t1, t2] = np.meshgrid(t, t)
    dt = t1 - t2

    y = np.zeros(t.size)

    # Amplitude, period, quality
    for i in range(len(freqs)):
        A, P, Q = amps[i], 1.0/freqs[i], qualities[i]
        w0 = 2*np.pi/P
        tau = abs(dt)
        eta = np.sqrt(1.0 - 1.0/(4.0*Q**2))
        C = A**2*np.exp(-w0*tau/(2*Q))*(np.cos(eta*w0*tau) + np.sin(eta*w0*tau)/(2.0*eta*Q))

        n = np.matrix(np.random.randn(len(t))).T
        L = cholesky(C)

        yy = (L*n).T
        y += np.array(yy).flatten()


    noise = np.sqrt(noise_var)
    data = np.empty((len(t), 2))
    data[:,0], data[:,1] = t, y
    yerr = np.sqrt(noise)*np.random.randn(data.shape[0])
    abs_yerr = abs(yerr)
    data[:,1] += yerr  # Adding noise.
    np.savetxt(run_path.joinpath('data.txt'), data)

    plt.figure(figsize=(12,6))
    plt.errorbar(data[:,0], data[:,1], yerr=noise, marker=".", markersize=5)
    plt.xlabel("Days")
    plt.ylabel("Log Activity")
    plt.savefig(run_path.joinpath("log_data.pdf"))

    plt.figure(figsize=(12,6))
    plt.plot(data[:,0], np.exp(data[:,1]))
    plt.xlabel("Days")
    plt.ylabel("Activity")
    plt.savefig(run_path.joinpath("data.pdf"))

    # Theoretical periodogram.
    fs = np.linspace(0.001, 24.0, 10001)
    pgram = periodogram(t, y, fs, noise)
    plt.figure()
    plt.plot(fs, pgram)
    plt.vlines(freqs, ymin = 0.0, ymax = 1.1*max(pgram),
        linestyles = 'dashed',
        colors = 'k')
    plt.xlabel("Cycles / Day")
    plt.ylabel("Power")
    plt.title("Periodogram")
    plt.savefig(run_path.joinpath("theoretical_periodogram.pdf"))

    # Inferred periodogram
    f, Pxx = welch(y, fs = 1.0/24.0, nperseg = 256)
    plt.plot(f, Pxx)
    plt.xlabel('Frequency')
    plt.ylabel('Power spectral density')
    plt.savefig(run_path.joinpath("empirical_periodogram.pdf"))

    # Log likelihood
    y = np.matrix(data[:,1]).T
    for i in range(0, len(t)):
        C[i, i] += 0.9**2
    L = cholesky(C)
    logl = -0.5*len(t)*np.log(2*np.pi) - 0.5*2*sum(np.log(np.diag(L))) - 0.5*y.T*inv(C)*y
    print(logl)
