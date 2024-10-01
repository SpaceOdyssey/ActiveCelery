from pylab import *
from pathlib import Path
import os
import uuid

# This is the script that was used to generate example_data.txt

root = Path(os.environ.get("ACTIGRAPHY_PATH"))
run_id = str(uuid.uuid4())
run_path = root.joinpath("Simulated", run_id)
run_path.mkdir()

# Inputs
noise = 1.0
n_days = 14
n_obs = 1000
freqs = [1.0, 3.0]  # cycles per day  # linspace(0.05, 0.15, 11)
periods = [1/f for f in freqs]
amps = [1.0, 1.0]
qualities = [25.0, 25.0]
s = randint(0, 2**32 - 1)

# Save inputs.
with open(run_path.joinpath("inputs.csv"), "w") as f:
    f.write("noise,%s\n" % noise)
    f.write("n_days,%s\n" % n_days)
    f.write("n_obs,%s\n" % n_obs)
    f.write("seed,%s\n" % s)

with open(run_path.joinpath("component_inputs.csv"), "w") as f:
    f.write(f"amplitude,{','.join(map(str, amps))}\n")
    f.write(f"period,{','.join(map(str, periods))}\n")
    f.write(f"quality,{','.join(map(str, qualities))}\n")

seed(s)
t = sort(n_days*rand(n_obs))

[t1, t2] = meshgrid(t, t)
dt = t1 - t2

y = zeros(t.size)

# Amplitude, period, quality
for i in range(len(freqs)):
    A, P, Q = amps[i], 1.0/freqs[i], qualities[i]
    w0 = 2*pi/P
    tau = abs(dt)
    eta = sqrt(1.0 - 1.0/(4.0*Q**2))
    C = A**2*exp(-w0*tau/(2*Q))*(cos(eta*w0*tau) + sin(eta*w0*tau)/(2.0*eta*Q))

    n = matrix(randn(len(t))).T
    L = cholesky(C)

    yy = (L*n).T
    y += np.array(yy).flatten()

data = empty((len(t), 2))
data[:,0], data[:,1] = t, y
yerr = noise*randn(data.shape[0])
abs_yerr = abs(yerr)
data[:,1] += yerr  # Adding noise.
savetxt(run_path.joinpath('example_data.txt'), data)

figure(figsize=(12,6))
errorbar(data[:,0], data[:,1], yerr=noise, marker=".", markersize=5)
xlabel("Days")
ylabel("Log Activity")
plt.savefig(run_path.joinpath("log_example_data.pdf"))

figure(figsize=(12,6))
plot(data[:,0], exp(data[:,1]))
xlabel("Days")
ylabel("Activity")
plt.savefig(run_path.joinpath("example_data.pdf"))

def periodogram(fs):
    """
    Compute the periodogram.
    """
    pgram = empty(len(fs))
    for i in range(len(fs)):
        sine   = sin(2*pi*fs[i]*t)
        cosine = cos(2*pi*fs[i]*t)
        pgram[i] = sum(sine*y/noise)**2 + sum(cosine*y/noise)**2
    return pgram/len(fs)

fs = linspace(0.001, 24.0, 10001)
pgram = periodogram(fs)
plt.figure()
plot(fs, pgram)
vlines(freqs, ymin = 0.0, ymax = 1.1*max(pgram),
       linestyles = 'dashed',
       colors = 'k')
xlabel("Cycles / Day")
ylabel("Power")
title("Periodogram")
plt.savefig(run_path.joinpath("periodogram.pdf"))


# Log likelihood
y = matrix(data[:,1]).T
for i in range(0, len(t)):
  C[i, i] += 0.9**2
L = cholesky(C)
logl = -0.5*len(t)*log(2*pi) - 0.5*2*sum(log(diag(L))) - 0.5*y.T*inv(C)*y
print(logl)
