import dnest4.classic as dn4
import matplotlib.pyplot as plt
import numpy as np

print("Generating DNest4 plots. Close these to continue.")

# Run postprocess from DNest4
dn4.postprocess()

print("Converting posterior samples to YAML for extra convenience.")
from to_yaml import posterior_sample, indices, to_yaml
to_yaml()

# Extract amplitudes
start = indices["amplitude[0]"]
end   = indices["period[0]"]
all_amplitudes = posterior_sample[:, start:end].flatten()
all_amplitudes = all_amplitudes[all_amplitudes != 0.0]

# Periods
start = indices["period[0]"]
end   = indices["quality[0]"]
all_periods = posterior_sample[:, start:end].flatten()
all_periods = all_periods[all_periods != 0.0]

# Extract quality factors
start = indices["quality[0]"]
all_qualities = posterior_sample[:, start:-1].flatten()
all_qualities = all_qualities[all_qualities != 0.0]

# Histogram of inferred log-periods
plt.figure()
plt.hist(np.log10(all_periods), 1000, alpha=0.3)
plt.xlabel(r"$\log_{10}$(period)")
plt.ylabel("Relative probability")
plt.savefig("relative_probability.pdf")

# Histogram of inferred periods, weighted by amplitude
plt.figure()
plt.hist(np.log10(all_periods), bins=1000,
         weights=all_amplitudes, alpha=0.3)
plt.xlabel(r"$\log_{10}$(period/day)")
plt.ylabel("Relative expected amplitude")
plt.savefig("relative_expected_amplitude.pdf")

# Histogram of inferred periods, weighted by amplitude. In linear space.
plt.figure()
plt.hist(all_periods, bins=1000,
         weights=all_amplitudes, range = (0.0, 3.0), alpha=0.3)
plt.xlim((0.0, 3.0))
plt.xlabel(r"period (days)")
plt.ylabel("Relative expected amplitude")
plt.savefig("relative_expected_amplitude_linear.pdf")

# Plot period vs. amplitude
plt.figure()
plt.loglog(all_periods,
           all_amplitudes,
           ".", alpha=0.2)
plt.xlabel("Period")
plt.ylabel("Amplitude")
plt.savefig("period_amplitude.pdf")

# Plot period vs. amplitude in linear space.
plt.figure()
plt.plot(all_periods,
           all_amplitudes,
           ".", alpha=0.2)
plt.xlabel("Period")
plt.ylabel("Amplitude")
plt.yscale('log')
plt.savefig("period_amplitude_linear.pdf")

# Plot period vs. quality factor
plt.figure()
plt.loglog(all_periods,
           all_qualities,
           ".", alpha=0.2)
plt.xlabel("Period")
plt.ylabel("Quality factor")
plt.savefig("quality_factor.pdf")

# Histogram of number of modes
width = 0.7
bins = np.arange(0, posterior_sample[0, indices["max_num_components"]]+1)\
        - 0.5*width
plt.figure()
plt.hist(posterior_sample[:, indices["num_components"]],
         bins,
         width=width,
         alpha=0.3,
         density=True)
plt.xlabel("num_components")
plt.xticks([i for i in range(0, 21, 2)])
plt.xlim((0, 20))
plt.savefig("num_components.pdf")
