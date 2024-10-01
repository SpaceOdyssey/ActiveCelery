import sys
import dnest4.classic as dn4
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
import statistics

print("Generating DNest4 plots. Close these to continue.")

# Inputs
inputs = pd.read_csv("inputs.csv", index_col=0, header=None).T
component_inputs = pd.read_csv("component_inputs.csv", index_col=0, header=None).T

# Run postprocess from DNest4
dn4.postprocess()

print("Converting posterior samples to YAML for extra convenience.")
from to_yaml import posterior_sample, indices, to_yaml
to_yaml()

# Extract amplitudes
start = indices["amplitude[0]"]
end   = indices["period[0]"]
y_sd = posterior_sample[1, indices["y_sd"]]
ultradian_amplitudes = y_sd*posterior_sample[:, start:end]
circ_amplitudes = y_sd*posterior_sample[:, indices["amplitude[circ]"]]
all_amplitudes = np.concatenate([ultradian_amplitudes, circ_amplitudes[:, np.newaxis]],
                                axis = 1).flatten()
all_amplitudes = all_amplitudes[all_amplitudes != 0.0]

# Periods
start = indices["period[0]"]
end   = indices["quality[0]"]
ultradian_periods = posterior_sample[:, start:end]
circ_periods = posterior_sample[:, indices["period[circ]"]]
all_periods = np.concatenate([ultradian_periods, circ_periods[:, np.newaxis]],
                                axis = 1).flatten()
all_periods = all_periods[all_periods != 0.0]

# Extract quality factors
start = indices["quality[0]"]
end = indices["amplitude[circ]"]
ultradian_qualities = posterior_sample[:, start:end]
circ_qualities = posterior_sample[:, indices["quality[circ]"]]
all_qualities = np.concatenate([ultradian_qualities, circ_qualities[:, np.newaxis]],
                                axis = 1).flatten()
all_qualities = all_qualities[all_qualities != 0.0]

# Extract circadian component.
start = indices["amplitude[circ]"]
end = indices["sigma"]
all_circ = posterior_sample[:, start:end]

# Plot circadian component.
circ_keys = ["amplitude", "period", "quality"]
all_circ_df = pd.DataFrame(all_circ, columns = circ_keys)
all_circ_df = all_circ_df[all_circ_df["quality"] <= 100.0]
all_circ_df["amplitude"] = np.log10(all_circ_df["amplitude"])
sns.pairplot(all_circ_df)
plt.savefig('circadian_params.pdf')

# Histogram of inferred log-periods
plt.figure()
plt.hist(all_periods, 100, alpha=0.5)
plt.vlines(component_inputs["period"], 0.0, plt.gca().get_ylim()[1],
           linestyles = "dashed", colors = 'k')
plt.xlim(0.0, 1.3)
plt.xlabel(r"Period (days)")
plt.ylabel("Relative probability")
plt.savefig("relative_probability.pdf")

# Histogram of inferred periods, weighted by amplitude
plt.figure()
plt.hist(all_periods, bins=100,
         weights=all_amplitudes, alpha=0.3)
plt.vlines(component_inputs["period"], 0.0, plt.gca().get_ylim()[1],
           linestyles = "dashed", colors = 'k')
plt.xlim(0.0, 1.3)
plt.xlabel(r"Period (days)")
plt.ylabel("Relative expected amplitude")
plt.savefig("relative_expected_amplitude.pdf")

# Plot period vs. amplitude
plt.figure()
plt.plot(all_periods, all_amplitudes, ".", alpha=0.2)
plt.plot(component_inputs["period"], component_inputs["amplitude"], "*")
plt.xlim(0.0, 1.3)
plt.xlabel("Period (days)")
plt.ylabel("Amplitude")
plt.yscale('log')
plt.savefig("period_amplitude.pdf")

# Plot period vs. quality factor
plt.figure()
plt.plot(all_periods, all_qualities, ".", alpha=0.2)
plt.plot(component_inputs["period"], component_inputs["quality"], "*")
plt.xlim(0.0, 1.3)
plt.xlabel("Period")
plt.ylabel("Quality factor")
plt.yscale('log')
plt.savefig("quality_factor.pdf")

# Histogram of number of modes
width = 0.7
bins = np.arange(0, posterior_sample[0, indices["max_num_components"]]+1) - 0.5*width
plt.figure()
plt.hist(posterior_sample[:, indices["num_components"]],
         bins,
         width=width,
         alpha=0.3,
         density=True)
plt.vlines(component_inputs.shape[0] - 1, 0.0, plt.gca().get_ylim()[1],
           linestyles = "dashed", colors = 'k')
plt.xlabel("num_components")
plt.xticks([i for i in range(0, 21, 2)])
plt.xlim((0, 20))
plt.savefig("num_components.pdf")

# Histogram of high quality (Q > 1) modes
start = indices["quality[0]"]
end = indices["amplitude[circ]"]
all_qualities = posterior_sample[:, start:end]
n_quality_components = np.sum(all_qualities > 1, axis = 1)
width = 0.7
bins = np.arange(0, posterior_sample[0, indices["max_num_components"]]+1) - 0.5*width
plt.figure()
plt.hist(n_quality_components,
         bins,
         width=width,
         alpha=0.3,
         density=True)
plt.vlines(component_inputs.shape[0] - 1, 0.0, plt.gca().get_ylim()[1],
           linestyles = "dashed", colors = 'k')
plt.xlabel("Number of Ultradian Cycles")
plt.xticks([i for i in range(0, 21, 2)])
plt.xlim((0, 20))
plt.savefig("num_quality_components.pdf")

# Summary statistics.
with open('summary_stats.txt', 'w') as f:
    f.write(f'num_quality_components_map: %s' % statistics.mode(n_quality_components))
