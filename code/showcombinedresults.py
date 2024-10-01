# This needs to be run in the root directory of the multiple runs.

import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
import statistics

import glob

# Make combined posterior samples file.
column_names = ['num_dimensions', 'max_num_components', 'num_components'] + \
                [f'amplitude[{i}]' for i in range(20)] + \
                [f'period[{i}]' for i in range(20)] + \
                [f'quality[{i}]' for i in range(20)] + \
                ['amplitude[circ]', 'period[circ]', 'quality[circ]', 'sigma']

files = glob.glob('run*/posterior_sample.txt')

dfs = [pd.read_csv(f, sep = '\s+', header = None, names = column_names, comment = '#') for f in files]
for i, df in enumerate(dfs):
    df['chain'] = i

posterior_sample = pd.concat(dfs)
indices = {col: index for index, col in enumerate(posterior_sample.columns)}

# Extract amplitudes
start = indices["amplitude[0]"]
end   = indices["period[0]"]
all_amplitudes = posterior_sample.iloc[:, start:end].values.flatten()
all_amplitudes = all_amplitudes[all_amplitudes != 0.0]

# Periods
start = indices["period[0]"]
end   = indices["quality[0]"]
all_periods = posterior_sample.iloc[:, start:end].values.flatten()
all_periods = all_periods[all_periods != 0.0]

# Extract quality factors
start = indices["quality[0]"]
all_qualities = posterior_sample.iloc[:, start:-5].values.flatten()
all_qualities = all_qualities[all_qualities != 0.0]

# Extract circadian component.
start = indices["amplitude[circ]"]
all_circ = posterior_sample.iloc[:, start:-1]

# Plot circadian component.
circ_keys = ["amplitude[circ]", "period[circ]", "quality[circ]"]
all_circ_df = pd.DataFrame(all_circ, columns = circ_keys)
all_circ_df = all_circ_df[all_circ_df["quality[circ]"] <= 100.0]
all_circ_df["amplitude[circ]"] = np.log10(all_circ_df["amplitude[circ]"])
sns.pairplot(all_circ_df)
plt.savefig('circadian_params.pdf')

# Histogram of inferred log-periods
plt.figure()
plt.hist(all_periods, 100, alpha=0.3)
plt.xlim(0.0, 1.0)
plt.xlabel(r"Period (days)")
plt.ylabel("Relative probability")
plt.savefig("relative_probability.pdf")

# Histogram of inferred periods, weighted by amplitude
plt.figure()
plt.hist(all_periods, bins=1000,
         weights=all_amplitudes, alpha=0.3)
plt.xlim(0.0, 1.0)
plt.xlabel(r"Period (days)")
plt.ylabel("Relative expected amplitude")
plt.savefig("relative_expected_amplitude.pdf")

# Plot period vs. amplitude
plt.figure()
plt.plot(all_periods, all_amplitudes, ".", alpha=0.2)
plt.xlim(0.0, 1.0)
plt.xlabel("Period (days)")
plt.ylabel("Amplitude")
plt.yscale('log')
plt.savefig("period_amplitude.pdf")

# Plot period vs. quality factor
plt.figure()
plt.plot(all_periods, all_qualities, ".", alpha=0.2)
plt.xlim(0.0, 1.0)
plt.xlabel("Period")
plt.ylabel("Quality factor")
plt.yscale('log')
plt.savefig("quality_factor.pdf")

# Histogram of number of modes
width = 0.7
bins = np.arange(0, posterior_sample.iloc[0, indices["max_num_components"]]+1) - 0.5*width
plt.figure()
plt.hist(posterior_sample.iloc[:, indices["num_components"]],
         bins,
         width=width,
         alpha=0.3,
         density=True)
plt.xlabel("num_components")
plt.xticks([i for i in range(0, 21, 2)])
plt.xlim((0, 20))
plt.savefig("num_components.pdf")

# Histogram of high quality (Q > 1) modes
start = indices["quality[0]"]
all_qualities = posterior_sample.iloc[:, start:-4]
n_quality_components = np.sum(all_qualities > 1, axis = 1)
width = 0.7
bins = np.arange(0, posterior_sample.iloc[0, indices["max_num_components"]]+1) - 0.5*width
plt.figure()
plt.hist(n_quality_components,
         bins,
         width=width,
         alpha=0.3,
         density=True)
plt.xlabel("Number of Ultradian Cycles")
plt.xticks([i for i in range(0, 21, 2)])
plt.xlim((0, 20))
plt.savefig("num_quality_components.pdf")

# Create normalized histogram for each source
chains = posterior_sample['chain'].unique()
colors = ['blue', 'orange']  # Define a color for each source
bins = np.arange(0, posterior_sample.iloc[0, indices["max_num_components"]]+1) - 0.5
plt.figure()
for chain, color in zip(chains, colors):
    subset = posterior_sample[posterior_sample['chain'] == chain]
    counts, _ = np.histogram(subset['num_components'], bins = bins)
    normalised_counts = counts / counts.sum()  # Normalize counts
    plt.step(bins[:-1],
            normalised_counts,
            where='pre',
            alpha=0.25,
            label=chain,
            color=color)

plt.xticks([i for i in range(0, 21, 2)])
plt.xlim((-0.5, 20))

plt.xlabel('Number of components')
plt.ylabel('Probability')
plt.legend(title = 'Chains')
plt.savefig("num_components_per_chain.pdf")

# Summary statistics.
with open('summary_stats.txt', 'w') as f:
    f.write(f'num_quality_components_map: %s' % statistics.mode(n_quality_components))
