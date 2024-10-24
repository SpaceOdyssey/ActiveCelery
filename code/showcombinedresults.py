# This needs to be run in the root directory of the multiple runs.

import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
import statistics

import glob

# Get data
t_predict = pd.read_csv('./t_predict.txt', header = None, names = ['t'])
data = pd.read_csv('./data.txt', header = None, sep = ' ', names = ['t', 'y'])

# Make combined posterior samples file.
max_num_components = 20
column_names = ['num_dimensions', 'max_num_components', 'num_components'] + \
     [f'amplitude[{i}]' for i in range(max_num_components)] + \
     [f'period[{i}]' for i in range(max_num_components)] + \
     [f'quality[{i}]' for i in range(max_num_components)] + \
     ['amplitude[circ]', 'period[circ]', 'quality[circ]', 'sigma', 'y_mean', 'y_sd'] + \
     [f'y_predict[{i}]' for i in range(t_predict.shape[0])] + \
     [f'y_circadian[{i}]' for i in range(t_predict.shape[0])] + \
     [f'y_ultradian[{i}]' for i in range(t_predict.shape[0])] + \
     [f'y_trend[{i}]' for i in range(t_predict.shape[0])]

files = glob.glob('run*/posterior_sample.txt')

dfs = [pd.read_csv(f, sep = '\s+', header = None, names = column_names, comment = '#') for f in files]
for i, df in enumerate(dfs):
    df['chain'] = i

posterior_sample = pd.concat(dfs)
indices = {col: index for index, col in enumerate(posterior_sample.columns)}

# Extract amplitudes
start = indices["amplitude[0]"]
end   = indices["period[0]"]
y_sd = posterior_sample.iloc[1, indices["y_sd"]]
ultradian_amplitudes = y_sd*posterior_sample.iloc[:, start:end]
circ_amplitudes = y_sd*posterior_sample.iloc[:, indices["amplitude[circ]"]]
all_amplitudes = pd.concat([ultradian_amplitudes, circ_amplitudes],
                                axis = 1).values.flatten()
all_amplitudes = all_amplitudes[all_amplitudes != 0.0]

# Periods
start = indices["period[0]"]
end   = indices["quality[0]"]
ultradian_periods = posterior_sample.iloc[:, start:end]
circ_periods = posterior_sample.iloc[:, indices["period[circ]"]]
all_periods = pd.concat([ultradian_periods, circ_periods], axis = 1).values.flatten()
all_periods = all_periods[all_periods != 0.0]

# Extract quality factors
start = indices["quality[0]"]
end = indices["amplitude[circ]"]
ultradian_qualities = posterior_sample.iloc[:, start:end]
circ_qualities = posterior_sample.iloc[:, indices["quality[circ]"]]
all_qualities = pd.concat([ultradian_qualities, circ_qualities], axis = 1).values.flatten()
all_qualities = all_qualities[all_qualities != 0.0]

# Extract circadian component.
start = indices["amplitude[circ]"]
all_circ = posterior_sample.iloc[:, start:-1]

# Plot circadian components as a pairs plot.
circ_keys = ["amplitude[circ]", "period[circ]", "quality[circ]", "chain"]
all_circ_df = posterior_sample[circ_keys]
all_circ_df = all_circ_df[all_circ_df["quality[circ]"] <= 100.0]
all_circ_df["amplitude[circ]"] = np.log10(all_circ_df["amplitude[circ]"])
sns.pairplot(all_circ_df, hue = "chain")
plt.savefig('circadian_params.pdf')

# Plot the non-component parameters as a pairs plot.
const_keys = ["amplitude[circ]", "period[circ]", "quality[circ]",
              "sigma", "num_components", "chain"]
const_df = posterior_sample[const_keys]
const_df = const_df[const_df["quality[circ]"] <= 100.0]
const_df["amplitude[circ]"] = np.log10(const_df["amplitude[circ]"])
sns.pairplot(const_df, hue = 'chain')
plt.savefig('const_params.pdf')

# Histogram of inferred log-periods
plt.figure()
plt.hist(all_periods, 100, alpha=0.3)
plt.xlim(0.0, 1.3)
plt.xlabel(r"Period (days)")
plt.ylabel("Relative probability")
plt.savefig("relative_probability.pdf")

# Histogram of inferred periods, weighted by amplitude
plt.figure()
plt.hist(all_periods, bins=1000,
         weights=all_amplitudes, alpha=0.3)
plt.xlim(0.0, 1.3)
plt.xlabel(r"Period (days)")
plt.ylabel("Relative expected amplitude")
plt.savefig("relative_expected_amplitude.pdf")

# Plot period vs. amplitude
plt.figure()
plt.plot(all_periods, all_amplitudes, ".", alpha=0.2)
plt.xlim(0.0, 1.3)
plt.xlabel("Period (days)")
plt.ylabel("Amplitude")
plt.yscale('log')
plt.savefig("period_amplitude.pdf")

# Plot period vs. quality factor
plt.figure()
plt.plot(all_periods, all_qualities, ".", alpha=0.2)
plt.xlim(0.0, 1.3)
plt.xlabel("Period")
plt.ylabel("Quality factor")
plt.yscale('log')
plt.savefig("quality_factor.pdf")

# Plot the white noise component.
plt.figure()
plt.hist(posterior_sample.iloc[:, indices["sigma"]])
plt.savefig('white_noise.pdf')

# Create normalised histogram of number of components for each chain.
chains = posterior_sample.iloc[:, indices["chain"]].unique()
colors = ['blue', 'orange']
bins = np.arange(0, posterior_sample.iloc[0, indices["max_num_components"]]+1) - 0.5
plt.figure()
for chain, color in zip(chains, colors):
    subset = posterior_sample[posterior_sample['chain'] == chain]
    n_components = 1 + subset['num_components']
    counts, _ = np.histogram(n_components, bins = bins)
    normalised_counts = counts / counts.sum()  # Normalize counts
    plt.step(bins[:-1],
            normalised_counts,
            where='post',
            alpha=0.25,
            label=1+chain,
            color=color)

n_components = 1 + posterior_sample['num_components']
counts, _ = np.histogram(n_components, bins = bins)
normalised_counts = counts / counts.sum()  # Normalize counts
plt.step(bins[:-1],
         normalised_counts,
         where='post',
         label='total',
         color='k')

plt.xticks([i for i in range(0, max_num_components + 1, 2)])
plt.xlim((-0.5, max_num_components))

plt.xlabel('Number of components')
plt.ylabel('Probability')
plt.legend(title = 'Chains')
plt.savefig("num_components_per_chain.pdf")

# Create normalised histogram of number of quality components for each chain.
chains = posterior_sample['chain'].unique()
colors = ['blue', 'orange']  # Define a color for each source
bins = np.arange(0, posterior_sample.iloc[0, indices["max_num_components"]]+1) - 0.5
plt.figure()
for chain, color in zip(chains, colors):
    subset = posterior_sample[posterior_sample['chain'] == chain]
    start = indices["quality[0]"]
    end = indices["amplitude[circ]"]
    sub_ultradian_qualities = subset.iloc[:, start:end]
    sub_circ_qualities = subset.iloc[:, indices["quality[circ]"]]
    sub_all_qualities = pd.concat([sub_ultradian_qualities, sub_circ_qualities], axis = 1)
    n_sub_quality_components = np.sum(sub_all_qualities > 1, axis = 1)
    counts, _ = np.histogram(n_sub_quality_components, bins = bins)
    normalised_counts = counts / counts.sum()  # Normalize counts
    plt.step(bins[:-1],
             normalised_counts,
             where='post',
             alpha=0.25,
             label=1+chain,
             color=color)

start = indices["quality[0]"]
end = indices["amplitude[circ]"]
ultradian_qualities = posterior_sample.iloc[:, start:end]
circ_qualities = posterior_sample.iloc[:, indices["quality[circ]"]]
all_qualities = pd.concat([ultradian_qualities, circ_qualities], axis = 1)
n_quality_components = np.sum(all_qualities > 1, axis = 1)
counts, _ = np.histogram(n_quality_components, bins = bins)
normalised_counts = counts / counts.sum()  # Normalize counts
plt.step(bins[:-1],
         normalised_counts,
         where='post',
         label='total',
         color='k')

plt.xticks([i for i in range(0, max_num_components + 1, 2)])
plt.xlim((-0.5, max_num_components))

plt.xlabel('Number of quality components')
plt.ylabel('Probability')
plt.legend(title = 'Chains')
plt.savefig("num_quality_components_per_chain.pdf")

# Plot posterior predictive distribution.
y_mean = posterior_sample["y_mean"].values[1]

fig, ax = plt.subplots(4, 1, figsize = (30, 10), sharex = True, sharey = True)

start = indices["y_predict[0]"]
end = indices["y_circadian[0]"]
ax[0].plot(t_predict['t'], y_sd*posterior_sample.iloc[:, start:end].T + y_mean, color='k', alpha=0.1)
ax[0].scatter(data['t'], data['y'], color = 'r')
an_xy = (0.9*np.max(data['t']), 0.8*np.max(data['y']))
ax[0].annotate('Total model', xy=an_xy, fontsize=12.0)

# Plot posterior circadian component.
start = indices["y_circadian[0]"]
end = indices["y_ultradian[0]"]
ax[1].plot(t_predict['t'], y_sd*posterior_sample.iloc[:, start:end].T + y_mean, color='k', alpha=0.1)
ax[1].scatter(data['t'], data['y'], color = 'r')
ax[1].annotate('Circadian model', xy=an_xy, fontsize=12.0)

# Plot posterior ultradian components.
start = indices["y_ultradian[0]"]
end = indices["y_trend[0]"]
ax[2].plot(t_predict['t'], y_sd*posterior_sample.iloc[:, start:end].T + y_mean, color = 'k', alpha=0.1)
ax[2].scatter(data['t'], data['y'], color = 'r')
ax[2].annotate('Ultradian model', xy=an_xy, fontsize=12.0)

# Plot posterior ultradian components.
start = indices["y_trend[0]"]
end = indices["chain"]
ax[3].plot(t_predict['t'], y_sd*posterior_sample.iloc[:, start:end].T + y_mean, color='k', alpha=0.1)
ax[3].scatter(data['t'], data['y'], color = 'r')
ax[3].set_xlabel('Days since baseline', fontsize=16.0)
ax[3].annotate('Trend model', xy=an_xy, fontsize=12.0)

plt.subplots_adjust(hspace = 0)
plt.tight_layout()

plt.savefig('log_predict.pdf')

# Plot posterior predictive in linear space.
y_mean = posterior_sample["y_mean"].values[1]

fig, ax = plt.subplots(4, 1, figsize = (30, 10), sharex = True, sharey = True)

start = indices["y_predict[0]"]
end = indices["y_circadian[0]"]
ax[0].plot(t_predict['t'], np.exp(y_sd*posterior_sample.iloc[:, start:end].T + y_mean), color='k', alpha=0.1)
ax[0].scatter(data['t'], np.exp(data['y']), color = 'r')
an_xy = (0.9*np.max(data['t']), 0.8*np.max(data['y']))
ax[0].annotate('Total model', xy=an_xy, fontsize=12.0)

# Plot posterior circadian component.
start = indices["y_circadian[0]"]
end = indices["y_ultradian[0]"]
ax[1].plot(t_predict['t'], np.exp(y_sd*posterior_sample.iloc[:, start:end].T + y_mean), color='k', alpha=0.1)
ax[1].scatter(data['t'], data['y'], color = 'r')
ax[1].annotate('Circadian model', xy=an_xy, fontsize=12.0)

# Plot posterior ultradian components.
start = indices["y_ultradian[0]"]
end = indices["y_trend[0]"]
ax[2].plot(t_predict['t'], np.exp(y_sd*posterior_sample.iloc[:, start:end].T + y_mean), color = 'k', alpha=0.1)
ax[2].scatter(data['t'], np.exp(data['y']), color = 'r')
ax[2].annotate('Ultradian model', xy=an_xy, fontsize=12.0)

# Plot posterior ultradian components.
start = indices["y_trend[0]"]
end = indices["chain"]
ax[3].plot(t_predict['t'], np.exp(y_sd*posterior_sample.iloc[:, start:end].T + y_mean), color='k', alpha=0.1)
ax[3].scatter(data['t'], np.exp(data['y']), color = 'r')
ax[3].set_xlabel('Days since baseline', fontsize=16.0)
ax[3].annotate('Trend model', xy=an_xy, fontsize=12.0)

plt.subplots_adjust(hspace = 0)
plt.tight_layout()

plt.savefig('predict.pdf')

# Summary statistics.
with open('summary_stats.txt', 'w') as f:
    f.write(f'num_quality_components_map %s\n' % np.argmax(counts))
    f.write(f'circ_amplitude_mean %s\n' % np.mean(posterior_sample.iloc[:, indices["amplitude[circ]"]]))
    f.write(f'sigma_mean %s\n' % np.mean(posterior_sample.iloc[:, indices["sigma"]]))
