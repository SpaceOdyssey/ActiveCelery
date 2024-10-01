import dnest4.classic as dn4
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sample = dn4.my_loadtxt("sample.txt")
indices = dn4.load_column_names("sample.txt")["indices"]

# print("Converting posterior samples to YAML for extra convenience.")
# from to_yaml import posterior_sample, indices, to_yaml

# # Extract component hyperparameters.
# hp_keys = ['mu_log_quality', 'sig_log_quality']
# all_hp_keys = [indices[k] for k in hp_keys]
# all_hps = sample[:, all_hp_keys]

# Extract amplitudes
start = indices["amplitude[0]"]
end   = indices["period[0]"]
all_amplitudes = sample[:, start:end].flatten()
all_amplitudes = all_amplitudes[all_amplitudes != 0.0]

# Periods
start = indices["period[0]"]
end   = indices["quality[0]"]
all_periods = sample[:, start:end].flatten()
all_periods = all_periods[all_periods != 0.0]

# Extract quality factors
start = indices["quality[0]"]
all_qualities = sample[:, start:-4].flatten()
all_qualities = all_qualities[all_qualities != 0.0]

# Extract circadian component.
start = indices["amplitude[circ]"]
all_circ = sample[:, start:-1]

# Plot circadian component.
circ_keys = ["amplitude", "period", "quality"]
all_circ_df = pd.DataFrame(all_circ, columns = circ_keys)
all_circ_df["amplitude"] = np.log10(all_circ_df["amplitude"])
all_circ_df["quality"] = np.log10(all_circ_df["quality"])
sns.pairplot(all_circ_df)
plt.savefig('circadian_params.pdf')

# # Pairplot for component hyperparameters.
# all_hps_df = pd.DataFrame(all_hps, columns = hp_keys)
# # all_hps_df['log_mu_period'] = np.log(all_hps_df['mu_period'])
# # all_hps_df['mu_log_quality'] = np.log(all_hps_df['mu_quality'])
# # all_hps_df['log_scale_amplitude'] = np.log(all_hps_df['scale_amplitude'])
# # all_hps_df = all_hps_df.drop(columns = ['mu_period', 'mu_quality', 'scale_amplitude'])
# sns.pairplot(all_hps_df)
# plt.savefig('hyperparameters.pdf')

# Histogram of inferred log-periods
plt.figure()
plt.hist(all_periods, 100, alpha=0.3)
plt.xlabel(r"Period (days)")
plt.ylabel("Relative probability")
plt.savefig("relative_probability.pdf")

# Histogram of inferred periods, weighted by amplitude
plt.figure()
plt.hist(all_periods, bins=1000,
         weights=all_amplitudes, alpha=0.3)
plt.xlabel(r"Period (days)")
plt.ylabel("Relative expected amplitude")
plt.savefig("relative_expected_amplitude.pdf")

# Plot period vs. amplitude
plt.figure()
plt.plot(all_periods, all_amplitudes, ".", alpha=0.2)
plt.xlabel("Period (days)")
plt.ylabel("Amplitude")
plt.yscale('log')
plt.savefig("period_amplitude.pdf")

# Plot period vs. quality factor
plt.figure()
plt.plot(all_periods, all_qualities, ".", alpha=0.2)
plt.xlabel("Period")
plt.ylim(0.0, 1000.0)
plt.ylabel("Quality factor")
plt.savefig("quality_factor.pdf")

# Histogram of number of modes
width = 0.7
bins = np.arange(0, sample[0, indices["max_num_components"]]+1)\
        - 0.5*width
plt.figure()
plt.hist(sample[:, indices["num_components"]],
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
all_qualities = sample[:, start:-1]
n_quality_components = np.sum(all_qualities > 1, axis = 1)
width = 0.7
bins = np.arange(0, sample[0, indices["max_num_components"]]+1)\
        - 0.5*width
plt.figure()
plt.hist(n_quality_components,
         bins,
         width=width,
         alpha=0.3,
         density=True)
plt.xlabel("num_components")
plt.xticks([i for i in range(0, 21, 2)])
plt.xlim((0, 20))
plt.savefig("num_quality_components.pdf")
