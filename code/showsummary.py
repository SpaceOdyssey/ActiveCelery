# This should be run from the root directory of the data folders.
from pathlib import Path
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sum_stats = glob.glob("./*/summary_stats.txt")

s = []
for s_file in sum_stats:
    s_path = Path(s_file)
    file = str(s_path.parent)
    s_tmp = pd.read_csv(s_path,
                        header = None,
                        sep = ' ',
                        index_col = 0,
                        engine = 'python').T
    s_tmp['file'] = file
    s.append(s_tmp)

s = pd.concat(s)

# Distribution of number of components estimated using the MAP.
bins = np.arange(0, 11) - 0.5
counts, _ = np.histogram(s['num_quality_components_map'], bins = bins)
plt.figure()
plt.xlabel('Number of cycles')
plt.step(bins[:-1],
            counts,
            where='post',
            alpha=1.0)
plt.savefig('num_components.pdf')

# Signal-to-noise ratio.
fig, ax = plt.subplots(1, 3, figsize = (12, 8))
ax[0].hist(s['circ_amplitude_mean'])
ax[0].set_xlabel('Mean amplitude')

ax[1].hist(s['sigma_mean'])
ax[1].set_xlabel('White noise')

s['snr'] = s['circ_amplitude_mean']**2/s['sigma_mean']**2
bins = np.logspace(np.log10(0.8*min(s['snr'])),np.log10(1.2*max(s['snr'])), 20)
ax[2].hist(s['snr'], bins = bins)
ax[2].set_xscale('log')
ax[2].set_xlabel('SNR (MeanAmp**2/MeanNoise**2)')

plt.savefig('signal_noise.pdf')

# MAP components compared to SNR.
summary = s.groupby('num_quality_components_map')['snr'].agg(['mean', 'std', 'count'])
summary['se'] = summary['std'] / np.sqrt(summary['count'])
plt.figure()
plt.errorbar(summary.index, summary['mean'],
             yerr = summary['se'],
             marker = '.',
             markersize=5)
plt.xlabel('MAP Number of Components')
plt.ylabel('SNR (MeanAmp**2/MeanNoise**2)')
plt.savefig('num_components_snr.pdf')
