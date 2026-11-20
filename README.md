# simply-aoa (archived)

> **This repository is archived.** It holds the exploratory scripts written
> during the internship. The maintained code, with one shared array geometry,
> tests, and the simulations used in the report, is
> [acoustic-source-localisation](https://github.com/Coman-Radu/acoustic-source-localisation).
> Use that for anything new.

The scripts here estimate the direction of arrival (DOA) of drone noise with a
MiniDSP UMA-16 (4×4, 42 mm spacing) or UMA-8 (circular) microphone array,
using time delays between microphones. They are kept as a record of how the
work developed, not as a library.

## Layout

| Folder | Contents |
|---|---|
| `realtime/` | Live-capture scripts (`rt_*.py`, `sound_device_*.py`), one per estimator: delay-and-sum, GCC-PHAT, MUSIC, ESPRIT, combined; `uac_detect.py` lists audio devices |
| `analysis/` | Offline estimation and plotting: WAV plots, time/frequency/phase-domain estimates, Hilbert-transform AoA, GCC-PHAT and SRP analysis of recorded drone audio |
| `doppler/` | Doppler studies for moving sources; see [`doppler/DOPPLER_ANALYSIS.md`](doppler/DOPPLER_ANALYSIS.md) |
| `Paper Testing/` | First simulation experiments for the report (`exp1`–`exp3`, `sim_common.py`) |
| `Pictures/` | Generated figures |
| root | The original tracked scripts: `mic_array_plot.py`, `interactive_aoa.py`, `test_example.py`, `config.json` |

Run scripts from the repository root, for example
`python realtime/rt_beamforming_v2.py`: they open audio files and save figures
using paths relative to the current directory.

Audio is not committed (see `.gitignore`). The drone recordings come from the
public DADS corpus; see the data notes in acoustic-source-localisation for how
to obtain them.

## Known issues

These are documented rather than fixed, because the code is archived.

**Steering-delay sign.** Several scripts apply the steering delay with the
wrong sign, so delay-and-sum points 180° away from the source. It is fixed in
`realtime/rt_beamforming_v2.py`. The same delay line (`time_delay =
-distance_proj / SPEED_OF_SOUND`) remains in:

- `realtime/rt_beamforming_circular.py` (UMA-8)
- `analysis/real_drone_srp_analysis.py`
- `analysis/plot_gccphat.py`, `analysis/plot_gccphat_freq_time.py`
- `Paper Testing/rt_beamforming.py`
- `realtime/rt_music.py`, `realtime/rt_combined.py`, which convert the delay
  to a phase, so whether their result is wrong depends on their phase
  convention; untested

**Channel map.** The capture-channel-to-microphone map differs between
scripts. Five assign 15 of 16 channels to different positions than the
as-wired map used by `realtime/rt_beamforming_v2.py`:

- `Paper Testing/sim_common.py`, `Paper Testing/snr_filter_comparison.py`,
  `Paper Testing/rt_beamforming.py`
- `realtime/rt_sound_device_4x4_array.py`,
  `realtime/rt_sound_device_4x4_array_avg.py`

In pure simulation this cancels out, because the same map synthesises and
processes the signals; on real recordings these scripts report wrong bearings.

**UMA-8 geometry.** `realtime/rt_beamforming_circular.py` assumes one centre
microphone and seven on a 45 mm ring. Check this against the UMA-8 manual
before trusting its bearings.

**`test_example.py`** calls a function that is commented out in
`mic_array_plot.py`, so it does not run. Run `python mic_array_plot.py` and
see `Pictures/` instead.

## Method notes

Microphones along each axis of the array give one time delay per pair. Row
pairs give the azimuth and column pairs the elevation. For a pair spaced
$d$ apart and a plane wave arriving at angle $\theta$:

$$\Delta t = \frac{d \sin\theta}{c}$$

To avoid spatial aliasing the spacing must satisfy $d \le \lambda/2$, which for
the UMA-16's 42 mm spacing limits useful frequencies to about 4.1 kHz.

For moving sources the Doppler scripts use retarded time: for each received
sample at $t_r$, solve $t_r = t_e + r(t_e)/c$ for the emission time $t_e$ and
sample the source there with $1/r$ attenuation, so the Doppler shift emerges
from the geometry. Results and recommendations from those studies are in
[`doppler/DOPPLER_ANALYSIS.md`](doppler/DOPPLER_ANALYSIS.md).
