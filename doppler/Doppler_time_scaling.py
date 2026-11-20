"""
Doppler time-scaling compensation demo for a 5-microphone array (SRP-PHAT)

This script simulates a moving acoustic drone source received by a 5-mic array
and compares DOA estimation using SRP-PHAT:

  (A) WITHOUT Doppler time-scaling compensation
  (B) WITH Doppler time-scaling compensation via time warping / resampling

---------------------------------------------------------------------------
Mathematical formulation (used explicitly below)
---------------------------------------------------------------------------

Array model over a (short) time interval:
    x_m(t) ≈ s( α(t) t − τ_m(u) ) + n_m(t)         (Eq. 1)
where
  - s(·) is the emitted waveform in an emission-time-like parameterization
  - α(t) is a (common) Doppler time-scaling factor (far-field, small array)
  - τ_m(u) is the propagation delay to microphone m for DOA direction u
  - n_m(t) is additive noise

Far-field plane-wave delays (choose mic 0 as reference):
    Δτ_m(u) = ((r_m − r_0)^T u) / c                (Eq. 2)
and τ_m(u) = τ_0 + Δτ_m(u).

Doppler compensation via time warping:
Define the warped time coordinate
    u(t) = ∫_0^t α(ξ) dξ                           (Eq. 3)
and resample each channel onto a uniform grid in u:
    x̃_m(u) = x_m( t(u) )                          (Eq. 4)

If α(t) is correctly used, then (approximately)
    x̃_m(u) ≈ s( u − τ_m(u_true) ) + ñ_m(u)        (Eq. 5)
i.e., the Doppler time scaling is removed, and standard array processing applies.

SRP-PHAT:
For each mic pair (i,j), compute GCC-PHAT:
    R_ij^PHAT(τ) = F^{-1}{ (X_i X_j*) / |X_i X_j*| } (Eq. 6)
Then scan candidate directions u on a grid and score:
    P(u) = Σ_{i<j} R_ij^PHAT( Δτ_ij(u) )            (Eq. 7)
where Δτ_ij(u)=Δτ_j(u)−Δτ_i(u). DOA is argmax_u P(u).

---------------------------------------------------------------------------
What this simulation illustrates
---------------------------------------------------------------------------

We use a long frame duration (0.5 s) and a time-varying Doppler factor α(t),
which causes intra-frame "chirpiness" / time-warp in Eq. 1.
This smears GCC peaks and reduces SRP contrast.

We then apply the exact α(t) from the simulation to build u(t) (Eq. 3–4),
and show SRP-PHAT becomes sharper / closer to the true DOA.

Requirements:
  - numpy
  - scipy
  - matplotlib

Run:
  python doppler_srp_phat_demo.py
"""

import numpy as np
from numpy.fft import rfft, irfft
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# -----------------------------
# Configuration
# -----------------------------
fs = 16000                 # sample rate [Hz]
T = 0.5                    # frame duration [s] (your case)
N = int(fs * T)
t = np.arange(N) / fs
c = 343.0                  # speed of sound [m/s]

# 5-mic geometry: square + center in the x-y plane
# side length ~0.20 m (typical small array)
d = 0.20
r = np.array([
    [-d/2, -d/2, 0.0],
    [ d/2, -d/2, 0.0],
    [ d/2,  d/2, 0.0],
    [-d/2,  d/2, 0.0],
    [ 0.0,  0.0, 0.0],
], dtype=float)

M = r.shape[0]
ref = 4  # choose center as reference for convenience

# True DOA (azimuth only, 2D scan): u_true points "toward array" from source
az_true_deg = 35.0
az_true = np.deg2rad(az_true_deg)
u_true = np.array([np.cos(az_true), np.sin(az_true), 0.0])

# Propagation delays from Eq. 2 (relative to reference mic)
delta_tau = ((r - r[ref]) @ u_true) / c           # shape (M,)

# Emitted drone-like signal s(·): harmonics + mild AM
# (Doppler time scaling will "chirp" these harmonics when α(t) varies)
f0 = 180.0  # base harmonic spacing (not necessarily rotor BPF; just for tone structure)
harmonics = [1, 2, 3, 4, 5, 6]
amps = np.array([1.0, 0.6, 0.4, 0.3, 0.2, 0.15])

# Amplitude modulation (e.g., beating) at ~3 Hz
am = 1.0 + 0.15 * np.sin(2*np.pi*3.0*t)

def s_emitted(u_time):
    """
    s(u): emitted signal as a function of 'emission-time-like' variable u.
    We evaluate it at (α(t)t - τ_m) for each mic (Eq. 1).
    """
    sig = np.zeros_like(u_time, dtype=float)
    for k, a in zip(harmonics, amps):
        sig += a * np.sin(2*np.pi*(k*f0) * u_time)
    # apply slow AM using receiver-time AM as an approximation
    # (AM also gets time-warped under Doppler; this is fine for demonstration)
    return sig

# Time-varying Doppler factor α(t) (common across array, far field)
# We simulate an approaching drone with changing radial speed during the 0.5 s.
# α(t) = c / (c - v_r(t))
v0 = 10.0        # initial radial speed toward array [m/s]
a = 30.0         # radial acceleration [m/s^2] -> changes Doppler within the 0.5s window
v_r = v0 + a * (t - T/2)  # ramp around mid-frame (can be negative at start, positive later)
# clamp to keep physically reasonable and avoid v_r close to c
v_r = np.clip(v_r, -20.0, 25.0)

alpha = c / (c - v_r)  # Eq. 1 Doppler time-scaling factor

# Noise level
snr_db = 5.0


# -----------------------------
# Simulate received signals (Eq. 1)
# -----------------------------
# x_m(t) = s( α(t) t − τ_m ) + n_m(t)
X = np.zeros((M, N), dtype=float)

for m in range(M):
    u_arg = alpha * t - delta_tau[m]  # Eq. 1 argument
    X[m] = am * s_emitted(u_arg)

# Add white Gaussian noise at desired SNR
signal_power = np.mean(X[ref]**2)
snr_lin = 10**(snr_db/10)
noise_power = signal_power / snr_lin
rng = np.random.default_rng(0)
X += rng.normal(scale=np.sqrt(noise_power), size=X.shape)

# Optional bandpass (typical for drones). This helps PHAT stability.
def bandpass(x, f_lo=200.0, f_hi=3000.0, fs=fs, order=4):
    b, a = butter(order, [f_lo/(fs/2), f_hi/(fs/2)], btype='band')
    return filtfilt(b, a, x)

for m in range(M):
    X[m] = bandpass(X[m])

# -----------------------------
# Doppler compensation via time warping (Eq. 3–4)
# -----------------------------
# Build u(t) = ∫ α(ξ) dξ by discrete integration (Eq. 3)
u = np.cumsum(alpha) / fs  # u[n] ≈ Σ α[i] Ts
# Resample onto a uniform u-grid with same sample rate in u-domain
u_uniform = np.arange(N) / fs

# Invert mapping t(u): since u(t) is monotonic, interpolate t as a function of u.
# t is known (original time). We want t_at_u_uniform = t(u_uniform).
t_at_u = np.interp(u_uniform, u, t, left=t[0], right=t[-1])

# Apply same time warp to every mic (Eq. 4)
Xc = np.zeros_like(X)
for m in range(M):
    Xc[m] = np.interp(t_at_u, t, X[m])


# -----------------------------
# SRP-PHAT implementation (Eq. 6–7), 2D azimuth scan
# -----------------------------
def gcc_phat(xi, xj, eps=1e-12):
    """
    GCC-PHAT in one shot over the full frame:
      R_ij^PHAT(τ) = IFFT{ (X_i X_j*) / |X_i X_j*| }   (Eq. 6)
    Returns correlation over lags in samples in range [-(N-1), ..., (N-1)] (circular-ish).
    We use an FFT length with zero-padding to get finer lag resolution.
    """
    n = len(xi)
    nfft = 2**int(np.ceil(np.log2(4*n)))  # generous zero padding for sharper peak sampling
    Xi = rfft(xi, nfft)
    Xj = rfft(xj, nfft)
    G = Xi * np.conj(Xj)
    W = G / (np.abs(G) + eps)
    r = irfft(W, nfft)
    # shift to have negative lags at the end; create symmetric lag indexing
    # For SRP, we just need to sample r at predicted lags; we’ll use modular indexing.
    return r, nfft

def srp_phat(Xmulti, r_pos, az_grid_deg, c=343.0, fs=16000):
    """
    Compute SRP-PHAT spatial spectrum over an azimuth grid (Eq. 7).
    - Xmulti: shape (M, N)
    - r_pos: mic positions shape (M, 3)
    Returns P[az] and predicted DOA = argmax.
    """
    M, N = Xmulti.shape

    # Precompute GCC-PHAT for all pairs
    pairs = []
    gcc = []
    nffts = []
    for i in range(M):
        for j in range(i+1, M):
            rij, nfft = gcc_phat(Xmulti[i], Xmulti[j])
            pairs.append((i, j))
            gcc.append(rij)
            nffts.append(nfft)

    # Evaluate SRP score for each azimuth
    P = np.zeros(len(az_grid_deg), dtype=float)

    # Reference mic for relative delays (can be any; SRP uses pairwise TDOAs anyway)
    for a_idx, az_deg in enumerate(az_grid_deg):
        az = np.deg2rad(az_deg)
        u_dir = np.array([np.cos(az), np.sin(az), 0.0])  # candidate DOA unit vector

        # Predicted relative delays (Eq. 2)
        tau = (r_pos @ u_dir) / c  # absolute up to constant; pairwise differences matter

        score = 0.0
        for (k, (i, j)) in enumerate(pairs):
            dtau = tau[j] - tau[i]                     # Δτ_ij(u)
            lag_samp = int(np.round(dtau * fs))         # lag in samples

            # Sample GCC at that lag. Using modular indexing (circular corr from IFFT).
            nfft = nffts[k]
            idx = lag_samp % nfft
            score += gcc[k][idx]

        P[a_idx] = score

    az_hat = az_grid_deg[int(np.argmax(P))]
    return P, az_hat

# Azimuth search grid (2D)
az_grid = np.linspace(-90, 90, 721)  # 0.25 degree resolution

# SRP-PHAT without and with compensation
P_raw, az_hat_raw = srp_phat(X,  r, az_grid, c=c, fs=fs)
P_cmp, az_hat_cmp = srp_phat(Xc, r, az_grid, c=c, fs=fs)

# -----------------------------
# Plot results
# -----------------------------
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel 1: SRP-PHAT without compensation
ax1 = axes[0, 0]
ax1.plot(az_grid, P_raw, 'b-', linewidth=1.5)
ax1.axvline(az_true_deg, color='g', linestyle='--', linewidth=2, label=f'True: {az_true_deg}°')
ax1.axvline(az_hat_raw, color='r', linestyle='--', linewidth=2, label=f'Est: {az_hat_raw:.2f}°')
ax1.set_title("SRP-PHAT (NO Doppler compensation)", fontsize=12, fontweight='bold')
ax1.set_xlabel("Azimuth (deg)")
ax1.set_ylabel("SRP-PHAT score")
ax1.grid(True, alpha=0.3)
ax1.legend()
error_raw = az_hat_raw - az_true_deg
ax1.text(0.05, 0.95, f'Error: {error_raw:.2f}°', transform=ax1.transAxes,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='red', alpha=0.3))

# Panel 2: SRP-PHAT with compensation
ax2 = axes[0, 1]
ax2.plot(az_grid, P_cmp, 'b-', linewidth=1.5)
ax2.axvline(az_true_deg, color='g', linestyle='--', linewidth=2, label=f'True: {az_true_deg}°')
ax2.axvline(az_hat_cmp, color='r', linestyle='--', linewidth=2, label=f'Est: {az_hat_cmp:.2f}°')
ax2.set_title("SRP-PHAT (WITH Doppler compensation)", fontsize=12, fontweight='bold')
ax2.set_xlabel("Azimuth (deg)")
ax2.set_ylabel("SRP-PHAT score")
ax2.grid(True, alpha=0.3)
ax2.legend()
error_cmp = az_hat_cmp - az_true_deg
ax2.text(0.05, 0.95, f'Error: {error_cmp:.2f}°', transform=ax2.transAxes,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='green', alpha=0.3))

# Panel 3: Time-varying Doppler factor α(t)
ax3 = axes[1, 0]
ax3.plot(t, alpha, 'r-', linewidth=2)
ax3.axhline(1.0, color='k', linestyle='--', linewidth=1, alpha=0.5, label='α = 1 (no Doppler)')
ax3.set_title("Time-varying Doppler time-scaling factor α(t)", fontsize=12, fontweight='bold')
ax3.set_xlabel("Time (s)")
ax3.set_ylabel("α(t) = c / (c - v_r)")
ax3.grid(True, alpha=0.3)
ax3.legend()
alpha_range = np.max(alpha) - np.min(alpha)
ax3.text(0.05, 0.95, f'Range: {np.min(alpha):.4f} to {np.max(alpha):.4f}\nΔα: {alpha_range:.4f}',
         transform=ax3.transAxes, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7), fontsize=9)

# Panel 4: Comparison summary
ax4 = axes[1, 1]
ax4.axis('off')

summary_text = f"""DOPPLER TIME-SCALING COMPENSATION RESULTS

True Azimuth: {az_true_deg:.2f}°

WITHOUT Compensation:
  Estimated: {az_hat_raw:.2f}°
  Error: {error_raw:.2f}°

WITH Compensation:
  Estimated: {az_hat_cmp:.2f}°
  Error: {error_cmp:.2f}°

Improvement:
  Error reduction: {abs(error_raw) - abs(error_cmp):.2f}°
  Percentage: {(1 - abs(error_cmp)/abs(error_raw))*100:.1f}%

Simulation Parameters:
  Frame duration: {T} s
  Sample rate: {fs} Hz
  Number of mics: {M}
  Array size: {d*1000:.0f} mm
  SNR: {snr_db} dB
  Doppler factor range: {np.min(alpha):.4f} - {np.max(alpha):.4f}
"""

ax4.text(0.1, 0.5, summary_text, transform=ax4.transAxes,
         verticalalignment='center', fontsize=10, family='monospace',
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

plt.suptitle('Doppler Time-Scaling Compensation for SRP-PHAT DOA Estimation',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('doppler_time_scaling_comparison.png', dpi=150, bbox_inches='tight')
print("\nSaved: doppler_time_scaling_comparison.png")

print("\nTrue azimuth (deg):", az_true_deg)
print("Estimated azimuth WITHOUT compensation (deg):", float(az_hat_raw))
print("Estimated azimuth WITH compensation    (deg):", float(az_hat_cmp))

plt.show()
