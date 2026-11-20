"""
Doppler time-scaling compensation demo for a 7-microphone CIRCULAR array (SRP-PHAT)

This script simulates a moving acoustic drone source received by a 7-mic circular array
and compares DOA estimation using SRP-PHAT:

  (A) WITHOUT Doppler time-scaling compensation
  (B) WITH Doppler time-scaling compensation via time warping / resampling

---------------------------------------------------------------------------
Mathematical formulation (explicitly referenced in code)
---------------------------------------------------------------------------

Array model over a (short) time interval:
    x_m(t) ≈ s( α(t) t − τ_m(u) ) + n_m(t)                      (Eq. 1)
where
  - s(·) is the emitted waveform in an emission-time-like parameterization
  - α(t) is a (common) Doppler time-scaling factor (far-field, small array)
  - τ_m(u) is the propagation delay to microphone m for DOA direction u
  - n_m(t) is additive noise

Far-field plane-wave delays (relative delays sufficient for DOA):
    Δτ_m(u) = ((r_m − r_ref)^T u) / c                           (Eq. 2)

Doppler compensation via time warping:
Define warped time
    u_warp(t) = ∫_0^t α(ξ) dξ                                   (Eq. 3)
Resample each channel onto a uniform grid in u_warp:
    x̃_m(u_warp) = x_m( t(u_warp) )                              (Eq. 4)

If α(t) is correctly used, then approximately
    x̃_m(u_warp) ≈ s( u_warp − τ_m(u_true) ) + ñ_m(u_warp)       (Eq. 5)

SRP-PHAT:
For each mic pair (i,j), compute GCC-PHAT:
    R_ij^PHAT(τ) = F^{-1}{ (X_i X_j*) / (|X_i X_j*| + ε) }        (Eq. 6)
Then scan candidate directions u on a grid and score:
    P(u) = Σ_{i<j} R_ij^PHAT( Δτ_ij(u) )                          (Eq. 7)
where Δτ_ij(u)=Δτ_j(u)−Δτ_i(u). DOA is argmax_u P(u).

---------------------------------------------------------------------------
Array geometry (changed per your request)
---------------------------------------------------------------------------

Circular array, 7 microphones equally spaced on circumference.
Circle diameter = 10 cm -> radius = 5 cm = 0.05 m.

Mic positions in the x-y plane:
    r_m = [R cos(2π m/M), R sin(2π m/M), 0],  m=0..M-1.

---------------------------------------------------------------------------
What this simulation illustrates
---------------------------------------------------------------------------

We use a long frame duration (0.5 s) and a time-varying Doppler factor α(t),
which causes intra-frame time-warp in Eq. 1. That degrades SRP-PHAT.

We then apply the exact α(t) used in the simulation to warp time per Eq. 3–4,
and show SRP-PHAT becomes sharper / closer to the true DOA.

Requirements:
  - numpy
  - scipy
  - matplotlib

Run:
  python doppler_srp_phat_demo_circular7.py
"""

import numpy as np
from numpy.fft import rfft, irfft
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# -----------------------------
# Configuration
# -----------------------------
fs = 16000                 # sample rate [Hz]
T = 0.5                    # frame duration [s]
N = int(fs * T)
t = np.arange(N) / fs
c = 343.0                  # speed of sound [m/s]

# -----------------------------
# 7-mic circular array geometry
# -----------------------------
M = 7
diameter = 0.10            # 10 cm
R = diameter / 2.0         # radius = 5 cm = 0.05 m

angles = 2.0 * np.pi * np.arange(M) / M
r = np.stack([R*np.cos(angles), R*np.sin(angles), np.zeros(M)], axis=1)  # shape (M,3)

# Choose an arbitrary reference mic; with a symmetric ring, any is fine.
# (Eq. 2 uses differences w.r.t. this ref for convenience.)
ref = 0

# -----------------------------
# True DOA (2D, azimuth-only scan)
# -----------------------------
# u_true points toward the array from the source (plane-wave assumption)
az_true_deg = 35.0
az_true = np.deg2rad(az_true_deg)
u_true = np.array([np.cos(az_true), np.sin(az_true), 0.0])

# Relative delays from Eq. 2
delta_tau = ((r - r[ref]) @ u_true) / c  # shape (M,)

# -----------------------------
# Emitted drone-like signal s(·): harmonics + mild AM
# -----------------------------
f0 = 180.0  # harmonic spacing [Hz] (toy "BPF-like" spacing for demonstration)
harmonics = [1, 2, 3, 4, 5, 6]
amps = np.array([1.0, 0.6, 0.4, 0.3, 0.2, 0.15])

# Amplitude modulation ~3 Hz (toy beating)
am = 1.0 + 0.15 * np.sin(2*np.pi*3.0*t)

def s_emitted(u_time: np.ndarray) -> np.ndarray:
    """
    s(u): emitted signal as a function of an "emission-time-like" variable u.
    The received signal is evaluated at u = α(t) t − τ_m (Eq. 1).
    """
    sig = np.zeros_like(u_time, dtype=float)
    for k, a in zip(harmonics, amps):
        sig += a * np.sin(2*np.pi*(k*f0) * u_time)
    return sig

# -----------------------------
# Time-varying Doppler time-scaling factor α(t)
# -----------------------------
# Approaching/receding within the 0.5 s to force noticeable intra-frame time-warp.
# α(t) = c / (c - v_r(t)) where v_r(t) is radial velocity toward array.
v0 = 10.0      # [m/s]
a = 30.0       # [m/s^2] -> changes Doppler inside the frame
v_r = v0 + a * (t - T/2)          # ramp about mid-frame
v_r = np.clip(v_r, -20.0, 25.0)   # keep physical and away from c
alpha = c / (c - v_r)             # Eq. 1

# -----------------------------
# Simulate received signals (Eq. 1)
# -----------------------------
# x_m(t) = s( α(t) t − Δτ_m ) + n_m(t)
X = np.zeros((M, N), dtype=float)
for m in range(M):
    u_arg = alpha * t - delta_tau[m]  # Eq. 1 argument (Δτ_m used; absolute offset cancels)
    X[m] = am * s_emitted(u_arg)

# Add white noise (set SNR relative to reference mic)
snr_db = 5.0
signal_power = np.mean(X[ref]**2)
snr_lin = 10**(snr_db/10)
noise_power = signal_power / snr_lin
rng = np.random.default_rng(0)
X += rng.normal(scale=np.sqrt(noise_power), size=X.shape)

# Optional bandpass (typical for drone acoustics); helps PHAT stability
def bandpass(x, f_lo=200.0, f_hi=3000.0, fs=fs, order=4):
    b, a = butter(order, [f_lo/(fs/2), f_hi/(fs/2)], btype='band')
    return filtfilt(b, a, x)

for m in range(M):
    X[m] = bandpass(X[m])

# -----------------------------
# Doppler compensation via time warping (Eq. 3–4)
# -----------------------------
# Build u_warp(t) = ∫ α(ξ) dξ with discrete integration (Eq. 3)
u_warp = np.cumsum(alpha) / fs  # u_warp[n] ≈ Σ α[i] Ts

# Define uniform sampling grid in warped time (same nominal sample rate)
u_uniform = np.arange(N) / fs

# Invert mapping t(u_warp): since u_warp(t) is monotone, interpolate t as a function of u_warp
t_at_u = np.interp(u_uniform, u_warp, t, left=t[0], right=t[-1])

# Apply same warp to all microphones (Eq. 4)
Xc = np.zeros_like(X)
for m in range(M):
    Xc[m] = np.interp(t_at_u, t, X[m])

# -----------------------------
# SRP-PHAT implementation (Eq. 6–7), 2D azimuth scan
# -----------------------------
def gcc_phat(xi, xj, eps=1e-12):
    """
    GCC-PHAT (Eq. 6) computed over the whole frame with zero-padding.
    Returns:
      r: GCC-PHAT correlation samples (circular due to IFFT)
      nfft: FFT length used for modular lag indexing
    """
    n = len(xi)
    nfft = 2**int(np.ceil(np.log2(4*n)))  # generous zero padding
    Xi = rfft(xi, nfft)
    Xj = rfft(xj, nfft)
    G = Xi * np.conj(Xj)
    W = G / (np.abs(G) + eps)
    r = irfft(W, nfft)
    return r, nfft

def srp_phat(Xmulti, r_pos, az_grid_deg, c=343.0, fs=16000):
    """
    Compute SRP-PHAT spatial spectrum over an azimuth grid (Eq. 7).
    - Xmulti: (M, N)
    - r_pos: (M, 3)
    Returns (P, az_hat_deg).
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

    P = np.zeros(len(az_grid_deg), dtype=float)

    for a_idx, az_deg in enumerate(az_grid_deg):
        az = np.deg2rad(az_deg)
        u_dir = np.array([np.cos(az), np.sin(az), 0.0])

        # Predicted (up to constant) propagation times: τ_m(u) ∝ (r_m^T u)/c
        # Pairwise differences implement Eq. 2 inside Eq. 7.
        tau = (r_pos @ u_dir) / c

        score = 0.0
        for k, (i, j) in enumerate(pairs):
            dtau = tau[j] - tau[i]                   # Δτ_ij(u)
            lag_samp = int(np.round(dtau * fs))       # predicted lag (samples)

            # Sample GCC at this lag using modular indexing (circular correlation)
            nfft = nffts[k]
            idx = lag_samp % nfft
            score += gcc[k][idx]

        P[a_idx] = score

    az_hat = az_grid_deg[int(np.argmax(P))]
    return P, az_hat

# Azimuth scan grid
az_grid = np.linspace(-90, 90, 721)  # 0.25° resolution

# SRP-PHAT without and with Doppler time-warp compensation
P_raw, az_hat_raw = srp_phat(X,  r, az_grid, c=c, fs=fs)
P_cmp, az_hat_cmp = srp_phat(Xc, r, az_grid, c=c, fs=fs)

# -----------------------------
# Plot results
# -----------------------------
plt.figure()
plt.plot(az_grid, P_raw)
plt.axvline(az_true_deg, linestyle='--', label='True DOA')
plt.axvline(az_hat_raw, linestyle=':', label='Estimated (no comp)')
plt.title("SRP-PHAT spatial spectrum (NO Doppler compensation) — 7-mic circular array")
plt.xlabel("Azimuth (deg)")
plt.ylabel("SRP-PHAT score")
plt.legend()

plt.figure()
plt.plot(az_grid, P_cmp)
plt.axvline(az_true_deg, linestyle='--', label='True DOA')
plt.axvline(az_hat_cmp, linestyle=':', label='Estimated (with comp)')
plt.title("SRP-PHAT spatial spectrum (WITH Doppler time-scaling compensation) — 7-mic circular array")
plt.xlabel("Azimuth (deg)")
plt.ylabel("SRP-PHAT score")
plt.legend()

plt.figure()
plt.plot(t, alpha)
plt.title("Time-varying Doppler time-scaling factor α(t)")
plt.xlabel("Time (s)")
plt.ylabel("α(t)")

print("Array geometry: 7-mic circular ring, diameter = 10 cm (R = 0.05 m)")
print("True azimuth (deg):", az_true_deg)
print("Estimated azimuth WITHOUT compensation (deg):", float(az_hat_raw))
print("Estimated azimuth WITH compensation    (deg):", float(az_hat_cmp))

plt.show()
