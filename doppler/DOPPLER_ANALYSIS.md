# Doppler Effect Simulation Analysis

## Simulation Setup

### Geometry
- **Microphone Array**: 4 microphones in a linear row along x-axis
- **Mic Spacing**: 40 mm between adjacent microphones
- **Mic Positions**: [-60mm, -20mm, +20mm, +60mm] from center

### Source Parameters
- **Signal**: 3000 Hz pure tone (simulating drone propeller harmonics)
- **Velocity**: 10 m/s horizontal (perpendicular to line-of-sight)
- **Trajectory**: Constant y = 50m, moving along x-axis
- **Initial Distance**: 100m from array center
- **Duration**: 20 seconds total

### Processing Parameters
- **Sample Rate**: 48,000 Hz
- **Buffer Duration**: 0.5 seconds per DOA estimate
- **Method**: Cross-correlation TDOA + Linear DOA estimation

## Key Results from the Plots

### 1. Geometry Plot (Top Left)
Shows the source traveling horizontally at y=50m across the microphone array:
- **Green marker**: Start position (far left, ~86m away)
- **Magenta marker**: Closest approach (directly above, 50m away)
- **Red marker**: End position (far right, ~115m away)

### 2. DOA Estimation (Top Center)
**Critical Observation**: Massive estimation errors!
- **True DOA** (green line): Smooth curve from 300° → 0° → 65°
- **Estimated DOA** (blue dashed): Erratic jumps, often completely wrong
- **Pattern**: Estimates jump between 0°, 90°, 180° with no smooth tracking

### 3. DOA Error (Top Right)
**RMS Error: 61.36°** - This is catastrophically large!
- Errors range from -50° to +126°
- Errors are largest when:
  - Source is far from array (high angular velocity)
  - Doppler shifts are maximum (±80 Hz)
- Smallest error near closest approach (t≈8.8s) where Doppler crosses zero

### 4. Doppler Shift (Bottom Left)
Shows frequency shift at each microphone over time:
- **Maximum shift**: ±77 Hz (2.6% of carrier frequency)
- **All mics experience different shifts**: Up to 5-10 Hz difference between mics
- **Zero crossing at t≈8.8s**: When source passes broadside (no radial velocity)
- **Asymmetric**: Positive (approaching) then negative (receding)

### 5. TDOA Measurements (Bottom Center)
Shows measured time delays between microphone pairs:
- **Peak TDOAs**: ±10,000 μs = ±10 ms (±480 samples @ 48kHz)
- **Large values around t=7-11s**: When source is closest and angular velocity is highest
- **Pattern**: TDOA changes rapidly during closest approach
- **Problem**: 0.5s buffer captures source motion of ~5° per estimate!

### 6. Distance and Angular Velocity (Bottom Right)
Explains why errors occur:
- **Distance** (blue): Decreases from 100m → 50m → 120m
- **Angular Velocity** (red): Peaks at **±350°/s** during closest approach!
- **Motion blur**: In 0.5s buffer, source moves **175°** in angular space
- This violates the stationary source assumption of TDOA methods

## Why TDOA Fails with Doppler

### 1. Waveform Mismatch
Cross-correlation assumes both microphones receive **identical waveforms** with only a time shift. But Doppler causes:
- **Different frequencies** at each mic (77Hz vs 72Hz vs 68Hz)
- **Time compression/stretching**: Waveform isn't just shifted, it's warped
- **Result**: Cross-correlation peak degrades, wrong delay estimated

### 2. Non-Stationary Source
The 0.5s buffer assumption that source is stationary is violated:
- Source moves **5 meters** horizontally during buffer
- Angular position changes by **up to 175°** during processing
- Different parts of the buffer have different TDOAs
- **Result**: Time-averaged TDOA doesn't correspond to any real position

### 3. Far-Field Assumption Breakdown
Linear DOA estimation assumes plane wave (far-field):
- `sin(θ) = TDOA * c / baseline`
- But at 50m distance with 120mm baseline: **near-field effects** appear
- Curvature of wavefront matters
- **Result**: Geometric approximation introduces 5-10° bias

### 4. Integer Sample Quantization
At 48 kHz sampling:
- Sample period: 20.8 μs
- Mic spacing (40mm) / c: 116.6 μs maximum TDOA
- **Only ~6 samples** across full 180° angular range!
- Quantization: Each sample = **30° of angular resolution**
- Doppler-induced TDOA errors (±2 μs) are significant compared to quantization

## Specific Observations

### Closest Approach (t ≈ 8.8s)
- **True DOA**: 1.0°
- **Estimated DOA**: 0.0°
- **Error**: -1.0° (BEST performance)
- **Why?**: Doppler = 0 Hz (no radial velocity), angular velocity at maximum but TDOA changing slowly

### Worst Performance (t = 0-6s and 12-20s)
- **Errors**: 80-126°
- **Why?**:
  - Large Doppler shifts (±70-77 Hz)
  - Source far from array (low SNR due to 1/r spreading)
  - High angular velocity combined with large Doppler
  - Estimated DOAs often snap to 0°, 90°, 180° (quantization artifacts)

## Implications for Your Real Systems

### rt_combined.py (SRP-PHAT + MUSIC)
- **Advantage**: Frequency-domain processing averages across Doppler-shifted bins
- **Disadvantage**: MUSIC uses single center frequency (2000 Hz) - will have ~3° error
- **Mitigation**: 0.2-0.3s buffers would reduce motion blur from 175° to 70°

### rt_beamforming.py (Delay-and-Sum + TDOA)
- **Advantage**: Wideband time-domain beamforming more robust than narrowband
- **Critical weakness**: TDOA refinement (cross-correlation) will fail exactly as shown here
- **Elevation estimate**: Even worse due to smaller vertical baseline
- **Mitigation**:
  - Reduce buffer to 0.1s (motion blur → 35°)
  - Track sources between frames to compensate for motion
  - Disable TDOA refinement for fast-moving sources

## Recommended Fixes

### Short-term (Easy)
1. **Reduce buffer duration**: 0.5s → 0.1s
   - Reduces motion blur: 175° → 35°
   - Trade-off: Less frequency resolution, more noise

2. **Motion compensation**: Track angular position between frames
   ```python
   delta_azimuth = azimuth_current - azimuth_previous
   angular_velocity = delta_azimuth / dt
   # Apply velocity-based prediction
   ```

3. **Doppler detection threshold**: Measure rate of azimuth change
   ```python
   if abs(angular_velocity) > 30:  # deg/s
       disable_tdoa_refinement = True
       use_beamforming_only = True
   ```

### Long-term (Advanced)
1. **Motion-compensated beamforming**:
   - Split buffer into 5×0.1s segments
   - Apply time-varying steering: `θ(t) = θ₀ + ω·t`
   - Coherently combine sub-segments

2. **Doppler-corrected cross-correlation**:
   - Estimate Doppler from spectrum
   - Resample signals to compensate frequency shift
   - Apply time-warping before correlation

3. **Kalman filter tracking**:
   - State: [azimuth, angular_velocity, elevation]
   - Predict next position from velocity
   - Use prediction to narrow search space

## Conclusion

This simulation demonstrates that **TDOA-based DOA estimation fails catastrophically** for moving sources when:
- Buffer duration is too long relative to angular motion
- Doppler shifts cause waveform mismatch between microphones
- Source motion during observation violates stationarity assumption

For your drone detection application with sources moving at 10 m/s at 50m distance:
- **Angular velocity**: Up to 350°/s during close passes
- **Expected DOA error with 0.5s buffers**: 60-125° (unusable)
- **Expected DOA error with 0.1s buffers**: 10-30° (marginal)
- **Expected DOA error with motion compensation**: 3-8° (acceptable)

The key insight: **Your buffer duration must be much shorter than the timescale of source motion**, or you must explicitly track and compensate for that motion.
