import sounddevice as sd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from scipy.signal import find_peaks
import queue

# Constants for DOA calculation
SPEED_OF_SOUND = 343.0  # m/s at room temperature
MIC_SPACING = 0.042  # meters (spacing between mics in the 4x4 array)

# Audio settings
SAMPLE_RATE = 48000  # Sample rate for 16-channel array
CHANNELS = 16  # 16 microphones in 4x4 array
BUFFER_DURATION = 0.5  # seconds of audio to process

# Multi-source detection settings
MAX_SOURCES = 8  # Maximum number of sources (channels / 2)
MIN_SEPARATION = 15  # Minimum angular separation between sources (degrees)
AZIMUTH_RESOLUTION = 2  # Degrees (scanning resolution for verification)
MIN_PEAK_HEIGHT = 0.3  # Minimum relative power for peak detection (0-1)
MIN_PROMINENCE = 0.15  # How much peak must stand out from background
SOUND_THRESHOLD = 0.002  # RMS threshold for sound detection

# ESPRIT algorithm settings
NUM_SOURCES_ESPRIT = 3  # Assumed number of sources for subspace estimation
SMOOTHING_FACTOR = 0.99  # Exponential smoothing for covariance matrix
CENTER_FREQ = 1000  # Hz - center frequency for narrowband assumption
USE_TLS = True  # Use Total Least Squares (TLS-ESPRIT) vs standard LS-ESPRIT

# Queue for audio data (non-blocking)
audio_queue = queue.Queue(maxsize=10)

# Audio buffer to accumulate samples
audio_buffer = []

# Latest detection results
latest_sources = []
latest_spectrum = None
sound_active = False

# Covariance matrix for spatial smoothing
covariance_matrix = None


def get_mic_positions_4x4():
    """
    Generate microphone positions for 4x4 array with origin at CENTER of array.

    Corrected geometry:
    Row 3 (North, Y=+63mm): MIC09, MIC08, MIC11, MIC10 → channels [8, 7, 10, 9]
    Row 2 (Y=+21mm): MIC07, MIC06, MIC13, MIC12 → channels [6, 5, 12, 11]
    Row 1 (Y=-21mm): MIC05, MIC04, MIC15, MIC14 → channels [4, 3, 14, 13]
    Row 0 (South, Y=-63mm): MIC03, MIC02, MIC01, MIC16 → channels [2, 1, 0, 15]

    X positions (West to East): -63mm, -21mm, +21mm, +63mm

    Returns:
        Dictionary mapping channel numbers to (x, y) positions in meters
    """
    spacing = MIC_SPACING

    # X positions (West to East)
    x_positions = [-1.5 * spacing, -0.5 * spacing, 0.5 * spacing, 1.5 * spacing]

    # Y positions (South to North)
    y_positions = [-1.5 * spacing, -0.5 * spacing, 0.5 * spacing, 1.5 * spacing]

    # Build position dictionary for each channel
    mic_positions = {}

    # Row 0 (South, Y=-63mm): channels [2, 1, 0, 15]
    mic_positions[2] = (x_positions[0], y_positions[0])  # MIC03
    mic_positions[1] = (x_positions[1], y_positions[0])  # MIC02
    mic_positions[0] = (x_positions[2], y_positions[0])  # MIC01
    mic_positions[15] = (x_positions[3], y_positions[0])  # MIC16

    # Row 1 (Y=-21mm): channels [4, 3, 14, 13]
    mic_positions[4] = (x_positions[0], y_positions[1])  # MIC05
    mic_positions[3] = (x_positions[1], y_positions[1])  # MIC04
    mic_positions[14] = (x_positions[2], y_positions[1])  # MIC15
    mic_positions[13] = (x_positions[3], y_positions[1])  # MIC14

    # Row 2 (Y=+21mm): channels [6, 5, 12, 11]
    mic_positions[6] = (x_positions[0], y_positions[2])  # MIC07
    mic_positions[5] = (x_positions[1], y_positions[2])  # MIC06
    mic_positions[12] = (x_positions[2], y_positions[2])  # MIC13
    mic_positions[11] = (x_positions[3], y_positions[2])  # MIC12

    # Row 3 (North, Y=+63mm): channels [8, 7, 10, 9]
    mic_positions[8] = (x_positions[0], y_positions[3])  # MIC09
    mic_positions[7] = (x_positions[1], y_positions[3])  # MIC08
    mic_positions[10] = (x_positions[2], y_positions[3])  # MIC11
    mic_positions[9] = (x_positions[3], y_positions[3])  # MIC10

    return mic_positions


def get_doublet_subarrays():
    """
    Define doublet subarrays for ESPRIT.

    ESPRIT requires two identical subarrays with a displacement.
    We'll use row-based subarrays (displaced vertically) for azimuth estimation.

    Returns:
        subarray1: List of channels in first subarray
        subarray2: List of channels in second subarray (displaced)
        displacement: Displacement vector (dx, dy) in meters
    """
    # Use rows 0-2 as subarray 1, rows 1-3 as subarray 2
    # This gives vertical displacement

    # Subarray 1: Rows 0, 1, 2
    subarray1 = [2, 1, 0, 15,    # Row 0
                 4, 3, 14, 13,    # Row 1
                 6, 5, 12, 11]    # Row 2

    # Subarray 2: Rows 1, 2, 3 (displaced by +MIC_SPACING in Y)
    subarray2 = [4, 3, 14, 13,    # Row 1
                 6, 5, 12, 11,    # Row 2
                 8, 7, 10, 9]     # Row 3

    # Displacement: one row up (+Y direction)
    displacement = (0.0, MIC_SPACING)

    return subarray1, subarray2, displacement


def detect_sound(signal, threshold=SOUND_THRESHOLD):
    """
    Detect if there's a sound event in the signal.

    Args:
        signal: Audio signal
        threshold: RMS threshold for sound detection

    Returns:
        bool: True if sound detected, False otherwise
    """
    rms = np.sqrt(np.mean(signal**2))
    return rms > threshold


def compute_rotation_matrix(subarray1, subarray2, displacement, freq):
    """
    Compute the theoretical rotation matrix Phi for ESPRIT.

    The rotation matrix relates the array manifolds of the two subarrays:
    A2 = A1 * Phi, where Phi = diag(exp(j*2*pi*f*tau_i))

    Args:
        subarray1: Channels in first subarray
        subarray2: Channels in second subarray
        displacement: Displacement vector (dx, dy)
        freq: Signal frequency

    Returns:
        None (rotation is signal-dependent, computed from eigenvalues)
    """
    # Note: In practice, we estimate Phi from the signal subspace
    # This function is for reference/documentation
    pass


def detect_multiple_sources_esprit(audio_data, num_sources_estimate=NUM_SOURCES_ESPRIT,
                                   use_tls=USE_TLS):
    """
    Detect multiple sound sources using ESPRIT algorithm.

    ESPRIT (Estimation of Signal Parameters via Rotational Invariance Techniques)
    exploits the rotational invariance structure in doublet subarrays to estimate DOAs
    without scanning the angular space.

    Args:
        audio_data: Audio data array [samples, channels]
        num_sources_estimate: Estimated number of sources
        use_tls: Use Total Least Squares (TLS-ESPRIT) vs LS-ESPRIT

    Returns:
        sources: List of dictionaries with 'azimuth', 'power', 'confidence', 'id'
        spectrum_data: Tuple of (azimuth_scan, esprit_spectrum) for visualization
        sound_detected: Boolean indicating if sound was detected
    """
    global covariance_matrix

    # Check if sound is detected
    avg_signal = np.mean(audio_data, axis=1)
    sound_detected = detect_sound(avg_signal, threshold=SOUND_THRESHOLD)

    if not sound_detected:
        return [], None, False

    # Get doublet subarrays
    subarray1, subarray2, displacement = get_doublet_subarrays()
    num_sensors = len(subarray1)

    # Extract subarray data
    X1 = audio_data[:, subarray1].T  # Shape: [num_sensors, num_samples]
    X2 = audio_data[:, subarray2].T  # Shape: [num_sensors, num_samples]

    # Compute spatial covariance matrices
    R1 = np.dot(X1, np.conj(X1).T) / len(audio_data)
    R2 = np.dot(X2, np.conj(X2).T) / len(audio_data)

    # Combined covariance for better estimation
    R = (R1 + R2) / 2.0

    # Apply exponential smoothing
    if covariance_matrix is not None:
        R = SMOOTHING_FACTOR * covariance_matrix + (1 - SMOOTHING_FACTOR) * R
    covariance_matrix = R

    # Eigenvalue decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(R)

    # Sort eigenvalues in descending order
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Signal subspace (first num_sources_estimate eigenvectors)
    Es = eigenvectors[:, :num_sources_estimate]

    # Split signal subspace into two parts corresponding to subarrays
    Es1 = Es  # For subarray 1
    Es2 = Es  # For subarray 2 (same, but represents displaced array)

    # Estimate rotation operator Psi
    # Psi relates the signal subspaces: Es2 ≈ Es1 * Psi

    if use_tls:
        # TLS-ESPRIT: Use SVD of combined matrix
        # C = [Es1^T; Es2^T] (concatenate transposes)
        C = np.hstack([Es1.T, Es2.T])

        # Perform SVD with full matrices to get proper dimensions
        U, S, Vh = np.linalg.svd(C, full_matrices=True)

        # Extract Psi from the right singular vectors
        # V is [2*num_sources_estimate, 2*num_sources_estimate]
        V = Vh.T.conj()
        n = num_sources_estimate

        # Check if we have enough dimensions
        if V.shape[0] >= 2*n and V.shape[1] >= 2*n:
            # Partition V into blocks
            V12 = V[:n, n:2*n]
            V22 = V[n:2*n, n:2*n]

            # Avoid singular matrix
            try:
                if V22.size > 0 and np.linalg.matrix_rank(V22) == V22.shape[0]:
                    Psi = -V12 @ np.linalg.inv(V22)
                else:
                    # Fallback to LS if TLS fails
                    Psi = np.linalg.lstsq(Es1, Es2, rcond=None)[0]
            except:
                # Fallback to LS if anything fails
                Psi = np.linalg.lstsq(Es1, Es2, rcond=None)[0]
        else:
            # Fallback to LS if dimensions don't match
            Psi = np.linalg.lstsq(Es1, Es2, rcond=None)[0]
    else:
        # LS-ESPRIT: Simple least squares
        Psi = np.linalg.lstsq(Es1, Es2, rcond=None)[0]

    # Eigenvalues of Psi give the phase shifts
    phi_eigenvalues = np.linalg.eigvals(Psi)

    # Convert phase shifts to DOAs
    # Phase shift: phi = exp(j * 2 * pi * f * tau)
    # where tau = displacement · source_direction / c

    # Extract phases
    phases = np.angle(phi_eigenvalues)

    # Compute azimuths from phases
    # For vertical displacement (0, dy):
    # tau = dy * sin(elevation) * cos(azimuth) / c
    # For horizontal plane (elevation=0): tau ≈ 0 (ambiguous)

    # For horizontal displacement analysis, we need different subarrays
    # Let's use the phase information to estimate directions

    # Wavelength at center frequency
    wavelength = SPEED_OF_SOUND / CENTER_FREQ
    k = 2 * np.pi / wavelength  # Wave number

    # Displacement magnitude
    d = np.linalg.norm(displacement)

    azimuths_esprit = []
    confidences = []

    for phase in phases:
        # Phase = k * d * cos(theta)
        # where theta is angle between displacement and source direction

        # For vertical displacement and horizontal sources:
        # This gives us information about azimuth indirectly

        # Compute angle from phase
        cos_theta = phase / (k * d)

        # Handle numerical errors
        if np.abs(cos_theta) <= 1.0:
            theta = np.arccos(np.real(cos_theta))

            # Convert to azimuth (depends on displacement direction)
            # If displacement is in Y direction (North):
            # theta relates to source azimuth
            azimuth_rad = theta
            azimuth_deg = np.degrees(azimuth_rad)

            # Adjust to compass convention (0° = North)
            azimuth_deg = (90 - azimuth_deg) % 360

            azimuths_esprit.append(azimuth_deg)

            # Confidence based on eigenvalue magnitude
            confidence = np.abs(phi_eigenvalues[phases == phase][0])
            confidences.append(confidence)

    # If no valid sources found, return empty
    if len(azimuths_esprit) == 0:
        return [], None, False

    # Normalize confidences
    if len(confidences) > 0:
        max_conf = max(confidences)
        if max_conf > 0:
            confidences = [c / max_conf for c in confidences]

    # Create ESPRIT pseudo-spectrum for visualization
    azimuth_scan = np.arange(0, 360, AZIMUTH_RESOLUTION)
    esprit_spectrum = np.zeros(len(azimuth_scan))

    # Add Gaussian peaks at estimated azimuths
    for az, conf in zip(azimuths_esprit, confidences):
        for idx, scan_az in enumerate(azimuth_scan):
            # Gaussian peak centered at detected azimuth
            sigma = 5.0  # degrees
            esprit_spectrum[idx] += conf * np.exp(-0.5 * ((scan_az - az) / sigma) ** 2)

    # Normalize spectrum
    if np.max(esprit_spectrum) > 0:
        esprit_spectrum = esprit_spectrum / np.max(esprit_spectrum)

    # Find peaks in spectrum (for consistency with other methods)
    peaks, _ = find_peaks(
        esprit_spectrum,
        height=MIN_PEAK_HEIGHT,
        distance=int(MIN_SEPARATION / AZIMUTH_RESOLUTION),
        prominence=MIN_PROMINENCE
    )

    # Create source list
    sources = []
    if len(peaks) > 0:
        peak_powers = esprit_spectrum[peaks]
        sorted_indices = np.argsort(peak_powers)[::-1]
        peaks = peaks[sorted_indices]
        peak_powers = peak_powers[sorted_indices]

        # Limit to MAX_SOURCES
        peaks = peaks[:MAX_SOURCES]
        peak_powers = peak_powers[:MAX_SOURCES]

        for peak_idx, power in zip(peaks, peak_powers):
            azimuth = azimuth_scan[peak_idx]

            sources.append({
                'azimuth': float(azimuth),
                'elevation': 0.0,  # Horizontal plane only
                'power': float(power),
                'confidence': float(power),
                'id': len(sources)
            })

    return sources, (azimuth_scan, esprit_spectrum), sound_detected


def audio_callback(indata, frames, time, status):
    """
    Callback function for sounddevice stream.
    """
    if status:
        print(f"Status: {status}")

    try:
        audio_queue.put_nowait(indata.copy())
    except queue.Full:
        pass


def update_plot(frame):
    """
    Update function for matplotlib animation.
    """
    global latest_sources, latest_spectrum, sound_active, audio_buffer

    # Get all available audio chunks from queue
    try:
        while True:
            chunk = audio_queue.get_nowait()
            audio_buffer.append(chunk)
    except queue.Empty:
        pass

    # Keep buffer size manageable
    max_buffer_samples = int(SAMPLE_RATE * 1.0)
    if len(audio_buffer) > 0:
        total_samples = sum(len(chunk) for chunk in audio_buffer)

        while total_samples > max_buffer_samples and len(audio_buffer) > 1:
            removed = audio_buffer.pop(0)
            total_samples -= len(removed)

    # Process if we have enough samples
    if len(audio_buffer) > 0:
        audio_data = np.vstack(audio_buffer)

        # Only process if we have enough samples
        if len(audio_data) >= int(BUFFER_DURATION * SAMPLE_RATE):
            # Take last BUFFER_DURATION worth of samples
            audio_data = audio_data[-int(BUFFER_DURATION * SAMPLE_RATE):]

            # Remove DC offset from each channel
            audio_data = audio_data - np.mean(audio_data, axis=0)

            # Normalize
            audio_data = audio_data / (np.max(np.abs(audio_data)) + 1e-10)

            # Detect multiple sources using ESPRIT
            sources, spectrum_data, sound_detected = detect_multiple_sources_esprit(
                audio_data,
                num_sources_estimate=NUM_SOURCES_ESPRIT,
                use_tls=USE_TLS
            )

            if sound_detected and len(sources) > 0:
                latest_sources = sources
                latest_spectrum = spectrum_data
                sound_active = True

                # Print detected sources
                if frame % 5 == 0:  # Print every 5 frames
                    sources_info = ", ".join([f"S{i+1}: Az={s['azimuth']:.1f}° (P:{s['power']:.2f})"
                                             for i, s in enumerate(sources)])
                    print(f">>> Detected {len(sources)} source(s): {sources_info}")
            else:
                sound_active = sound_detected
                if sound_detected and len(sources) == 0:
                    if frame % 10 == 0:
                        print(">>> Sound detected but no clear sources identified")

    # Clear previous plots
    ax1.clear()
    ax2.clear()

    # ===== Plot 1: Polar plot for DOA =====
    ax1.set_theta_zero_location('N')
    ax1.set_theta_direction(-1)
    ax1.set_title('Multi-Source DOA\n(ESPRIT Algorithm)', fontsize=12, weight='bold')

    if sound_active and len(latest_sources) > 0:
        # Define colors for different sources
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'yellow', 'magenta']

        # Plot each detected source
        for idx, source in enumerate(latest_sources):
            azimuth_rad = np.radians(source['azimuth'])
            color = colors[idx % len(colors)]

            # Arrow length based on confidence
            arrow_length = 0.6 + 0.3 * source['confidence']

            ax1.arrow(azimuth_rad, 0, 0, arrow_length,
                     head_width=0.25, head_length=0.1,
                     fc=color, ec=color, linewidth=2.5, alpha=0.8)

            # Label with source number and azimuth
            ax1.text(azimuth_rad, arrow_length + 0.15,
                    f"S{idx+1}\n{source['azimuth']:.1f}°",
                    ha='center', va='bottom', fontsize=9, color=color, weight='bold')

        # Add grid
        ax1.set_ylim(0, 1)
        ax1.grid(True, alpha=0.3)

    else:
        ax1.text(0, 0, 'No sound\ndetected',
                ha='center', va='center', fontsize=12, color='gray')
        ax1.set_ylim(0, 1)

    # ===== Plot 2: 3D Representation =====
    # Always plot microphone array in XY plane
    mic_positions = get_mic_positions_4x4()
    for ch, pos in mic_positions.items():
        ax2.scatter(pos[0], pos[1], 0, c='blue', s=30, alpha=0.5)

    if sound_active and len(latest_sources) > 0:
        # Define colors for different sources
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'yellow', 'magenta']

        # Plot each detected source in horizontal plane (Z=0)
        for idx, source in enumerate(latest_sources):
            azimuth_rad = np.radians(source['azimuth'])
            color = colors[idx % len(colors)]

            # Unit vector in horizontal plane (elevation = 0)
            # Azimuth: 0° = North (+Y), 90° = East (+X)
            x = np.sin(azimuth_rad)
            y = np.cos(azimuth_rad)
            z = 0

            # Scale arrow based on confidence
            scale = 0.08 + 0.04 * source['confidence']

            # Draw arrow from origin to direction
            ax2.quiver(0, 0, 0, x*scale, y*scale, z,
                      color=color, arrow_length_ratio=0.3, linewidth=2.5, alpha=0.8)

        ax2.set_xlabel('X (East) [m]', fontsize=9)
        ax2.set_ylabel('Y (North) [m]', fontsize=9)
        ax2.set_zlabel('Z (Up) [m]', fontsize=9)
        ax2.set_title(f'3D DOA - {len(latest_sources)} Source(s)\n(Horizontal Plane)',
                     fontsize=11, weight='bold')

        ax2.set_xlim(-0.1, 0.1)
        ax2.set_ylim(-0.1, 0.1)
        ax2.set_zlim(0, 0.15)

    else:
        ax2.text(0, 0, 0.05, 'No sound detected',
                ha='center', va='center', fontsize=12, color='gray')
        ax2.set_xlabel('X (East) [m]', fontsize=9)
        ax2.set_ylabel('Y (North) [m]', fontsize=9)
        ax2.set_zlabel('Z (Up) [m]', fontsize=9)
        ax2.set_xlim(-0.1, 0.1)
        ax2.set_ylim(-0.1, 0.1)
        ax2.set_zlim(0, 0.15)


if __name__ == "__main__":
    print("=" * 80)
    print("MULTI-SOURCE ESPRIT ALGORITHM - 4x4 MICROPHONE ARRAY")
    print("=" * 80)
    print(f"Array configuration: 16 microphones in 4x4 grid")
    print(f"Microphone spacing: {MIC_SPACING*1000:.1f} mm")
    print(f"Origin: CENTER of array (0,0)")
    print(f"Detection method: {'TLS-' if USE_TLS else 'LS-'}ESPRIT")
    print(f"Maximum sources detected: {MAX_SOURCES} (channels / 2)")
    print(f"Signal subspace dimension: {NUM_SOURCES_ESPRIT}")

    subarray1, subarray2, displacement = get_doublet_subarrays()
    print(f"Subarray size: {len(subarray1)} sensors each")
    print(f"Displacement: ({displacement[0]*1000:.1f}, {displacement[1]*1000:.1f}) mm")

    print(f"Center frequency: {CENTER_FREQ} Hz")
    print(f"Covariance smoothing: {SMOOTHING_FACTOR}")
    print(f"Minimum separation: {MIN_SEPARATION}°")
    print(f"Sound threshold: {SOUND_THRESHOLD}")
    print("Note: All sources detected in horizontal plane (elevation = 0°)")
    print("=" * 80)
    print()

    # Find miniDSP device
    devices = sd.query_devices()
    minidsp_device = None

    for i, device in enumerate(devices):
        if 'line' in device['name'].lower():
            minidsp_device = i
            print(f"Found miniDSP device at index {i}")
            print(f"  Name: {device['name']}")
            print(f"  Max input channels: {device['max_input_channels']}")
            print(f"  Default sample rate: {device['default_samplerate']}")
            break

    if minidsp_device is None:
        print("Error: miniDSP device not found!")
        print("\nAvailable devices:")
        for i, device in enumerate(devices):
            print(f"  [{i}] {device['name']}")
        exit(1)

    print(f"\nStarting real-time ESPRIT-based multi-source detection...")
    print(f"Sample rate: {SAMPLE_RATE} Hz")
    print(f"Channels: {CHANNELS}")
    print(f"Buffer duration: {BUFFER_DURATION} seconds")

    # Start audio stream
    stream = sd.InputStream(
        device=minidsp_device,
        channels=CHANNELS,
        samplerate=SAMPLE_RATE,
        callback=audio_callback,
        blocksize=int(SAMPLE_RATE * 0.1)
    )

    stream.start()

    # Set up matplotlib figure
    fig = plt.figure(figsize=(16, 8))
    ax1 = fig.add_subplot(121, projection='polar')
    ax2 = fig.add_subplot(122, projection='3d')

    # Create animation
    ani = animation.FuncAnimation(
        fig, update_plot, interval=200, cache_frame_data=False, blit=False
    )

    plt.tight_layout()

    try:
        print("\nReal-time processing active. Close the plot window to stop.")
        print("=" * 70)
        plt.show()
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        stream.stop()
        stream.close()
        print("Stream closed.")
