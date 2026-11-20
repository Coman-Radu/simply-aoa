import sounddevice as sd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from scipy.signal import correlate, find_peaks
import queue

# Constants for DOA calculation
SPEED_OF_SOUND = 343.0  # m/s at room temperature
MIC_SPACING = 0.042  # meters (spacing between mics in the 4x4 array)

# Audio settings
SAMPLE_RATE = 48000  # Sample rate for 16-channel array
CHANNELS = 16  # 16 microphones in 4x4 array
BUFFER_DURATION = 0.5  # seconds of audio to process

# Multi-source detection settings
MAX_SOURCES = None  # Maximum number of sources (None = dynamic, detect all valid peaks)
MIN_SEPARATION = 15  # Minimum angular separation between sources (degrees)
AZIMUTH_RESOLUTION = 2  # Degrees (scanning resolution)
MIN_PEAK_HEIGHT = 0.3  # Minimum relative power for peak detection (0-1)
MIN_PROMINENCE = 0.15  # How much peak must stand out from background
SOUND_THRESHOLD = 0.002  # RMS threshold for sound detection (absolute RMS amplitude)

# Queue for audio data (non-blocking)
audio_queue = queue.Queue(maxsize=10)


# Audio buffer to accumulate samples
audio_buffer = []

# Latest detection results
latest_sources = []
latest_spectrum = None
sound_active = False


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


def get_row_layout():
    """
    Define the 4 rows of the 4x4 array for azimuth estimation using TDOA.

    Rows from South to North:
    Row 0 (South): [2, 1, 0, 15] (MIC03, MIC02, MIC01, MIC16)
    Row 1: [4, 3, 14, 13] (MIC05, MIC04, MIC15, MIC14)
    Row 2: [6, 5, 12, 11] (MIC07, MIC06, MIC13, MIC12)
    Row 3 (North): [8, 7, 10, 9] (MIC09, MIC08, MIC11, MIC10)

    Returns:
        List of tuples (row_channels, reference_channel)
    """
    rows = [
        ([2, 1, 0, 15], 1),      # Row 0, ref: MIC02 (channel 1, center-left)
        ([4, 3, 14, 13], 3),     # Row 1, ref: MIC04 (channel 3, center-left)
        ([6, 5, 12, 11], 5),     # Row 2, ref: MIC06 (channel 5, center-left)
        ([8, 7, 10, 9], 7)       # Row 3, ref: MIC08 (channel 7, center-left)
    ]
    return rows


def get_column_layout():
    """
    Define the 4 columns of the 4x4 array for elevation estimation using TDOA.

    Columns from West to East (each goes South to North):
    Column 0 (West): [2, 4, 6, 8] (MIC03, MIC05, MIC07, MIC09) (ref: channel 2)
    Column 1: [1, 3, 5, 7] (MIC02, MIC04, MIC06, MIC08) (ref: channel 1)
    Column 2: [0, 14, 12, 10] (MIC01, MIC15, MIC13, MIC11) (ref: channel 0)
    Column 3 (East): [15, 13, 11, 9] (MIC16, MIC14, MIC12, MIC10) (ref: channel 15)

    Returns:
        List of tuples (column_channels, reference_channel)
    """
    columns = [
        ([2, 4, 6, 8], 2),       # Column 0 (West), ref: MIC03 (channel 2, South)
        ([1, 3, 5, 7], 1),       # Column 1, ref: MIC02 (channel 1, South)
        ([0, 14, 12, 10], 0),    # Column 2, ref: MIC01 (channel 0, South)
        ([15, 13, 11, 9], 15)    # Column 3 (East), ref: MIC16 (channel 15, South)
    ]
    return columns


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


def detect_multiple_sources_beamforming(audio_data, num_sources=MAX_SOURCES,
                                       min_separation=MIN_SEPARATION,
                                       azimuth_resolution=AZIMUTH_RESOLUTION):
    """
    Detect multiple sound sources using delay-and-sum beamforming and peak detection.

    This function scans all azimuth directions, applies beamforming to enhance signals
    from each direction, and detects multiple sources by finding peaks in the
    resulting power spectrum.

    Args:
        audio_data: Audio data array [samples, channels]
        num_sources: Maximum number of sources to detect (None = dynamic, detect all valid peaks)
        min_separation: Minimum angular separation between sources (degrees)
        azimuth_resolution: Angular resolution for scanning (degrees)

    Returns:
        sources: List of dictionaries with 'azimuth', 'power', 'confidence', 'id'
        spectrum_data: Tuple of (azimuth_scan, power_spectrum) for visualization
        sound_detected: Boolean indicating if sound was detected
    """
    # Normalize signals
    audio_data = audio_data / (np.max(np.abs(audio_data)) + 1e-10)

    # Check if sound is detected
    avg_signal = np.mean(audio_data, axis=1)
    sound_detected = detect_sound(avg_signal, threshold=SOUND_THRESHOLD)

    if not sound_detected:
        return [], None, False

    # Get mic positions
    mic_positions = get_mic_positions_4x4()

    # Azimuth scan range (0 to 360 degrees)
    azimuth_scan = np.arange(0, 360, azimuth_resolution)
    power_spectrum = np.zeros(len(azimuth_scan))

    num_samples = len(audio_data)

    # Beamforming: scan all directions
    for idx, azimuth in enumerate(azimuth_scan):
        # Convert azimuth to radians (accounting for coordinate system)
        # 0° = North, 90° = East, 180° = South, 270° = West
        theta_rad = np.radians(90 - azimuth)

        # Source direction unit vector
        source_direction = np.array([np.cos(theta_rad), np.sin(theta_rad)])

        # Calculate steering delays for each microphone
        delayed_signals = []
        for ch in range(CHANNELS):
            pos = np.array(mic_positions[ch])

            # Time delay for this mic relative to origin
            # Positive when source is in direction of microphone
            distance_proj = np.dot(pos, source_direction)
            # A mic displaced towards the source hears the wavefront early, so delay it to align
            time_delay = distance_proj / SPEED_OF_SOUND

            # Apply delay (shift signal)
            delay_samples = int(time_delay * SAMPLE_RATE)

            # Shift signal by delay
            if delay_samples > 0:
                # Pad at beginning, remove from end
                delayed = np.pad(audio_data[:, ch], (delay_samples, 0), mode='constant')[:num_samples]
            elif delay_samples < 0:
                # Pad at end, remove from beginning
                delayed = np.pad(audio_data[:, ch], (0, -delay_samples), mode='constant')[-num_samples:]
            else:
                delayed = audio_data[:, ch]

            delayed_signals.append(delayed)

        # Sum aligned signals (delay-and-sum beamforming)
        beamformed = np.mean(delayed_signals, axis=0)

        # Calculate power in this direction
        power_spectrum[idx] = np.mean(beamformed**2)

    # Normalize power spectrum
    power_spectrum = power_spectrum / (np.max(power_spectrum) + 1e-10)

    # Find peaks in power spectrum
    # distance parameter ensures minimum separation between peaks
    peaks, _ = find_peaks(
        power_spectrum,
        height=MIN_PEAK_HEIGHT,  # Minimum relative power
        distance=int(min_separation / azimuth_resolution),  # Minimum samples between peaks
        prominence=MIN_PROMINENCE  # How much peak stands out from background
    )

    # Sort peaks by power (strongest first)
    if len(peaks) > 0:
        peak_powers = power_spectrum[peaks]
        sorted_indices = np.argsort(peak_powers)[::-1]
        peaks = peaks[sorted_indices]
        peak_powers = peak_powers[sorted_indices]

        # Limit to num_sources if specified (None = dynamic, keep all peaks)
        if num_sources is not None:
            peaks = peaks[:num_sources]
            peak_powers = peak_powers[:num_sources]

        # Create source list with azimuth from beamforming
        sources = []
        for peak_idx, power in zip(peaks, peak_powers):
            # Azimuth from beamforming peak
            azimuth = azimuth_scan[peak_idx]

            # Estimate confidence based on peak power
            confidence = float(power)

            sources.append({
                'azimuth': float(azimuth),
                'elevation': 0.0,  # Horizontal plane only
                'power': float(power),
                'confidence': confidence,
                'id': len(sources)  # Simple ID for coloring
            })

        return sources, (azimuth_scan, power_spectrum), sound_detected
    else:
        return [], (azimuth_scan, power_spectrum), sound_detected


# =============================================================================
# TDOA Functions (Currently Unused - Kept for Future Reference)
# =============================================================================
# These functions are commented out as we're using beamforming-only approach.
# They can be re-enabled if TDOA refinement is needed in the future.

"""
def estimate_azimuth_tdoa(audio_data):
    '''
    Estimate azimuth using row-based TDOA.

    For each row:
    1. Calculate time delays between microphones using cross-correlation
    2. Compute direction of arrival from time delays using geometry
    3. Combine angles using weighted circular mean

    Args:
        audio_data: Audio data array [samples, channels]

    Returns:
        azimuth: Azimuth angle in degrees (0° is North, clockwise)
        row_azimuths: Dictionary of azimuth values per row for debugging
    '''
    rows = get_row_layout()
    mic_positions = get_mic_positions_4x4()

    row_azimuths = {}

    # Process each row independently
    for row_idx, (row_channels, ref_ch) in enumerate(rows):
        # Use reference channel for this row
        ref_signal = audio_data[:, ref_ch]
        time_delays = {ref_ch: 0.0}

        # Calculate time delays for other mics in this row
        other_channels = [ch for ch in row_channels if ch != ref_ch]
        for ch in other_channels:
            mic_signal = audio_data[:, ch]
            correlation = correlate(mic_signal, ref_signal, mode='same')
            lags = np.arange(-len(ref_signal)//2, len(ref_signal)//2)
            peak_idx = np.argmax(np.abs(correlation))
            peak_lag = lags[peak_idx]
            time_delays[ch] = peak_lag / SAMPLE_RATE

        # Calculate DOA for this row using time delays
        angles = []
        weights = []

        for ch, td in time_delays.items():
            pos = mic_positions[ch]

            # Calculate distance difference
            distance_diff = td * SPEED_OF_SOUND

            # Calculate actual distance from reference mic to this mic
            ref_pos = mic_positions[ref_ch]
            mic_distance = np.sqrt((pos[0] - ref_pos[0])**2 + (pos[1] - ref_pos[1])**2)

            # Avoid division by zero for reference mic
            if mic_distance < 1e-6:
                continue

            # Get mic angle from reference position
            mic_angle = np.arctan2(pos[1] - ref_pos[1], pos[0] - ref_pos[0])

            # The time delay tells us if sound arrives early (positive) or late (negative)
            if abs(distance_diff) < mic_distance:
                # Use cosine rule: d = baseline * cos(theta)
                cos_theta = distance_diff / mic_distance
                cos_theta = np.clip(cos_theta, -1, 1)  # Ensure valid range

                # Source angle calculation
                source_angle = mic_angle - np.arccos(cos_theta)

                angles.append(source_angle)
                weights.append(abs(td) + 1e-6)  # Weight by time delay magnitude

        if len(angles) > 0 and sum(weights) > 0:
            # Circular mean for angles
            angles = np.array(angles)
            weights = np.array(weights)

            # Convert to unit vectors, average, then back to angle
            x_avg = np.average(np.cos(angles), weights=weights)
            y_avg = np.average(np.sin(angles), weights=weights)

            avg_angle = np.arctan2(y_avg, x_avg)

            # Convert to degrees (0° = East, counter-clockwise)
            # Then convert to compass bearing (0° = North, clockwise)
            row_azimuth = (90 - np.degrees(avg_angle)) % 360
            row_azimuths[f'row_{row_idx}'] = row_azimuth

    if len(row_azimuths) > 0:
        # Combine row azimuths using circular mean
        row_angles_rad = [np.radians(az) for az in row_azimuths.values()]

        # Convert to unit vectors
        x_components = [np.cos(angle) for angle in row_angles_rad]
        y_components = [np.sin(angle) for angle in row_angles_rad]

        # Calculate mean of components
        x_mean = np.mean(x_components)
        y_mean = np.mean(y_components)

        # Convert back to angle
        combined_angle_rad = np.arctan2(y_mean, x_mean)
        combined_azimuth = (90 - np.degrees(combined_angle_rad)) % 360

        return float(combined_azimuth), row_azimuths
    else:
        return None, {}


def estimate_elevation_tdoa(audio_data, azimuth):
    '''
    Estimate elevation angle using TDOA on columns.

    For each column, we calculate time delays relative to the bottom microphone (row 0).

    Args:
        audio_data: Audio data array [samples, channels]
        azimuth: Detected azimuth angle in degrees

    Returns:
        elevation: Elevation angle in degrees (0° = horizontal, +90° = directly above)
        column_delays: Dictionary of time delays for each column
    '''
    columns = get_column_layout()
    mic_positions = get_mic_positions_4x4()

    # Determine which column(s) are most aligned with the azimuth
    theta_rad = np.radians(90 - azimuth)
    direction_vector = np.array([np.cos(theta_rad), np.sin(theta_rad)])

    # Calculate alignment score for each column
    column_scores = []
    for col_idx, (col_channels, ref_ch) in enumerate(columns):
        # Get position of bottom mic in this column
        bottom_pos = np.array(mic_positions[col_channels[0]])
        # Get position of top mic in this column
        top_pos = np.array(mic_positions[col_channels[3]])
        # Column direction vector (bottom to top)
        col_vector = top_pos - bottom_pos
        col_vector = col_vector / (np.linalg.norm(col_vector) + 1e-10)

        # Score based on perpendicularity to source direction
        score = abs(np.dot(col_vector, direction_vector))
        column_scores.append(score)

    # Use all columns and average (weighted by alignment)
    column_elevations = []
    column_weights = []
    column_delays_dict = {}

    for col_idx, (col_channels, ref_ch) in enumerate(columns):
        # Use reference channel for this column
        ref_signal = audio_data[:, ref_ch]

        # Calculate time delays for other mics in this column
        time_delays = {ref_ch: 0.0}

        other_channels = [ch for ch in col_channels if ch != ref_ch]
        for ch in other_channels:
            mic_signal = audio_data[:, ch]
            correlation = correlate(mic_signal, ref_signal, mode='same')
            lags = np.arange(-len(ref_signal)//2, len(ref_signal)//2)
            peak_idx = np.argmax(np.abs(correlation))
            peak_lag = lags[peak_idx]
            time_delays[ch] = peak_lag / SAMPLE_RATE

        column_delays_dict[f'column_{col_idx}'] = time_delays

        # Calculate elevation from time delays
        delays = [time_delays[ch] for ch in other_channels]
        avg_delay = np.mean(delays)

        # Distance difference from bottom to top
        bottom_pos = np.array(mic_positions[col_channels[0]])
        top_pos = np.array(mic_positions[col_channels[3]])
        vertical_baseline = np.linalg.norm(top_pos - bottom_pos)

        # Calculate elevation
        distance_diff = avg_delay * SPEED_OF_SOUND

        if abs(distance_diff) < vertical_baseline:
            # sin(elevation) = distance_diff / vertical_baseline
            sin_elevation = -distance_diff / vertical_baseline
            sin_elevation = np.clip(sin_elevation, -1, 1)
            elevation_rad = np.arcsin(sin_elevation)
            elevation_deg = np.degrees(elevation_rad)

            column_elevations.append(elevation_deg)
            column_weights.append(1.0 - column_scores[col_idx])

    if len(column_elevations) > 0 and sum(column_weights) > 0:
        # Weighted average of elevations
        elevation = np.average(column_elevations, weights=column_weights)
        return float(elevation), column_delays_dict
    else:
        return 0.0, column_delays_dict
"""

# =============================================================================
# Active Functions
# =============================================================================


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

            # Detect multiple sources using beamforming
            sources, spectrum_data, sound_detected = detect_multiple_sources_beamforming(
                audio_data,
                num_sources=MAX_SOURCES,
                min_separation=MIN_SEPARATION,
                azimuth_resolution=AZIMUTH_RESOLUTION
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
    ax1.set_title('Multi-Source DOA\n(Beamforming)', fontsize=12, weight='bold')

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
    print("MULTI-SOURCE BEAMFORMING - 4x4 MICROPHONE ARRAY (V2)")
    print("=" * 80)
    print(f"Array configuration: 16 microphones in 4x4 grid")
    print(f"Microphone spacing: {MIC_SPACING*1000:.1f} mm")
    print(f"Origin: CENTER of array (0,0)")
    print(f"Detection method: Delay-and-Sum Beamforming")
    print(f"Maximum sources: {'Dynamic (all valid peaks)' if MAX_SOURCES is None else MAX_SOURCES}")
    print(f"Minimum separation: {MIN_SEPARATION}°")
    print(f"Azimuth resolution: {AZIMUTH_RESOLUTION}°")
    print(f"Peak detection threshold: {MIN_PEAK_HEIGHT}")
    print(f"Sound threshold: {SOUND_THRESHOLD} (RMS amplitude)")
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

    print(f"\nStarting real-time multi-source beamforming...")
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
