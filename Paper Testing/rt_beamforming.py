import sounddevice as sd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from scipy.signal import find_peaks, correlate
import queue

# Constants for DOA calculation
SPEED_OF_SOUND = 343.0  # m/s at room temperature
MIC_SPACING = 0.042  # meters (spacing between mics in the 4x4 array)

# Audio settings
SAMPLE_RATE = 48000  # Sample rate for 16-channel array
CHANNELS = 16  # 16 microphones in 4x4 array
BUFFER_DURATION = 0.5  # seconds of audio to process
SOUND_THRESHOLD = 0.002  # RMS threshold for sound detection

# Multi-source detection settings
MAX_SOURCES = 3  # Maximum number of sources to detect
MIN_SEPARATION = 15  # Minimum angular separation between sources (degrees)
AZIMUTH_RESOLUTION = 2  # Degrees (lower = higher resolution but slower)
ELEVATION_RESOLUTION = 5  # Degrees (for elevation estimation)
MIN_PEAK_HEIGHT = 0.3  # Minimum relative power for peak detection (0-1)
MIN_PROMINENCE = 0.15  # How much peak must stand out from background

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

    Physical layout - actual configuration:
    - Outer perimeter (12 mics), clockwise from bottom-left:
      MIC02, MIC04, MIC06, MIC08, MIC07, MIC10, MIC09, MIC11, MIC13, MIC15, MIC16, MIC01
    - Inner square (4 mics), clockwise from bottom-left:
      MIC03, MIC05, MIC12, MIC14

    Channel mapping: channel 0 = MIC01, channel 1 = MIC02, etc.

    Returns:
        Dictionary mapping channel numbers to (x, y) positions in meters
        Origin (0, 0) is at the CENTER of the array (between rows 1 & 2, columns 1 & 2)
    """
    # Spacing between outer perimeter mics
    outer_spacing = MIC_SPACING

    # Center of array is at 1.5 * outer_spacing from bottom-left corner in original coords
    # We'll shift everything so origin is at the center
    center_offset_x = 1.5 * outer_spacing
    center_offset_y = 1.5 * outer_spacing

    # Inner square spacing
    inner_half_spacing = outer_spacing / 2

    # Define all positions with origin at CENTER of array
    # Each position is shifted by -center_offset to put origin at center

    mic_positions = {
        # Outer perimeter - Bottom edge
        1: (0 - center_offset_x, 0 - center_offset_y),                            # MIC02 (channel 1) - bottom-left corner
        3: (outer_spacing - center_offset_x, 0 - center_offset_y),                # MIC04 (channel 3)
        5: (2 * outer_spacing - center_offset_x, 0 - center_offset_y),            # MIC06 (channel 5)
        7: (3 * outer_spacing - center_offset_x, 0 - center_offset_y),            # MIC08 (channel 7) - bottom-right corner

        # Outer perimeter - Right edge
        6: (3 * outer_spacing - center_offset_x, outer_spacing - center_offset_y),        # MIC07 (channel 6)
        9: (3 * outer_spacing - center_offset_x, 2 * outer_spacing - center_offset_y),    # MIC10 (channel 9)

        # Outer perimeter - Top edge
        8: (3 * outer_spacing - center_offset_x, 3 * outer_spacing - center_offset_y),    # MIC09 (channel 8) - top-right corner
        10: (2 * outer_spacing - center_offset_x, 3 * outer_spacing - center_offset_y),   # MIC11 (channel 10)
        12: (outer_spacing - center_offset_x, 3 * outer_spacing - center_offset_y),       # MIC13 (channel 12)
        14: (0 - center_offset_x, 3 * outer_spacing - center_offset_y),                   # MIC15 (channel 14) - top-left corner

        # Outer perimeter - Left edge
        15: (0 - center_offset_x, 2 * outer_spacing - center_offset_y),                   # MIC16 (channel 15)
        0: (0 - center_offset_x, outer_spacing - center_offset_y),                        # MIC01 (channel 0)

        # Inner square (centered around origin)
        # MIC03, MIC05, MIC12, MIC14
        2: (-inner_half_spacing, -inner_half_spacing),      # MIC03 (channel 2) - inner bottom-left
        4: (inner_half_spacing, -inner_half_spacing),       # MIC05 (channel 4) - inner bottom-right
        11: (inner_half_spacing, inner_half_spacing),       # MIC12 (channel 11) - inner top-right
        13: (-inner_half_spacing, inner_half_spacing)       # MIC14 (channel 13) - inner top-left
    }

    return mic_positions


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


def get_row_layout():
    """
    Define the 4 rows of the 4x4 array for azimuth estimation using TDOA.

    Row layout (bottom to top):
    Row 0 (bottom): [1, 0, 15, 14]  - MIC02, MIC01, MIC16, MIC15 (ref: channel 14)
    Row 1:          [2, 3, 13, 12]  - MIC03, MIC04, MIC14, MIC13 (ref: channel 13)
    Row 2:          [4, 5, 11, 10]  - MIC05, MIC06, MIC12, MIC11 (ref: channel 5)
    Row 3 (top):    [6, 7, 9, 8]    - MIC07, MIC08, MIC10, MIC09 (ref: channel 6)

    Returns:
        List of tuples (row_channels, reference_channel)
    """
    rows = [
        ([1, 0, 15, 14], 14),   # Row 0, ref: MIC15 (channel 14)
        ([2, 3, 13, 12], 13),   # Row 1, ref: MIC14 (channel 13)
        ([4, 5, 11, 10], 5),    # Row 2, ref: MIC06 (channel 5)
        ([6, 7, 9, 8], 6)       # Row 3, ref: MIC07 (channel 6)
    ]
    return rows


def get_column_layout():
    """
    Define the 4 columns of the 4x4 array for elevation estimation using TDOA.

    Column layout (left to right, with row 0 at bottom):
    Column 0 (leftmost):  [1, 0, 15, 14]  - MIC02, MIC01, MIC16, MIC15
    Column 1:             [3, 2, 13, 12]  - MIC04, MIC03, MIC14, MIC13
    Column 2:             [5, 4, 11, 10]  - MIC06, MIC05, MIC12, MIC11
    Column 3 (rightmost): [7, 6, 9, 8]    - MIC08, MIC07, MIC10, MIC09

    Returns:
        List of 4 column arrays, each containing channel numbers from bottom to top
    """
    columns = [
        [1, 0, 15, 14],   # Column 0 (leftmost)
        [3, 2, 13, 12],   # Column 1
        [5, 4, 11, 10],   # Column 2
        [7, 6, 9, 8]      # Column 3 (rightmost)
    ]
    return columns


def estimate_azimuth_tdoa(audio_data, approximate_azimuth):
    """
    Estimate azimuth using row-based TDOA.

    For each row:
    1. Calculate time delays between microphones using cross-correlation
    2. Compute direction of arrival from time delays using geometry
    3. Combine angles using weighted circular mean

    Then combine all row azimuths using RMS method.

    Args:
        audio_data: Audio data array [samples, channels]
        approximate_azimuth: Rough azimuth from beamforming peak (fallback value)

    Returns:
        refined_azimuth: Azimuth angle in degrees (0° is North, clockwise)
        row_azimuths: Dictionary of azimuth values per row for debugging
    """
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
            mic_distance = np.sqrt(pos[0]**2 + pos[1]**2)

            # Avoid division by zero for reference mic
            if mic_distance < 1e-6:
                continue

            # Get mic angle from reference position
            mic_angle = np.arctan2(pos[1], pos[0])

            # The time delay tells us if sound arrives early (positive) or late (negative)
            # at this mic compared to reference
            if abs(distance_diff) < mic_distance:
                # Use cosine rule: d = baseline * cos(theta)
                cos_theta = distance_diff / mic_distance
                cos_theta = np.clip(cos_theta, -1, 1)  # Ensure valid range

                # Source angle calculation
                source_angle = mic_angle - np.arccos(cos_theta)

                angles.append(source_angle)
                weights.append(abs(td))  # Weight by time delay magnitude

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
        # Combine row azimuths using RMS method (like original script)
        row_angles_rad = [np.radians(az) for az in row_azimuths.values()]

        # Convert to unit vectors
        x_components = [np.cos(angle) for angle in row_angles_rad]
        y_components = [np.sin(angle) for angle in row_angles_rad]

        # Calculate RMS of components
        x_rms = np.sqrt(np.mean(np.array(x_components)**2))
        y_rms = np.sqrt(np.mean(np.array(y_components)**2))

        # Convert back to angle
        combined_angle_rad = np.arctan2(y_rms, x_rms)
        combined_azimuth = (90 - np.degrees(combined_angle_rad)) % 360

        return float(combined_azimuth), row_azimuths
    else:
        return float(approximate_azimuth), {}


def estimate_elevation_tdoa(audio_data, azimuth):
    """
    Estimate elevation angle using TDOA on columns.

    For each column, we calculate time delays relative to the bottom microphone (row 0).
    We then use the column that is most aligned with the detected azimuth direction.

    Args:
        audio_data: Audio data array [samples, channels]
        azimuth: Detected azimuth angle in degrees

    Returns:
        elevation: Elevation angle in degrees (0° = horizontal, +90° = directly above)
        column_delays: Dictionary of time delays for each column
    """
    columns = get_column_layout()
    mic_positions = get_mic_positions_4x4()

    # Determine which column(s) are most aligned with the azimuth
    # Convert azimuth to angle (0° = North = +Y, 90° = East = +X)
    theta_rad = np.radians(90 - azimuth)
    direction_vector = np.array([np.cos(theta_rad), np.sin(theta_rad)])

    # Calculate alignment score for each column
    # Columns aligned perpendicular to azimuth are best for elevation
    column_scores = []
    for col_idx, col_channels in enumerate(columns):
        # Get position of bottom mic in this column
        bottom_pos = np.array(mic_positions[col_channels[0]])
        # Get position of top mic in this column
        top_pos = np.array(mic_positions[col_channels[3]])
        # Column direction vector (bottom to top)
        col_vector = top_pos - bottom_pos
        col_vector = col_vector / (np.linalg.norm(col_vector) + 1e-10)

        # Score based on perpendicularity to source direction
        # We want columns aligned with Y-axis for N/S sources, X-axis for E/W sources
        score = abs(np.dot(col_vector, direction_vector))
        column_scores.append(score)

    # Use all columns and average (weighted by alignment)
    column_elevations = []
    column_weights = []
    column_delays_dict = {}

    for col_idx, col_channels in enumerate(columns):
        # Reference is row 0 (bottom mic) of each column
        ref_channel = col_channels[0]
        ref_signal = audio_data[:, ref_channel]

        # Calculate time delays for other mics in this column
        time_delays = {ref_channel: 0.0}

        for ch in col_channels[1:]:
            mic_signal = audio_data[:, ch]
            correlation = correlate(mic_signal, ref_signal, mode='same')
            lags = np.arange(-len(ref_signal)//2, len(ref_signal)//2)
            peak_idx = np.argmax(np.abs(correlation))
            peak_lag = lags[peak_idx]
            time_delays[ch] = peak_lag / SAMPLE_RATE

        column_delays_dict[f'column_{col_idx}'] = time_delays

        # Calculate elevation from time delays
        # Average time delay across the column
        delays = [time_delays[ch] for ch in col_channels[1:]]
        avg_delay = np.mean(delays)

        # Distance difference from bottom to top
        bottom_pos = np.array(mic_positions[col_channels[0]])
        top_pos = np.array(mic_positions[col_channels[3]])
        vertical_baseline = np.linalg.norm(top_pos - bottom_pos)

        # Calculate elevation
        # Positive delay means sound arrives at top mics later (source below horizontal)
        # Negative delay means sound arrives at top mics earlier (source above horizontal)
        distance_diff = avg_delay * SPEED_OF_SOUND

        if abs(distance_diff) < vertical_baseline:
            # sin(elevation) = distance_diff / vertical_baseline
            sin_elevation = -distance_diff / vertical_baseline  # Negative for correct sign
            sin_elevation = np.clip(sin_elevation, -1, 1)
            elevation_rad = np.arcsin(sin_elevation)
            elevation_deg = np.degrees(elevation_rad)

            column_elevations.append(elevation_deg)
            column_weights.append(1.0 - column_scores[col_idx])  # Weight by perpendicularity

    if len(column_elevations) > 0 and sum(column_weights) > 0:
        # Weighted average of elevations
        elevation = np.average(column_elevations, weights=column_weights)
        return float(elevation), column_delays_dict
    else:
        return 0.0, column_delays_dict


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
        num_sources: Maximum number of sources to detect
        min_separation: Minimum angular separation between sources (degrees)
        azimuth_resolution: Angular resolution for scanning (degrees)

    Returns:
        sources: List of dictionaries with 'azimuth', 'elevation', 'power', 'confidence'
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
            time_delay = -distance_proj / SPEED_OF_SOUND

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

        # Limit to num_sources
        peaks = peaks[:num_sources]
        peak_powers = peak_powers[:num_sources]

        # Create source list with refined azimuth and elevation estimation
        sources = []
        for peak_idx, power in zip(peaks, peak_powers):
            # Approximate azimuth from beamforming peak
            approximate_azimuth = azimuth_scan[peak_idx]

            # Refine azimuth using row-based TDOA
            refined_azimuth, row_azimuths = estimate_azimuth_tdoa(audio_data, approximate_azimuth)

            # Estimate elevation using TDOA on columns
            elevation, column_delays = estimate_elevation_tdoa(audio_data, refined_azimuth)

            # Estimate confidence based on peak power
            confidence = float(power)

            sources.append({
                'azimuth': float(refined_azimuth),
                'elevation': float(elevation),
                'power': float(power),
                'confidence': confidence,
                'id': len(sources),  # Simple ID for coloring
                'row_azimuths': row_azimuths,
                'column_delays': column_delays,
                'beamforming_azimuth': float(approximate_azimuth)  # Keep original for comparison
            })

        return sources, (azimuth_scan, power_spectrum), sound_detected
    else:
        return [], (azimuth_scan, power_spectrum), sound_detected


def audio_callback(indata, frames, time, status):
    """
    Callback function for sounddevice stream.
    Called for each audio block received from the device.
    """
    if status:
        print(f"Status: {status}")

    # Put data in queue (non-blocking, discard if full)
    try:
        audio_queue.put_nowait(indata.copy())
    except queue.Full:
        pass  # Discard if queue is full


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

    # Keep buffer size manageable (last 1 second of audio)
    max_buffer_samples = int(SAMPLE_RATE * 1.0)
    if len(audio_buffer) > 0:
        total_samples = sum(len(chunk) for chunk in audio_buffer)

        # Remove old chunks if buffer is too large
        while total_samples > max_buffer_samples and len(audio_buffer) > 1:
            removed = audio_buffer.pop(0)
            total_samples -= len(removed)

    # Process if we have enough samples
    if len(audio_buffer) > 0:
        audio_data = np.vstack(audio_buffer)

        # Debug: print buffer size
        if frame % 10 == 0:  # Print every 10 frames
            # Calculate average RMS across all channels
            avg_rms = np.sqrt(np.mean(np.mean(audio_data**2, axis=1)))
            print(f"Buffer size: {len(audio_data)} samples, Avg RMS: {avg_rms:.4f}")

        # Only process if we have enough samples
        if len(audio_data) >= int(BUFFER_DURATION * SAMPLE_RATE):
            # Take last BUFFER_DURATION worth of samples
            audio_data = audio_data[-int(BUFFER_DURATION * SAMPLE_RATE):]

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
                sources_info = ", ".join([f"S{i+1}: Az={s['azimuth']:.1f}° (BF:{s['beamforming_azimuth']:.1f}°) El={s['elevation']:.1f}° (P:{s['power']:.2f})"
                                         for i, s in enumerate(sources)])
                print(f">>> Detected {len(sources)} source(s): {sources_info}")
            else:
                sound_active = sound_detected
                if sound_detected and len(sources) == 0:
                    print(">>> Sound detected but no clear sources identified")

    # Clear previous plots
    ax1.clear()
    ax2.clear()
    ax3.clear()

    # ===== Plot 1: Polar plot for DOA =====
    ax1.set_theta_zero_location('N')
    ax1.set_theta_direction(-1)
    ax1.set_title('Multi-Source 3D DOA\n(Beamforming + Row/Column TDOA)', fontsize=11, weight='bold')

    if sound_active and len(latest_sources) > 0:
        # Define colors for different sources
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan']

        # Plot each detected source
        for idx, source in enumerate(latest_sources):
            azimuth_rad = np.radians(source['azimuth'])
            color = colors[idx % len(colors)]

            # Arrow length based on confidence
            arrow_length = 0.6 + 0.3 * source['confidence']

            ax1.arrow(azimuth_rad, 0, 0, arrow_length,
                     head_width=0.25, head_length=0.1,
                     fc=color, ec=color, linewidth=2.5, alpha=0.8)

            # Label with source number, azimuth, and elevation
            ax1.text(azimuth_rad, arrow_length + 0.15,
                    f"S{idx+1}\nAz:{source['azimuth']:.1f}°\nEl:{source['elevation']:.1f}°",
                    ha='center', va='bottom', fontsize=9, color=color, weight='bold')

        # Add grid
        ax1.set_ylim(0, 1)
        ax1.grid(True, alpha=0.3)

    else:
        ax1.text(0, 0, 'No sound\ndetected',
                ha='center', va='center', fontsize=12, color='gray')
        ax1.set_ylim(0, 1)

    # ===== Plot 2: Beamforming Power Spectrum =====
    if sound_active and latest_spectrum is not None:
        azimuth_scan, power_spectrum = latest_spectrum

        # Plot power spectrum
        ax2.plot(azimuth_scan, power_spectrum, 'b-', linewidth=1.5, alpha=0.7, label='Power Spectrum')

        # Mark detected sources
        if len(latest_sources) > 0:
            colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan']
            for idx, source in enumerate(latest_sources):
                color = colors[idx % len(colors)]
                ax2.axvline(x=source['azimuth'], color=color, linestyle='--',
                           linewidth=2, alpha=0.8, label=f"Source {idx+1}")
                ax2.plot(source['azimuth'], source['power'], 'o',
                        color=color, markersize=10, markeredgewidth=2,
                        markeredgecolor='white')

        # Threshold line
        ax2.axhline(y=MIN_PEAK_HEIGHT, color='gray', linestyle=':',
                   linewidth=1, alpha=0.5, label=f'Threshold ({MIN_PEAK_HEIGHT})')

        ax2.set_xlabel('Azimuth (degrees)', fontsize=10)
        ax2.set_ylabel('Normalized Power', fontsize=10)
        ax2.set_title('Beamforming Power Spectrum', fontsize=11, weight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, 360)
        ax2.set_ylim(0, 1.1)
        ax2.legend(loc='upper right', fontsize=8)

        # Add compass labels
        ax2.set_xticks([0, 90, 180, 270, 360])
        ax2.set_xticklabels(['N (0°)', 'E (90°)', 'S (180°)', 'W (270°)', 'N (360°)'])
    else:
        ax2.text(0.5, 0.5, 'Waiting for sound...',
                ha='center', va='center', transform=ax2.transAxes,
                fontsize=12, color='gray')
        ax2.set_xlim(0, 360)
        ax2.set_ylim(0, 1)

    # ===== Plot 3: Source Information Table =====
    ax3.axis('tight')
    ax3.axis('off')

    if sound_active and len(latest_sources) > 0:
        # Create table data
        table_data = []
        table_data.append(['Source', 'Azimuth', 'Elevation', 'Power', 'Confidence'])

        colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan']
        cell_colors = [['lightgray', 'lightgray', 'lightgray', 'lightgray', 'lightgray']]

        for idx, source in enumerate(latest_sources):
            color = colors[idx % len(colors)]
            table_data.append([
                f"S{idx+1}",
                f"{source['azimuth']:.1f}°",
                f"{source['elevation']:.1f}°",
                f"{source['power']:.3f}",
                f"{source['confidence']:.2%}"
            ])
            # Light version of the color for cell background
            cell_colors.append([color, 'white', 'white', 'white', 'white'])

        table = ax3.table(cellText=table_data, cellLoc='center', loc='center',
                         cellColours=cell_colors, colWidths=[0.12, 0.20, 0.20, 0.20, 0.20])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)

        # Bold header
        for i in range(5):
            table[(0, i)].set_text_props(weight='bold')

        ax3.set_title('Detected Sources (Azimuth & Elevation)', fontsize=11, weight='bold', pad=20)
    else:
        ax3.text(0.5, 0.5, 'No sources detected',
                ha='center', va='center', transform=ax3.transAxes,
                fontsize=12, color='gray')


if __name__ == "__main__":
    print("=" * 80)
    print("MULTI-SOURCE DETECTION - 4x4 MICROPHONE ARRAY")
    print("=" * 80)
    print(f"Array configuration: 16 microphones")
    print(f"Origin: CENTER of array (0,0)")
    print(f"Detection method: Beamforming spatial scan (finds multiple sources)")
    print(f"Azimuth refinement: Row-based TDOA (4 rows, adaptive references)")
    print(f"Elevation estimation: Column-based TDOA (row 0 reference)")
    print(f"Microphone spacing: {MIC_SPACING*1000:.1f} mm")
    print(f"Maximum sources: {MAX_SOURCES}")
    print(f"Minimum separation: {MIN_SEPARATION}°")
    print(f"Azimuth resolution: {AZIMUTH_RESOLUTION}° (beamforming scan)")
    print(f"Detection threshold: {MIN_PEAK_HEIGHT}")
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

    print(f"\nStarting real-time multi-source detection...")
    print(f"Sample rate: {SAMPLE_RATE} Hz")
    print(f"Channels: {CHANNELS}")
    print(f"Sound threshold: {SOUND_THRESHOLD}")
    print(f"Buffer duration: {BUFFER_DURATION} seconds")

    # Start audio stream
    stream = sd.InputStream(
        device=minidsp_device,
        channels=CHANNELS,
        samplerate=SAMPLE_RATE,
        callback=audio_callback,
        blocksize=int(SAMPLE_RATE * 0.1)  # 100ms blocks
    )

    stream.start()

    # Set up matplotlib figure
    fig = plt.figure(figsize=(18, 10))
    ax1 = plt.subplot(2, 2, 1, projection='polar')
    ax2 = plt.subplot(2, 2, 2)
    ax3 = plt.subplot(2, 1, 2)

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
