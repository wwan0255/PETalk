# audio_processing.py

import librosa
import soundfile as sf
import numpy as np
from scipy.signal import butter, sosfiltfilt, find_peaks
import matplotlib.pyplot as plt
import librosa.display

# ==============================================================================
# SECTION 0: VISUALIZATION HELPERS (Optional but useful)
# ==============================================================================
def visualize_spectrograms(y_orig, y_proc, sr, low_cutoff, high_cutoff, out_path):
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    spec_params = {'sr': sr, 'x_axis': 'time', 'y_axis': 'log'}
    D_orig = librosa.amplitude_to_db(np.abs(librosa.stft(y_orig)), ref=np.max)
    librosa.display.specshow(D_orig, ax=ax[0], **spec_params)
    ax[0].set_title('Original Audio Spectrogram', fontsize=14)
    ax[0].axhline(y=low_cutoff, color='r', linestyle='--', label=f'{low_cutoff} Hz Cutoff')
    ax[0].axhline(y=high_cutoff, color='r', linestyle='--', label=f'{high_cutoff} Hz Cutoff')
    ax[0].legend(loc='upper right')
    D_proc = librosa.amplitude_to_db(np.abs(librosa.stft(y_proc)), ref=np.max)
    img_proc = librosa.display.specshow(D_proc, ax=ax[1], **spec_params)
    ax[1].set_title('Processed Audio (After Bandpass Filter)', fontsize=14)
    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(img_proc, cax=cbar_ax, format='%+2.0f dB')
    cbar.set_label('Power (dB)', fontsize=12)
    plt.suptitle('Effect of Bandpass Filter', fontsize=18)
    plt.savefig(out_path)
    plt.close()
    print(f"  - Spectrogram comparison saved to: {out_path}")

# ==============================================================================
# SECTION 1: CORE AUDIO FUNCTIONS
# ==============================================================================
def bandpass_filter(y, sr, low=50, high=14000):
    nyquist = sr / 2
    if high >= nyquist: high = nyquist * 0.99
    sos = butter(8, [low, high], btype='band', fs=sr, output='sos')
    return sosfiltfilt(sos, y)

def detect_plosives(y, sr, percentile_threshold=96, distance_ms=80):
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=2048, hop_length=512)[0]
    high_freq_y = bandpass_filter(y, sr, low=2000, high=12000)
    high_energy = librosa.feature.rms(y=high_freq_y, frame_length=2048, hop_length=512)[0]
    energy_diff = np.abs(np.diff(high_energy, prepend=high_energy[0]))
    plosive_score = zcr * energy_diff[:len(zcr)]
    distance_frames = int((distance_ms / 1000) * sr / 512)
    peaks, _ = find_peaks(plosive_score, height=np.percentile(plosive_score, percentile_threshold), distance=distance_frames)
    return peaks, plosive_score

def enhance_plosives_audio(y, sr, enhancement_factor=1.5):
    peaks, scores = detect_plosives(y, sr)
    if len(peaks) > 0:
        print(f"  - Plosive Enhancement: Found {len(peaks)} potential plosives.")
        hop_length = 512
        for peak in peaks:
            start, end = max(0, peak * hop_length - 1024), min(len(y), peak * hop_length + 1024)
            segment = y[start:end].copy()
            if len(segment) > 0:
                high_freq_comp = bandpass_filter(segment, sr, low=1500, high=8000)
                y[start:end] = segment * 0.9 + high_freq_comp * enhancement_factor * 0.1
        print(f"    -> Enhancement complete.")
    else:
        print("  - Plosive Enhancement: No significant plosives detected.")
    return y

def optimize_lip_sync_timing(y, sr):
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr, backtrack=True, units='frames')
    if len(onset_frames) > 0:
        print(f"  - Lip-Sync Timing: Found {len(onset_frames)} audio onsets.")
        pre_emphasis_samples = int(0.02 * sr)
        y_shifted = np.pad(y[pre_emphasis_samples:], (0, pre_emphasis_samples), mode='constant')
        print("    -> Audio timing advanced slightly for better sync.")
        return y_shifted
    return y

# ==============================================================================
# SECTION 2: MAIN PROCESSING WORKFLOW
# ==============================================================================
def process_audio(in_path, out_path, visualize=False):
    print("\n--- Starting Audio Pre-processing Pipeline ---")
    y, sr = librosa.load(in_path, sr=None)
    y_original = y.copy()

    low_cutoff, high_cutoff = 50, 14000
    y_processed = bandpass_filter(y, sr, low=low_cutoff, high=high_cutoff)
    y_processed = enhance_plosives_audio(y_processed, sr, enhancement_factor=1.2)
    y_processed = optimize_lip_sync_timing(y_processed, sr)

    # Normalize audio to prevent clipping
    max_val = np.max(np.abs(y_processed))
    if max_val > 0.95:
        y_processed = y_processed * 0.95 / max_val
        
    sf.write(out_path, y_processed, sr)
    print(f"--- Audio processing complete. File saved to: {out_path} ---")

    if visualize:
        vis_path = out_path.replace(".wav", "_spectrogram.png")
        visualize_spectrograms(y_original, y_processed, sr, low_cutoff, high_cutoff, vis_path)
    
    return out_path