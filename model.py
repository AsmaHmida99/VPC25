import librosa
import numpy as np
import soundfile as sf
import scipy.signal

def anonymize(input_audio_path):
    """
    Applies multiple transformations to anonymize the voice.

    Parameters
    ----------
    input_audio_path : str
        Path to the original audio file.

    Returns
    -------
    anonymized_audio : numpy.ndarray
        The anonymized audio data.
    sr : int
        The sample rate of the processed audio.
    """

    # Load the audio file (ensure 16kHz sampling rate for consistency)
    audio, sr = librosa.load(input_audio_path, sr=16000)

    # Ensure the audio is at least 1 second long to avoid errors
    if len(audio) < sr:
        raise ValueError(f"Audio file {input_audio_path} is too short (<1s). Please use a longer file.")

    # 1️⃣ Modify pitch (change vocal height)
    try:
        audio = librosa.effects.pitch_shift(audio, sr=sr, n_steps=5)
    except Exception as e:
        print(f"Error in pitch shift for {input_audio_path}: {e}")

    # 2️⃣ Apply time stretching (speed up/slow down)
    try:
        audio = librosa.effects.time_stretch(audio, rate=1.1)
    except Exception as e:
        print(f"Error in time stretch for {input_audio_path}: {e}")

    # 3️⃣ Add white noise
    noise = np.random.normal(0, 0.005, audio.shape)
    audio = audio + noise

    # 4️⃣ Apply band-pass filtering (alter spectral characteristics)
    try:
        b, a = scipy.signal.butter(4, [300/(sr/2), 3400/(sr/2)], btype='band')
        audio = scipy.signal.filtfilt(b, a, audio)
    except Exception as e:
        print(f"Error in band-pass filtering for {input_audio_path}: {e}")

    # 5️⃣ Apply Vocal Tract Length Normalization (VTLN)
    try:
        warp_factor = np.random.uniform(0.9, 1.1)
        anonymized_audio = librosa.effects.pitch_shift(audio, sr=sr, n_steps=np.log2(warp_factor) * 12)
    except Exception as e:
        print(f"Error in VTLN for {input_audio_path}: {e}")
        anonymized_audio = audio  # Fallback to original audio if error occurs

    # Ensure audio remains within [-1, 1] range to avoid distortion
    anonymized_audio = np.clip(anonymized_audio, -1, 1)

    return anonymized_audio, sr  # ✅ Ensuring compatibility with evaluation.py
