# DSP Toolbox in Python

import numpy as np 
from scipy import signal as sg
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt 
import math 
 
"""
x = np.arange(-10.0, 10, 0.01) 
y = np.sin(x) 
 
plt.plot(x, y) 
plt.show() 
"""

"""
F = 10
T = 10/F
Fs = 5000
Ts = 1./Fs
N = int(T/Ts)

t = np.linspace(0, T, N)
signal = np.sin(2*np.pi*F*t)

plt.plot(t, signal)
plt.show()
"""

#start_time = 0
#end_time = 1
"""
sample_rate = float(input("Enter sampe rate (1/fs): "))
sampling_frequency = 1/sample_rate
#sample_rate = 1000 #1/fs
time = np.arange(start_time, end_time, sample_rate)
"""
#sampling_frequency = float(input("Enter sampling frequency (in Hz): "))  # e.g., 1000 Hz
#sample_period = 1 / sampling_frequency  # Time interval between samples (in seconds)

# Create time array using sample period as the step size
#time = np.arange(start_time, end_time, sample_period)

# PART 1 : Signal generator

# Choose the singal you want - sin / cos /  square / triangular
#signal_input = input("Enter desired signal between sin/cos/square/triangular: ").strip().lower()

def generate_signal(signal_input, time, amplitude, frequency, phase_shift_degrees):
    phase_shift_radians = np.deg2rad(phase_shift_degrees) 
    t = np.linspace(0, float(time), 1000)  
    if signal_input == 'sin':
        signal = amplitude * np.sin(2 * np.pi * frequency * t + phase_shift_radians)
        
    elif signal_input == 'cos':
        signal = amplitude * np.cos(2 * np.pi * frequency * t + phase_shift_radians)
        
    elif signal_input == 'square':
        #signal = [amplitude if np.sin(2*np.pi*frequency * (_time - phase)) > 0 else -amplitude for _time in time]
        duty_cycle = float(input("Enter duty cycle: "))
        signal = amplitude*sg.square(2*np.pi*frequency*t + 2*np.pi*phase_shift_radians, duty=duty_cycle)
        
    elif signal_input == 'triangular':
        width_input = float(input("Enter width: "))
        signal = amplitude*sg.sawtooth(2*np.pi*frequency*t + 2*np.pi*phase_shift_radians, width=width_input)
        
    else:
         raise ValueError("Unsupported signal type")

    return t,signal 
    

# Plot 
"""
def plot_signal(time, signal, signal_input, amplitude, frequency, phase_shift_radians):
    if signal is not None:
        plt.figure(figsize=(20, 6), dpi=80)
        plt.plot(time, signal)  
        plt.title(f'{signal_input.capitalize()} Wave - Amplitude: {amplitude}, Frequency: {frequency} Hz, Phase: {phase_shift_radians:.2f} radians')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.grid(True)
        plt.show()  
"""
#  PART 2: Filters
def apply_filter(signal, filter_type, filter_order, sampling_frequency):

# Choose the desired filter
#filter_type = input("Enter the filter between lowpass, highpass, bandpass, bandstop: ").lower()
#filter_order = int(input("Enter the order of the filter: "))

    # Dictionary to map filter types to btype values of butter
    filter_mapping = {
        'lowpass': 'low',
        'highpass': 'high',
        'bandpass': 'bandpass',
        'bandstop': 'bandstop'
    }

    if filter_type not in filter_mapping:
        raise ValueError("Unsupported filter type. Choose from 'lowpass', 'highpass', 'bandpass', 'bandstop'.")
    
    if filter_type in ['lowpass', 'highpass']:
        critical_frequency = float(input("Enter critical frequency: "))
        filter = sg.butter(filter_order, critical_frequency, btype=filter_mapping[filter_type], analog=False, output='ba', fs=sampling_frequency)

    elif filter_type in ['bandpass', 'bandstop']:
        low_cutoff_frequency = float(input("Enter low cutoff frequency: "))
        high_cutoff_frequency = float(input("Enter high cutoff frequency: "))
        filter = sg.butter(filter_order, [low_cutoff_frequency, high_cutoff_frequency], btype=filter_mapping[filter_type], fs=sampling_frequency)
    
        
    # Plot filter
    b, a = filter
    #w, h = sg.freqs(b, a) # Use freqz for analog filters
    w, h = sg.freqz(b, a, fs=sampling_frequency)  # Use freqz for digital filters
    plt.plot(w, abs(h))
    plt.title(f'{filter_type.capitalize()} Filter - Order: {filter_order}')
    plt.xlabel('Frequency [radians / second]')
    plt.ylabel('Amplitude')
    plt.margins(0, 0.1)
    plt.grid(which='both', axis='both')
    plt.show()

    filtered_signal = sg.filtfilt(b, a, signal)
    return filtered_signal

#  PART 3 : FFT & IFFT
def my_fft(signal,sampling_frequency):
    signal_after_fft = np.fft.fft(signal)
    len_signal = len(signal)  # Length of the signal
    # Get frequency values for the x-axis
    frequencies = np.fft.fftfreq(len_signal, d=1/sampling_frequency)
    # Get magnitude (absolute value) of the FFT and normalize it
    magnitude = np.abs(signal_after_fft) / len_signal
    # Only plot the positive half of the spectrum (real signals are symmetric)
    n = len_signal // 2
    frequencies = frequencies[:n] 
    magnitude = magnitude[:n]

    peaks, _ = sg.find_peaks(magnitude)  # Default peak detection

    # Get the maximum peak value and its index
    max_peak_value = np.max(magnitude[peaks])
    max_peak_index = peaks[np.argmax(magnitude[peaks])]
    max_peak_frequency = frequencies[max_peak_index]

    # Plot the FFT
    plt.figure(figsize=(10, 6))
    plt.plot(frequencies, magnitude)
    plt.title('FFT of the Signal')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.plot(max_peak_frequency, max_peak_value, "ro")  # Red dot at the maximum peak
    plt.text(max_peak_frequency, max_peak_value, f'{max_peak_frequency:.2f} Hz', fontsize=12, color='red', ha='left')
    plt.grid(True)
    plt.show()
    
    return signal_after_fft
 
def my_ifft(fft_result, start_time, end_time):
    signal_after_ifft = np.fft.ifft(fft_result)
    ifft_time = np.linspace(start_time, end_time, len(signal_after_ifft))
    # Plot the IFFT
    plt.figure(figsize=(10, 6))
    plt.plot(ifft_time,signal_after_ifft.real)
    plt.title('IFFT of the Signal')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.show()
    return signal_after_ifft

# PART 4: Noise Addition
#desired_noise = input("Enter the desired noise - white/pink/brown: ").lower()

    # Add Gaussian (White) Noise
def add_white_noise(signal, mean, variance):
    noise = np.random.normal(mean, variance, len(signal))
    return noise

def add_pink_noise(signal, sampling_frequency):
    white_noise = np.random.normal(0, 1, len(signal))
    noise_fft = np.fft.fft(white_noise)
    frequencies = np.fft.fftfreq(len(white_noise), d=1/sampling_frequency)
    # Scale the FFT by 1/sqrt(f) for pink noise, avoiding division by 0
    with np.errstate(divide='ignore', invalid='ignore'):
        scaling_factor = np.where(frequencies == 0, 0, 1 / np.sqrt(np.abs(frequencies)))
    noise_fft_scaled_to_pink = noise_fft * scaling_factor
    noise = np.fft.ifft(noise_fft_scaled_to_pink).real
    return noise

def add_brown_noise(signal,sampling_frequency):
    white_noise = np.random.normal(0, 1, len(signal))
    # FFT of the white noise
    noise_fft = np.fft.fft(white_noise)
    frequencies = np.fft.fftfreq(len(white_noise), d=1/sampling_frequency)
    # Scale the FFT by 1/(f^2) for brown noise, avoiding division by 0
    with np.errstate(divide='ignore', invalid='ignore'):
        scaling_factor = np.where(frequencies == 0, 0, 1 / np.power(np.abs(frequencies), 2))
    noise_fft_scaled_to_brown = noise_fft * scaling_factor
    noise = np.fft.ifft(noise_fft_scaled_to_brown).real
    return noise

def scale_noise_to_snr(signal, noise, desired_snr):
    """Scale the noise to achieve the desired SNR."""
    signal_power = np.mean(signal ** 2)
    desired_noise_power = signal_power / (10 ** (desired_snr / 10))
    actual_noise_power = np.mean(noise ** 2)
    scale_factor = math.sqrt(desired_noise_power / actual_noise_power)
    return noise * scale_factor
 
def add_noise(signal, noise_type, sampling_frequency, desired_snr, mean=0, variance=1):
    """Add the specified noise type to the signal and scale it to the desired SNR."""
    if noise_type == 'white':
        noise = add_white_noise(signal, mean, variance)
    elif noise_type == 'pink':
        noise = add_pink_noise(signal, sampling_frequency)
    elif noise_type == 'brown':
        noise = add_brown_noise(signal, sampling_frequency)
    else:
        raise ValueError("Unsupported noise type.")

    # Scale noise to the desired SNR
    noise_scaled = scale_noise_to_snr(signal, noise, desired_snr)
    signal_noised = signal + noise_scaled

    return signal_noised, noise_scaled

def plot_noised_signal(time, signal_noised, signal_type, amplitude, frequency, phase_shift, noise_type, snr):
    """Plot the noised signal."""
    plt.figure(figsize=(20, 6), dpi=80)
    plt.plot(time, signal_noised)
    plt.title(f'{signal_type.capitalize()} Wave - Amplitude: {amplitude}, Frequency: {frequency} Hz, '
              f'Phase: {phase_shift:.2f} radians, Noise: {noise_type}, SNR: {snr} dB')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.show()

#  PART 2: Filters
# Choose the desired filter
def recommend_filters():
    print("Filter Recommendations:")
    print("1. Low-pass filter: Removes high-frequency noise. Recommended for signals with dominant low frequencies and to white noise.")
    print("   - Suggestion: Set the cutoff frequency slightly higher than the highest signal frequency.")

    print("2. High-pass filter: Removes low-frequency noise. Recommended for signals with dominant high frequencies and to brown noise.")
    print("   - Suggestion: Set the cutoff frequency slightly lower than the lowest signal frequency.")

    print("3. Band-pass filter: Recommended for signals with specific frequency ranges or dealing with pink noise.")
    print("   - Suggestion: Set the cutoff frequency slightly higher than the signal frequency.")
    print("The order is preffered to be 2 or 3. Higher-order filters can provide sharper transitions between pass and stop bands, but they can also introduce more phase distortion and complexity.")

def get_filter_parameters():
    filter_type = input("Enter the filter between lowpass, highpass, bandpass, bandstop: ").strip().lower()
    filter_order = int(input("Enter the order of the filter: "))
    if filter_type in ['lowpass', 'highpass']:
        critical_frequency = float(input("Enter critical frequency: "))
        return filter_type, filter_order, [critical_frequency]

    elif filter_type in ['bandpass', 'bandstop']:
        low_cutoff_frequency = float(input("Enter low cutoff frequency: "))
        high_cutoff_frequency = float(input("Enter high cutoff frequency: "))
        return filter_type, filter_order, [low_cutoff_frequency, high_cutoff_frequency]
    
    else:
        print("Unsupported filter type.")
        return None, None, None

def design_filter(filter_type, filter_order, cutoff_frequencies, sampling_frequency):
    """Design and return the specified filter."""
    filter_mapping = {
        'lowpass': 'low',
        'highpass': 'high',
        'bandpass': 'bandpass',
        'bandstop': 'bandstop'
    }
    b, a = sg.butter(filter_order, cutoff_frequencies, btype=filter_mapping[filter_type], fs=sampling_frequency)
    return b, a

def plot_filter_response(b, a, sampling_frequency, filter_type, filter_order):
    """Plot the frequency response of the designed filter."""
    w, h = sg.freqz(b, a, fs=sampling_frequency)
    plt.plot(w, abs(h))
    plt.title(f'{filter_type.capitalize()} Filter - Order: {filter_order}')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Amplitude')
    plt.grid(which='both', axis='both')
    plt.show()

def apply_filter(b, a, signal):
    """Apply the filter to the signal and return the filtered signal."""
    return sg.lfilter(b, a, signal)

def plot_time_domain_response(time, signal, title):
    """Plot the signal in the time domain."""
    plt.plot(time, signal)
    plt.title(title)
    plt.xlabel('Time [seconds]')
    plt.ylabel('Amplitude')
    plt.grid(which='both', axis='both')
    plt.show()
"""
# Check fft after adding noise
my_fft(signal=signal)
my_fft(signal=signal_noised)
my_fft(signal=filter_response)
"""

def remove_dc_component(signal):
    """Remove the DC component from the signal."""
    return signal - np.mean(signal)

def decimate_signal(signal, sampling_frequency, downsampling_factor):
    """Decimate (downsample) the signal by a given factor."""
    wave_duration = len(signal) / sampling_frequency
    samples_decimated = int(len(signal) / downsampling_factor)
    
    # Perform decimation
    signal_decimated = sg.decimate(signal, downsampling_factor)
    
    # Generate new time array for the decimated signal
    time_decimated = np.linspace(0, wave_duration, samples_decimated, endpoint=False)
    
    return time_decimated, signal_decimated

def interpolate_signal(signal, time, interpolation_factor):
    """Interpolate (upsample) the signal by a given factor."""
    wave_duration = time[-1] - time[0]
    samples_interpolated = len(signal) * interpolation_factor
    
    # Generate new time array for the interpolated signal
    time_interpolated = np.linspace(time[0], wave_duration, samples_interpolated, endpoint=False)
    
    # Perform interpolation
    interpolator = interp1d(time, signal, kind='linear', fill_value="extrapolate")
    signal_interpolated = interpolator(time_interpolated)
    
    return time_interpolated, signal_interpolated
"""
def plot_signal(time_original, signal_original, time_new, signal_new, title, labels):
   #Plot original and new signal on the same plot for comparison.
    plt.figure(figsize=(20, 6), dpi=80)
    plt.plot(time_original, signal_original, '.-', label=labels[0])
    plt.plot(time_new, signal_new, 'o-', label=labels[1])
    plt.title(title)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.legend(loc='best')
    plt.show()
"""

def plot_signal(time_original, signal_original, time_processed=None, signal_processed=None, labels=None):
    if labels is None:
        labels = ["Original Signal", "Processed Signal"]
    
    plt.plot(time_original, signal_original, '.-', label=labels[0],linewidth=1.5)
    if time_processed is not None and signal_processed is not None:
        plt.plot(time_processed, signal_processed, '.-', label=labels[1])
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.title("Signal Visualization")
    plt.grid(True)
    plt.show()

def main():
    # Simulation Parameters
    sampling_frequency = 1000  # Hz
    sample_period = 1 / sampling_frequency
    start_time, end_time = 0, 1
    time = np.arange(start_time, end_time, sample_period)

    # Generate a Signal
    signal_type = 'sin'
    amplitude = 1
    frequency = 10  # Hz
    phase_shift = 0  # degrees

    signal = generate_signal(signal_type, time, amplitude, frequency, phase_shift)
    plot_signal(time_original=time, signal_original=signal, time_processed=None, signal_processed=None, labels=None)

    # Add Noise to Signal
    noise_type = 'white'
    desired_snr = 20  # dB
    noised_signal, _ = add_noise(signal, noise_type, sampling_frequency, desired_snr)
    plot_noised_signal(time, noised_signal, signal_type, amplitude, frequency, phase_shift, noise_type, desired_snr)

    # Apply a Lowpass Filter
    filter_type, filter_order, cutoff_frequencies = 'lowpass', 2, [15]  # 15 Hz cutoff
    b,a = design_filter(filter_type, filter_order, cutoff_frequencies, sampling_frequency)
    filtered_signal = apply_filter(b, a,signal)
    plot_signal(time_original=time, signal_original=signal, time_processed=time, signal_processed=filtered_signal, labels=["Original", "Filtered"])
    #plot_signal(time, filtered_signal, f"{signal_type} (Filtered)", amplitude, frequency, phase_shift)

    # Perform FFT
    fft_result = my_fft(filtered_signal, sampling_frequency)
    time_decimated, signal_decimated = decimate_signal(signal, sampling_frequency, 10)
    plot_signal(time_original=time, signal_original=signal, time_processed=time_decimated, signal_processed=signal_decimated, labels=["Original", "Decimated"])


if __name__ == "__main__":
    main()