#define MINIAUDIO_IMPLEMENTATION

#include <algorithm> // for std::clamp
#include <audiocore2.h>
#include <cmath>
#include <stdlib.h>
#include <iostream>
#include <thread>
#include <atomic>
#include <cstring> // for std::memset and std::memcpy

// Definitions ONLY
#define M_PI 3.14159265358979323846f
#define SAMPLE_RATE 48000
#define FFT_SIZE 2048
#define MAX_CURVE_POINTS 100
#define NUM_BANDS 7 // Added definition

// Type Definitions ONLY
typedef struct
{
    float start_freq;
    float end_freq;
} Band;

// Thread-safe primitives
std::atomic<float> peak_amplitude(0.0f);
std::atomic<float> recent_max(0.0f); // For dynamic compression
std::atomic<float> smoothed_level(0.0f);
std::atomic<float> current_gain(1.0f);

// Shared resources
float *band_amplitudes = nullptr;
float *flat_btw_amplitudes = nullptr; // Its store all values is 1d array.
int *foot_btw_amplitudes = nullptr;   // Its carry of how many elements in each row.
float *weights = nullptr;

// FFTW / Buffer
std::atomic<float> bin_spacing((SAMPLE_RATE / FFT_SIZE));
float *magnitudes = nullptr;
float *sample_buffer = nullptr;
float *pre_computed_window = nullptr;

// Miniaudio (usually set up once, not thread-sensitive)
ma_result result;
ma_device device;
size_t sample_offset = 0;

// FFTW
fftwf_plan fft_plan;
float *fft_input = nullptr;
fftwf_complex *fft_output = nullptr;
Band *bands = nullptr;

// Helper function to free allocated memory
void free_memory()
{
    delete[] band_amplitudes;
    band_amplitudes = nullptr;

    delete[] flat_btw_amplitudes;
    flat_btw_amplitudes = nullptr;

    delete[] weights;
    weights = nullptr;

    delete[] magnitudes;
    magnitudes = nullptr;

    delete[] sample_buffer;
    sample_buffer = nullptr;

    delete[] pre_computed_window;
    pre_computed_window = nullptr;

    delete[] fft_input;
    fft_input = nullptr;

    delete[] fft_output;
    fft_output = nullptr;

    delete[] bands;
    bands = nullptr;
}

// Map bands logarithmically
void ac2_setup_bands()
{
    float min_freq = 20.0f;
    float max_freq = 20000.0f;

    float log_min = log10f(min_freq);
    float log_max = log10f(max_freq);

    for (int i = 0; i < NUM_BANDS; ++i)
    {
        float fraction = static_cast<float>(i) / static_cast<float>(NUM_BANDS);
        float start = powf(10.0f, log_min + fraction * (log_max - log_min));
        float end = powf(10.0f, log_min + (fraction + 1.0f / NUM_BANDS) * (log_max - log_min));
        bands[i].start_freq = start;
        bands[i].end_freq = end;
    }

    if (!weights)
    {
        weights = new float[FFT_SIZE / 2 + 1];
        for (size_t i = 0; i < FFT_SIZE / 2 + 1; ++i)
        {
            float frequency = i * bin_spacing.load();
            weights[i] = std::sqrt(frequency / 1000.0f);
        }
    }
}

void ac2_precompute_window()
{
    for (size_t i = 0; i < FFT_SIZE; ++i)
    {
        pre_computed_window[i] = 0.5f * (1.0f - cosf(2.0f * M_PI * i / (FFT_SIZE - 1)));
    }
}

// Hann Window
void ac2_apply_window(float *data)
{
    for (size_t i = 0; i < FFT_SIZE; ++i)
    {
        data[i] *= pre_computed_window[i];
    }
}

// Dynamic Compression
float ac2_compress_dynamic(float value)
{
    return logf(value + 1.0f);
}

//float ac2_compress_dynamic(float input) {
//    if (input <= 0.4f) {
//        return input; // No boost
//    }
//    else if (input <= 0.7f) {
//        float t = (input - 0.4f) / 0.3f; // Normalize 0.4–0.7 to 0–1
//        float boost = 1.0f + (1.0f - cosf(t * 3.14159f)) * 0.2f; // Smooth boost to 1.2×
//        return input * boost;
//    }
//    else {
//        // Smooth compression to avoid harsh clipping
//        float t = (input - 0.7f) / 0.3f;
//        float compressed = 0.7f + (1.0f - powf(1.0f - t, 2.0f)) * 0.3f;
//        return std::min<float>(compressed, 1.0f);
//    }
//}

float ac2_visual_scale(float input) {
    float gain = 100.0f; // You can tweak this!
    float boosted = input * gain;
    boosted = std::clamp(boosted, 0.0f, 1.0f);
    return ac2_compress_dynamic(boosted);
}

float update_recent_max(float input, float recentMax, float decay = 0.95f) {
    float recent = std::max<float>(recentMax * decay, input);
	recent_max.store(recent); // Update the atomic variable
    return recent;
}

// Adaptive gain function
float apply_adaptive_gain(float input, float recentMax) {
    float gain = 1.0f / std::max<float>(recentMax, 0.001f); // Avoid div by 0
    return std::clamp(input * gain, 0.0f, 1.0f);
}

float ac2_apply_adaptive_gain(float input, std::atomic<float>& smoothedLevel, std::atomic<float>& currentGain) {
    const float targetLevel = 0.8f;
    const float smoothing = 0.95f;
    const float minGain = 1.0f;
    const float maxGain = 10.0f;

    // Read current values
    float smoothed = smoothedLevel.load();
    float gain = currentGain.load();

    // Smooth the input (simple envelope follower)
    smoothed = smoothing * smoothed + (1.0f - smoothing) * std::fabs(input);
    smoothed = std::max<float>(smoothed, 0.0001f); // Avoid division by zero

    // Compute new desired gain
    float desiredGain = targetLevel / smoothed;

    // Smoothly adjust gain toward the desired value
    gain = 0.9f * gain + 0.1f * desiredGain;
    gain = std::clamp(gain, minGain, maxGain);

    // Store updated values
    smoothedLevel.store(smoothed);
    currentGain.store(gain);

    // Apply gain to input
    return input * gain;
}

std::vector<std::vector<float>> create_mel_filterbank(int num_bands, int fft_size, float sample_rate)
{
    auto hz_to_mel = [](float hz) {
        return 2595.0f * log10f(1.0f + hz / 700.0f);
        };

    auto mel_to_hz = [](float mel) {
        return 700.0f * (powf(10.0f, mel / 2595.0f) - 1.0f);
        };

    float mel_min = hz_to_mel(0.0f);
    float mel_max = hz_to_mel(sample_rate / 2.0f);
    std::vector<float> mel_points(num_bands + 2);

    for (int i = 0; i < mel_points.size(); ++i)
        mel_points[i] = mel_to_hz(mel_min + (mel_max - mel_min) * i / (num_bands + 1));

    std::vector<int> bin_points(mel_points.size());
    for (int i = 0; i < mel_points.size(); ++i)
        bin_points[i] = static_cast<int>(mel_points[i] / (sample_rate / fft_size));

    std::vector<std::vector<float>> filterbank(num_bands, std::vector<float>(fft_size / 2 + 1, 0.0f));
    for (int i = 0; i < num_bands; ++i)
    {
        int left = bin_points[i];
        int center = bin_points[i + 1];
        int right = bin_points[i + 2];

        for (int j = left; j < center; ++j)
            if (j >= 0 && j < filterbank[i].size())
                filterbank[i][j] = (j - left) / float(center - left);
        for (int j = center; j < right; ++j)
            if (j >= 0 && j < filterbank[i].size())
                filterbank[i][j] = (right - j) / float(right - center);
    }

    return filterbank;
}

float ac2_track_peak(
    float current_amplitude,
    float& peak_amplitude,
    int& silent_frame_counter,
    const float silence_threshold = 0.01f,
    const int max_silent_frames = 200)
{
    // Update peak immediately if current amplitude is higher
    if (current_amplitude > peak_amplitude)
        peak_amplitude = current_amplitude;

    // Silence tracking
    if (current_amplitude < silence_threshold)
        ++silent_frame_counter;
    else
        silent_frame_counter = 0;

    // Reset peak after prolonged silence
    if (silent_frame_counter > max_silent_frames)
        peak_amplitude = 0.0f;

    return peak_amplitude;
}

void ac2_process_fft()
{
    // Copy the sample buffer to fft_input
    std::memcpy(fft_input, sample_buffer, FFT_SIZE * sizeof(float));

    // Apply window function
    ac2_apply_window(fft_input);

    // Execute FFT
    fftwf_execute(fft_plan);

    // Calculate magnitudes from FFT output
    for (size_t i = 0; i < FFT_SIZE / 2 + 1; ++i)
    {
        float real = fft_output[i][0];
        float imag = fft_output[i][1];
        magnitudes[i] = sqrtf(real * real + imag * imag);
    }

    // Reuse memory by zeroing band amplitudes
    std::memset(band_amplitudes, 0, NUM_BANDS * sizeof(float));

    // Allocate flat_btw_amplitudes if not already allocated
    if (!flat_btw_amplitudes)
    {
        flat_btw_amplitudes = new float[NUM_BANDS * MAX_CURVE_POINTS];
    }

    // Fill flat_btw_amplitudes with -1 initially
    std::fill(flat_btw_amplitudes, flat_btw_amplitudes + NUM_BANDS * MAX_CURVE_POINTS, -1.0f);

    // Pre-calculate the number of columns in each row and store in foot_btw_amplitudes
    if (!foot_btw_amplitudes)
    {
        foot_btw_amplitudes = new int[NUM_BANDS];
    }

    for (int b = 0; b < NUM_BANDS; ++b)
    {
        int start_bin = static_cast<int>(bands[b].start_freq / bin_spacing.load());
        start_bin = std::max<int>(1, start_bin);
        int end_bin = static_cast<int>(bands[b].end_freq / bin_spacing.load());
        end_bin = std::min<int>(end_bin, static_cast<int>(FFT_SIZE / 2));

        int total_bins = end_bin - start_bin + 1;
        int step = std::max<int>(total_bins / MAX_CURVE_POINTS, 1);

        foot_btw_amplitudes[b] = (total_bins + step - 1) / step; // Calculate the number of columns

        float sum = 0.0f;
        int count = 0;

        // Process each bin in the frequency band
        for (int i = start_bin; i <= end_bin && count < foot_btw_amplitudes[b]; i += step, ++count)
        {
            float weighted = magnitudes[i] * weights[i];

            if (b * MAX_CURVE_POINTS + count < NUM_BANDS * MAX_CURVE_POINTS)
            {
                flat_btw_amplitudes[b * MAX_CURVE_POINTS + count] = weighted;
            }

            const float peak = std::max<float>(peak_amplitude.load(), static_cast<float>(weighted));
            peak_amplitude.store(peak);

            sum += weighted;
        }

        // Calculate average and compress the result
        float minVal = 0.0f;
        float maxVal = peak_amplitude.load(); // or dynamically updated peak

        float avg = (count > 0) ? (sum / count) : 0.0f;

        //float peak = std::max<float>(peak_amplitude.load(), 1e-6f); // avoid divide by zero
        //float raw_output = avg / peak; // raw output without clamping

        update_recent_max(avg, recent_max.load());  // Track sliding max
        float normalized = apply_adaptive_gain(avg, recent_max.load());
        float normalized2 = ac2_apply_adaptive_gain(avg, smoothed_level, current_gain);
        float visual_value = ac2_compress_dynamic(normalized2); // Optional smoothing/compression

        // Replace compressed with raw_output if you want completely unclamped
        float compressed = avg;

        band_amplitudes[b] = (compressed > 0 ? compressed : 0.0f);

        float current_amp = band_amplitudes[b];
        static float visual_peak = 0.0f;
        static int silent_counter = 0;

        float peak = ac2_track_peak(current_amp, visual_peak, silent_counter);
		peak_amplitude.store(peak);
    }
}

// Prepare FFT properties
void ac2_setup_fft()
{
    fft_input = new float[FFT_SIZE];
    fft_output = new fftwf_complex[FFT_SIZE / 2 + 1];
    magnitudes = new float[FFT_SIZE / 2 + 1];
    sample_buffer = new float[FFT_SIZE];
    pre_computed_window = new float[FFT_SIZE];
    bands = new Band[NUM_BANDS];
    band_amplitudes = new float[NUM_BANDS];

    std::memset(band_amplitudes, 0, NUM_BANDS * sizeof(float));
    std::memset(magnitudes, 0, (FFT_SIZE / 2 + 1) * sizeof(float));
    std::memset(sample_buffer, 0, FFT_SIZE * sizeof(float));

    fft_plan = fftwf_plan_dft_r2c_1d(FFT_SIZE, fft_input, fft_output, FFTW_MEASURE);
    ac2_setup_bands();
    ac2_precompute_window();
}

// Define the missing callback function
void ac2_data_callback(ma_device* pDevice, void* pOutput, const void* pInput, ma_uint32 frameCount)
{
    // Copy input audio data to the sample buffer
    std::memcpy(sample_buffer, pInput, frameCount * sizeof(float));

    // Process the FFT
    ac2_process_fft();
}

// Prepare AudioCore properties
bool ac2_initialize()
{
    ma_device_config deviceConfig;

    // Specify WASAPI backend explicitly
    ma_backend backends[] = {
        ma_backend_wasapi};

    ac2_setup_fft();

    // Configure loopback device
    deviceConfig = ma_device_config_init(ma_device_type_loopback);
    deviceConfig.capture.pDeviceID = NULL; // Use default playback device
    deviceConfig.capture.format = ma_format_f32;
    deviceConfig.capture.channels = 2;
    deviceConfig.sampleRate = SAMPLE_RATE;
    deviceConfig.periodSizeInFrames = 96;
    deviceConfig.periods = 2;
    deviceConfig.dataCallback = ac2_data_callback;

    // Initialize device
    result = ma_device_init_ex(backends, sizeof(backends) / sizeof(backends[0]), NULL, &deviceConfig, &device);
    if (result != MA_SUCCESS)
    {
        return false;
    }
    return true;
}

// Ready State of AudioCore
bool ac2_start()
{
    // Start recording
    result = ma_device_start(&device);
    if (result != MA_SUCCESS)
    {
        ma_device_uninit(&device);
        return false;
    }
    return true;
}

// Clean up resources
void ac2_destroy()
{
    ma_device_uninit(&device);
    fftwf_destroy_plan(fft_plan);
    free_memory();
}

float ac2_get_peak_amplitude()
{
    return peak_amplitude.load();
}

float *ac2_get_band_amplitudes()
{
    return band_amplitudes;
}

float *ac2_get_flat_btw_amplitudes()
{
    return flat_btw_amplitudes;
}

int *ac2_get_foot_btw_amplitudes()
{
    return foot_btw_amplitudes;
}

int ac2_get_num_bands()
{
    return NUM_BANDS;
}