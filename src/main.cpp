#include <audiocore2.h>
#include <iostream>
#include <iomanip>
#include <thread>
#include <atomic>
#include <chrono>
#include <string>

std::atomic<bool> keepRunning(true);

void logMessage(const std::string& message) {
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    std::cout << "[" << std::put_time(std::localtime(&time), "%Y-%m-%d %H:%M:%S") << "] " << message << "\n";
}

void showBandAmplitudes() {
    int len = 7;
    float peakAmplitudes = ac2_get_peak_amplitude();
    float* bandAmplitudes = ac2_get_band_amplitudes();

    if (!bandAmplitudes || len <= 0) {
        logMessage("Error: Failed to retrieve band amplitudes.");
        return;
    }

    logMessage("Peak Amplitude");
    std::cout << " Peak: " << std::fixed << std::setprecision(4) << peakAmplitudes << "\n";

    logMessage("Band Amplitudes:");
    for (int i = 0; i < len; ++i) {
        std::cout << "  Band " << i << ": " << std::fixed << std::setprecision(4) << bandAmplitudes[i] << "\n";
    }

    bandAmplitudes = nullptr;
}

void showFlatAndFootAmplitudes() {
    int len = 7;
    float* flatAmplitudes = ac2_get_flat_btw_amplitudes();
    int* footAmplitudes = ac2_get_foot_btw_amplitudes();

    if (!flatAmplitudes || !footAmplitudes || len <= 0) {
        logMessage("Error: Failed to retrieve flat or foot amplitudes.");
        return;
    }

    logMessage("Flat Amplitudes:");
    for (int i = 0; i < len * 100; ++i) {
        std::cout << std::fixed << std::setprecision(4) << flatAmplitudes[i] << " ";
    }
    std::cout << "\n";
    flatAmplitudes = nullptr;

    logMessage("Foot Amplitudes:");
    for (int i = 0; i < len; ++i) {
        std::cout << "  Row " << i << ": " << footAmplitudes[i] << " elements\n";
    }
    footAmplitudes = nullptr;
}

void audioLoop() {
    int iteration = 0;

    while (keepRunning.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10)); // Throttle for readability

        system("cls");

        logMessage("=== Iteration " + std::to_string(iteration++) + " ===");

        // Show band amplitudes
        showBandAmplitudes();
        std::cout << "\n";

        // Show flat and foot amplitudes
        showFlatAndFootAmplitudes();
        std::cout << "\n";

        logMessage("--- End of Iteration ---\n");
    }
}

int main() {
    logMessage("Initializing AudioCore2...");
    if (!ac2_initialize()) {
        logMessage("Failed to initialize AudioCore2.");
        return -1;
    }

    logMessage("Starting AudioCore2...");
    if (!ac2_start()) {
        logMessage("Failed to start AudioCore2.");
        ac2_destroy();
        return -2;
    }

    std::thread audioThread(audioLoop);

    logMessage("Press Enter to stop...");
    std::cin.get();

    keepRunning.store(false);
    if (audioThread.joinable()) {
        audioThread.join();
    }

    logMessage("Destroying AudioCore2...");
    ac2_destroy();
    logMessage("Clean exit.");

    return 0;
}