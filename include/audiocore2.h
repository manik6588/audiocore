#pragma once

#ifdef AUDIOCORE_EXPORTS
#define DLL_API __declspec(dllexport)
#else
#define DLL_API __declspec(dllimport)
#endif

#include "miniaudio.h"
#include <vector>
#include <atomic>
#include "fftw3.h" // Assuming FFTW library for FFT

extern "C"
{
    DLL_API bool    ac2_initialize();
    DLL_API bool    ac2_start();
    DLL_API void    ac2_destroy();
    DLL_API void    ac2_process_fft(); // Exported for FFT processing
    DLL_API void    ac2_setup_fft();   // Exported for FFT setup
    DLL_API void    ac2_setup_bands(); // Exported for band setup
    DLL_API void    free_memory();     // Exported for free allocated memory
    DLL_API float   ac2_get_peak_amplitude();
    DLL_API float*  ac2_get_band_amplitudes();
    DLL_API float*  ac2_get_flat_btw_amplitudes();
    DLL_API int*    ac2_get_foot_btw_amplitudes();
    DLL_API int     ac2_get_num_bands();
}