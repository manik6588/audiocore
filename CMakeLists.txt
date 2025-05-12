cmake_minimum_required(VERSION 3.20)
project(audiocore)

# Set C and C++ standards
set(CMAKE_C_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED YES)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

# Set paths
set(INCLUDE_DIR "${CMAKE_SOURCE_DIR}/include")
set(FFTW_LIBRARY "${CMAKE_SOURCE_DIR}/lib/fftw3f.lib")  # Your static FFTW lib

# Optional: Toggle ASAN
option(ENABLE_ASAN "Enable AddressSanitizer" OFF)

# Compiler flags
if(MSVC)
    add_compile_options(
        /fp:fast   # Fast floating-point
        /GL        # Whole program optimization
        #/RTC1
        #/arch:AVX2 # Use AVX2 SIMD instructions
    )
    add_link_options(/LTCG) # Link-time code generation

    # Enable ASan with dynamic runtime
    
    
endif()

#add_executable(${PROJECT_NAME} src/main.cpp "src/audiocore2.cpp")
# Add your DLL target
add_library(${PROJECT_NAME} SHARED src/audiocore2.cpp)

# Include headers
target_include_directories(${PROJECT_NAME} PRIVATE ${INCLUDE_DIR})

# Link FFTW
target_link_libraries(${PROJECT_NAME} PRIVATE ${FFTW_LIBRARY})

# Export define for Windows DLLs
target_compile_definitions(${PROJECT_NAME} PRIVATE AUDIOCORE_EXPORTS)
