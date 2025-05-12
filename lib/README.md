# Custom Build of FFTW3F (Single Thread & Single Precision)

This repository provides instructions to build your own custom `fftw3f.lib`, the single-precision (`float`) and single-threaded version of the [FFTW](http://www.fftw.org/) library for use in your C/C++ projects.

---

## ⚙️ Build Configuration

We focus on the following FFTW configuration:

- **Precision**: Single (`float`)
- **Threading**: Single-threaded (no OpenMP or pthreads)
- **Library Type**: Static (`.lib` for Windows)
- **Target**: Windows (can be adapted for Linux/macOS)

---

## 🛠️ Prerequisites

- CMake (≥ 3.10)
- A C/C++ compiler (e.g., MSVC, GCC, or Clang)
- [FFTW source code](https://fftw.org/pub/fftw/)

---

## 🔧 CMake Build Instructions

### Step 1: Clone FFTW

```bash
git clone https://github.com/FFTW/fftw3.git
cd fftw3