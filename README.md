# AudioCore

AudioCore is a Unity-based audio management system designed to simplify and enhance audio handling in your Unity projects. This project is part of the **Alpha Project** and serves as a core module for managing audio functionalities. It integrates with `miniaudio.h` and `fftw3f` for advanced audio processing.

## Features

- Easy-to-use audio management system.
- Seamless integration with Unity projects.
- Optimized for performance and scalability.
- Leverages `miniaudio.h` for low-level audio handling.
- Utilizes `fftw3f` for fast Fourier transform operations.
- Designed to work as part of the larger **Alpha Project**.

## Requirements

- Unity 2021.3 or later.
- `miniaudio.h` and `fftw3f` libraries must be included in the project.
- This module is intended to be used exclusively within Unity projects.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/manik6588/audiocore.git
    ```
2. Add the `AudioCore` folder to your Unity project.
3. Ensure that the **Alpha Project** dependencies, including `miniaudio.h` and `fftw3f`, are properly set up.

## Usage

1. Import the `AudioCore` namespace in your scripts:
    ```csharp
    using AudioCore;
    ```
2. Use the provided APIs to manage audio playback, volume, and other audio-related features.
3. Leverage the integration with `miniaudio.h` for efficient audio processing and `fftw3f` for advanced audio analysis.

## Credits

- **miniaudio.h**: A single-file audio playback and capture library by David Reid. For more details, visit the [miniaudio GitHub repository](https://github.com/mackron/miniaudio).
- **fftw3f**: A C library for computing the discrete Fourier transform, developed by Matteo Frigo and Steven G. Johnson. For more details, visit the [FFTW official website](http://www.fftw.org).

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Submit a pull request with a detailed description of your changes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## GitHub Page

For more details, visit the [AudioCore GitHub Page](https://github.com/yourusername/audiocore).

---
This project is a part of the **Alpha Project** ecosystem and integrates with `miniaudio.h` and `fftw3f` for enhanced functionality.