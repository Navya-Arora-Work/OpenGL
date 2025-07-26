# PBR Material Validation System

A comprehensive C++23 OpenGL application for validating Physically Based Rendering (PBR) workflows using the metallic-roughness material model.

## Features

- **Complete PBR Implementation**: Cook-Torrance BRDF with GGX normal distribution, Smith geometry function, and Schlick Fresnel approximation
- **Metallic-Roughness Workflow**: Industry-standard PBR material pipeline
- **Texture Support**: Albedo, normal, metallic, and roughness texture maps with STB Image
- **Tone Mapping & Gamma Correction**: Reinhard tone mapping with proper gamma correction
- **Cross-Platform**: Windows, Linux, and macOS support
- **Modern C++23**: Utilizes latest C++ features and best practices
- **Automatic Fallbacks**: Procedural textures when files are missing

## Technical Specifications

- **Language**: C++23
- **Graphics API**: OpenGL 3.3 Core Profile
- **Windowing**: GLFW 3.4
- **Image Loading**: STB Image
- **Build System**: CMake 3.20+
- **Supported Compilers**: MSVC 2019+, GCC 10+, Clang 12+

## Architecture

The system follows clean architecture principles with:

- **Modular Design**: Separate classes for texture management, shader compilation, and rendering
- **RAII Resource Management**: Automatic cleanup of OpenGL resources
- **Error Handling**: Comprehensive error reporting for shaders and textures
- **Performance Optimized**: Minimal state changes and efficient rendering

## Build Instructions

### Prerequisites

- **Windows**: Visual Studio 2019+ or MinGW-w64
- **Linux**: GCC 10+ or Clang 12+, OpenGL development libraries
- **macOS**: Xcode Command Line Tools

### Quick Start

#### Windows
