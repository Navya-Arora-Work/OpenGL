cmake_minimum_required(VERSION 3.20)

project(PBRMaterialSystem 
    VERSION 1.0.0
    DESCRIPTION "PBR Material Validation System with Metallic-Roughness Workflow"
    LANGUAGES CXX)

# Set C++23 standard with modern CMake
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CXX_EXTENSIONS OFF)

# Enable modern CMake policies
cmake_policy(SET CMP0077 NEW)  # option() honors normal variables
cmake_policy(SET CMP0079 NEW)  # target_link_libraries() allows use with targets in other directories

# Build configuration
set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type")
set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "RelWithDebInfo" "MinSizeRel")

# Platform-specific compiler flags
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    # GCC/Clang warnings and optimizations
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic -Wconversion")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -DNDEBUG -march=native")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g3 -fsanitize=address -fsanitize=undefined")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    # MSVC specific flags
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4 /permissive-")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2 /DNDEBUG")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /Od /Zi /RTC1")
endif()

# Dependencies directory
set(DEPS_DIR "${CMAKE_SOURCE_DIR}/dependencies")
add_subdirectory(${DEPS_DIR})

# Source files
set(SOURCES
    src/main.cpp
)

# Create executable
add_executable(${PROJECT_NAME} ${SOURCES})

# Link libraries with modern CMake syntax
target_link_libraries(${PROJECT_NAME} PRIVATE
    glfw
    ${CMAKE_DL_LIBS}  # For dynamic library loading
    $<$<PLATFORM_ID:Linux>:GL>
    $<$<PLATFORM_ID:Windows>:opengl32>
    $<$<PLATFORM_ID:Darwin>:-framework OpenGL -framework Cocoa -framework IOKit -framework CoreVideo>
)

# Include directories for headers
target_include_directories(${PROJECT_NAME} PRIVATE
    ${DEPS_DIR}/stb
    ${DEPS_DIR}/glad/include
)

# Platform-specific threading
find_package(Threads REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE Threads::Threads)

# Enable parallel compilation on MSVC
if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    target_compile_options(${PROJECT_NAME} PRIVATE /MP)
endif()

# Create texture directory structure
file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/textures")

# Copy example textures if they exist
if(EXISTS "${CMAKE_SOURCE_DIR}/textures")
    file(COPY "${CMAKE_SOURCE_DIR}/textures/" DESTINATION "${CMAKE_BINARY_DIR}/textures/")
endif()

# Print configuration summary
message(STATUS "=== PBR Material System Configuration ===")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Compiler: ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "System: ${CMAKE_SYSTEM_NAME}")
message(STATUS "Architecture: ${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "==========================================")
