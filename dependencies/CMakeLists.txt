# Dependencies configuration for PBR Material System
cmake_minimum_required(VERSION 3.20)

include(FetchContent)

# Configure FetchContent to use shallow clones for faster downloads
set(FETCHCONTENT_QUIET FALSE)
set(GIT_SHALLOW TRUE)

# GLFW Configuration
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
    glfw
    GIT_REPOSITORY https://github.com/glfw/glfw.git
    GIT_TAG        3.4
    GIT_SHALLOW    TRUE
)

# STB Image - Single header library
FetchContent_Declare(
    stb
    GIT_REPOSITORY https://github.com/nothings/stb.git
    GIT_TAG        master
    GIT_SHALLOW    TRUE
)

# GLAD - OpenGL loader (alternative approach using FetchContent)
FetchContent_Declare(
    glad
    GIT_REPOSITORY https://github.com/Dav1dde/glad.git
    GIT_TAG        v2.0.5
    GIT_SHALLOW    TRUE
)

# Make dependencies available
FetchContent_MakeAvailable(glfw stb glad)

# Create STB interface library
add_library(stb_interface INTERFACE)
target_include_directories(stb_interface INTERFACE ${stb_SOURCE_DIR})

# For systems that need explicit OpenGL extension loading
if(WIN32)
    # Create custom GLAD target for Windows
    set(GLAD_SOURCES 
        ${glad_SOURCE_DIR}/src/gl.c
    )
    
    add_library(glad_lib STATIC ${GLAD_SOURCES})
    target_include_directories(glad_lib PUBLIC ${glad_SOURCE_DIR}/include)
    
    # Link glad to our main target (handled in main CMakeLists.txt)
endif()

# Print dependency information
message(STATUS "=== Dependencies Configuration ===")
message(STATUS "GLFW: ${glfw_SOURCE_DIR}")
message(STATUS "STB: ${stb_SOURCE_DIR}")
message(STATUS "GLAD: ${glad_SOURCE_DIR}")
message(STATUS "==================================")
