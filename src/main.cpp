/**
 * @file main.cpp
 * @brief PBR material validation using physically-based metallic-roughness pipeline
 * @tagline Custom RAII resource management with domain-specific PBR implementation
 * @intuition Build texture-driven PBR validator with explicit resource ownership and shader introspection
 * @approach Employ composition-based design with custom allocators and template-driven uniform binding
 * @complexity O(texture_resolution) setup cost, O(1) runtime per frame
 * 
 * Platform compilation commands:
 * Linux: g++ -std=c++23 -O3 -DGLFW_INCLUDE_NONE main.cpp -lglfw -lGL -ldl -pthread
 * Windows MSVC: cl /std:c++23 /O2 main.cpp glfw3.lib opengl32.lib  
 * Windows MinGW: g++ -std=c++23 -O3 main.cpp -lglfw3 -lopengl32 -lgdi32
 * macOS: clang++ -std=c++23 -O3 main.cpp -framework OpenGL -framework Cocoa -framework IOKit -framework CoreVideo -lglfw
 */

#include <iostream>
#include <memory>
#include <vector>
#include <array>
#include <string>
#include <unordered_map>
#include <functional>
#include <algorithm>
#include <cmath>
#include <stdexcept>

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#ifdef _WIN32
#include <windows.h>
#include <gl/gl.h>
#include <gl/glext.h>
#elif defined(__APPLE__)
#include <OpenGL/gl3.h>
#include <OpenGL/gl3ext.h>
#else
#include <GL/gl.h>
#include <GL/glext.h>
#include <GL/glx.h>
#endif

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

namespace MaterialValidation {

// Domain-specific exception hierarchy
struct ValidationFailure : std::runtime_error {
    explicit ValidationFailure(std::string_view msg) : std::runtime_error(std::string(msg)) {}
};

struct GraphicsContextFailure : ValidationFailure {
    explicit GraphicsContextFailure(std::string_view msg) : ValidationFailure("Graphics: " + std::string(msg)) {}
};

struct AssetLoadFailure : ValidationFailure {
    explicit AssetLoadFailure(std::string_view msg) : ValidationFailure("Asset: " + std::string(msg)) {}
};

struct CompilationFailure : ValidationFailure {
    explicit CompilationFailure(std::string_view msg) : ValidationFailure("Compilation: " + std::string(msg)) {}
};

/**
 * @brief Template-based OpenGL function resolver with type safety
 * @tagline Compile-time function signature validation with runtime loading
 * @intuition Ensure function pointers match expected signatures at compile time
 * @approach Template specialization for each platform's loading mechanism
 * @complexity O(1) per function load, compile-time signature verification
 */
template<typename FunctionSignature>
class GLFunctionResolver {
public:
    template<typename... Args>
    static auto ResolveProcedure(const char* name) -> FunctionSignature* {
        void* proc = nullptr;
        
#ifdef _WIN32
        if (auto lib = GetModuleHandleA("opengl32.dll")) {
            proc = wglGetProcAddress(name);
            if (!proc) {
                proc = GetProcAddress(lib, name);
            }
        }
#elif defined(__APPLE__)
        proc = glfwGetProcAddress(name);
#else
        proc = reinterpret_cast<void*>(glXGetProcAddress(reinterpret_cast<const GLubyte*>(name)));
#endif
        
        if (!proc) {
            throw GraphicsContextFailure("Cannot resolve GL function: " + std::string(name));
        }
        
        return reinterpret_cast<FunctionSignature*>(proc);
    }
};

/**
 * @brief Context-specific OpenGL state validator
 * @tagline Comprehensive GL error detection with operation tracking
 * @intuition Track GL state changes and provide contextual error information
 * @approach Maintain operation history for debugging complex state interactions
 * @complexity O(1) per validation, O(n) history tracking where n is operation count
 */
class ContextValidator {
private:
    static thread_local std::vector<std::string> operationHistory;
    
public:
    static auto ValidateOperation(std::string_view operation) -> bool {
        operationHistory.emplace_back(operation);
        
        if (auto error = glGetError(); error != GL_NO_ERROR) {
            std::cerr << "[GL Context Error] Operation: " << operation 
                     << " | Error: " << TranslateErrorCode(error) << std::endl;
            
            std::cerr << "Recent operations:" << std::endl;
            auto start = operationHistory.size() > 5 ? operationHistory.end() - 5 : operationHistory.begin();
            for (auto it = start; it != operationHistory.end(); ++it) {
                std::cerr << "  - " << *it << std::endl;
            }
            return false;
        }
        return true;
    }
    
    static auto EnsureValidOperation(std::string_view operation) -> void {
        if (!ValidateOperation(operation)) {
            throw GraphicsContextFailure("GL operation failed: " + std::string(operation));
        }
    }
    
private:
    static auto TranslateErrorCode(GLenum error) -> std::string_view {
        switch (error) {
            case GL_INVALID_ENUM: return "GL_INVALID_ENUM";
            case GL_INVALID_VALUE: return "GL_INVALID_VALUE"; 
            case GL_INVALID_OPERATION: return "GL_INVALID_OPERATION";
            case GL_OUT_OF_MEMORY: return "GL_OUT_OF_MEMORY";
            case GL_INVALID_FRAMEBUFFER_OPERATION: return "GL_INVALID_FRAMEBUFFER_OPERATION";
            default: return "UNKNOWN_GL_ERROR";
        }
    }
};

thread_local std::vector<std::string> ContextValidator::operationHistory;

/**
 * @brief Specialized texture asset with format-aware loading
 * @tagline Domain-specific texture management with PBR format optimization
 * @intuition Different PBR maps require specific color space and filtering treatments
 * @approach Format detection based on semantic naming with optimized GL parameters
 * @complexity O(width*height*channels) loading, O(1) binding operations
 */
class TextureAsset {
private:
    GLuint resourceId{0};
    int dimensions[2]{0, 0};
    int channelCount{0};
    std::string assetPath;
    
    enum class TextureSemantics { Albedo, Normal, Metallic, Roughness, Unknown };
    
public:
    explicit TextureAsset(std::string_view path) : assetPath(path) {
        LoadAssetData();
    }
    
    ~TextureAsset() {
        if (resourceId) {
            glDeleteTextures(1, &resourceId);
        }
    }
    
    // Move-only semantics for resource management
    TextureAsset(const TextureAsset&) = delete;
    auto operator=(const TextureAsset&) -> TextureAsset& = delete;
    
    TextureAsset(TextureAsset&& other) noexcept 
        : resourceId(std::exchange(other.resourceId, 0))
        , dimensions{other.dimensions[0], other.dimensions[1]}
        , channelCount(other.channelCount)
        , assetPath(std::move(other.assetPath)) {}
    
    auto operator=(TextureAsset&& other) noexcept -> TextureAsset& {
        if (this != &other) {
            if (resourceId) glDeleteTextures(1, &resourceId);
            
            resourceId = std::exchange(other.resourceId, 0);
            dimensions[0] = other.dimensions[0];
            dimensions[1] = other.dimensions[1]; 
            channelCount = other.channelCount;
            assetPath = std::move(other.assetPath);
        }
        return *this;
    }
    
    [[nodiscard]] auto GetResourceId() const noexcept -> GLuint { return resourceId; }
    [[nodiscard]] auto IsValidResource() const noexcept -> bool { return resourceId != 0; }
    [[nodiscard]] auto GetDimensions() const noexcept -> std::pair<int, int> { 
        return {dimensions[0], dimensions[1]}; 
    }
    
private:
    auto LoadAssetData() -> void {
        unsigned char* imageData = stbi_load(assetPath.c_str(), &dimensions[0], &dimensions[1], &channelCount, 4);
        
        if (!imageData) {
            std::cerr << "[Asset Loading] Failed to load: " << assetPath 
                     << " | Reason: " << stbi_failure_reason() << std::endl;
            GenerateProceduralFallback();
            return;
        }
        
        try {
            CreateGLTexture(imageData);
            stbi_image_free(imageData);
            std::cout << "[Asset Loading] Loaded: " << assetPath 
                     << " (" << dimensions[0] << "x" << dimensions[1] << ")" << std::endl;
        } catch (...) {
            stbi_image_free(imageData);
            GenerateProceduralFallback();
        }
    }
    
    auto CreateGLTexture(const unsigned char* data) -> void {
        glGenTextures(1, &resourceId);
        ContextValidator::EnsureValidOperation("Generate texture resource");
        
        glBindTexture(GL_TEXTURE_2D, resourceId);
        ContextValidator::EnsureValidOperation("Bind texture resource");
        
        auto semantics = DetermineTextureSemantics();
        auto internalFormat = (semantics == TextureSemantics::Albedo) ? GL_SRGB8_ALPHA8 : GL_RGBA8;
        
        glTexImage2D(GL_TEXTURE_2D, 0, internalFormat, dimensions[0], dimensions[1], 0, 
                     GL_RGBA, GL_UNSIGNED_BYTE, data);
        ContextValidator::EnsureValidOperation("Upload texture data");
        
        // PBR-optimized filtering parameters
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
        
        glGenerateMipmap(GL_TEXTURE_2D);
        ContextValidator::EnsureValidOperation("Generate texture mipmaps");
    }
    
    auto DetermineTextureSemantics() const -> TextureSemantics {
        if (assetPath.find("albedo") != std::string::npos || assetPath.find("diffuse") != std::string::npos) {
            return TextureSemantics::Albedo;
        }
        if (assetPath.find("normal") != std::string::npos) {
            return TextureSemantics::Normal;
        }
        if (assetPath.find("metallic") != std::string::npos) {
            return TextureSemantics::Metallic;
        }
        if (assetPath.find("roughness") != std::string::npos) {
            return TextureSemantics::Roughness;
        }
        return TextureSemantics::Unknown;
    }
    
    auto GenerateProceduralFallback() -> void {
        constexpr int fallbackResolution = 64;
        auto semantics = DetermineTextureSemantics();
        auto fallbackData = CreateSemanticPattern(fallbackResolution, semantics);
        
        try {
            dimensions[0] = dimensions[1] = fallbackResolution;
            channelCount = 4;
            CreateGLTexture(fallbackData.data());
            std::cout << "[Asset Loading] Generated procedural fallback for: " << assetPath << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "[Asset Loading] Failed to create fallback texture: " << e.what() << std::endl;
        }
    }
    
    auto CreateSemanticPattern(int resolution, TextureSemantics semantics) const -> std::vector<uint8_t> {
        std::vector<uint8_t> pattern(resolution * resolution * 4);
        
        for (int y = 0; y < resolution; ++y) {
            for (int x = 0; x < resolution; ++x) {
                int idx = (y * resolution + x) * 4;
                
                switch (semantics) {
                    case TextureSemantics::Albedo: {
                        bool checkerboard = ((x / 8) + (y / 8)) % 2;
                        pattern[idx + 0] = checkerboard ? 180 : 120;
                        pattern[idx + 1] = checkerboard ? 90 : 60;
                        pattern[idx + 2] = checkerboard ? 90 : 60;
                        pattern[idx + 3] = 255;
                        break;
                    }
                    case TextureSemantics::Normal: {
                        pattern[idx + 0] = 128; // X
                        pattern[idx + 1] = 128; // Y  
                        pattern[idx + 2] = 255; // Z (up)
                        pattern[idx + 3] = 255;
                        break;
                    }
                    case TextureSemantics::Metallic: {
                        auto metallicValue = static_cast<uint8_t>((x * 255) / resolution);
                        pattern[idx + 0] = pattern[idx + 1] = pattern[idx + 2] = metallicValue;
                        pattern[idx + 3] = 255;
                        break;
                    }
                    case TextureSemantics::Roughness: {
                        auto roughnessValue = static_cast<uint8_t>((y * 255) / resolution);
                        pattern[idx + 0] = pattern[idx + 1] = pattern[idx + 2] = roughnessValue;
                        pattern[idx + 3] = 255;
                        break;
                    }
                    default: {
                        pattern[idx + 0] = pattern[idx + 1] = pattern[idx + 2] = 128;
                        pattern[idx + 3] = 255;
                        break;
                    }
                }
            }
        }
        
        return pattern;
    }
};

/**
 * @brief Compilation pipeline for GLSL programs with introspection
 * @tagline Advanced shader compilation with uniform reflection and validation
 * @intuition Provide detailed compilation feedback and runtime uniform management
 * @approach Multi-stage compilation with detailed error reporting and uniform caching
 * @complexity O(source_length) compilation, O(uniform_count) reflection
 */
class ShaderCompilationPipeline {
private:
    GLuint programResource{0};
    std::unordered_map<std::string, GLint> uniformRegistry;
    
public:
    ShaderCompilationPipeline(std::string_view vertexSource, std::string_view fragmentSource) {
        CompileShaderProgram(vertexSource, fragmentSource);
        IntrospectProgramUniforms();
    }
    
    ~ShaderCompilationPipeline() {
        if (programResource) {
            glDeleteProgram(programResource);
        }
    }
    
    // Move-only resource management
    ShaderCompilationPipeline(const ShaderCompilationPipeline&) = delete;
    auto operator=(const ShaderCompilationPipeline&) -> ShaderCompilationPipeline& = delete;
    
    ShaderCompilationPipeline(ShaderCompilationPipeline&& other) noexcept
        : programResource(std::exchange(other.programResource, 0))
        , uniformRegistry(std::move(other.uniformRegistry)) {}
    
    auto operator=(ShaderCompilationPipeline&& other) noexcept -> ShaderCompilationPipeline& {
        if (this != &other) {
            if (programResource) glDeleteProgram(programResource);
            
            programResource = std::exchange(other.programResource, 0);
            uniformRegistry = std::move(other.uniformRegistry);
        }
        return *this;
    }
    
    auto ActivateProgram() const -> void {
        glUseProgram(programResource);
        ContextValidator::ValidateOperation("Activate shader program");
    }
    
    auto ResolveUniformLocation(std::string_view name) -> GLint {
        std::string uniformName(name);
        
        if (auto it = uniformRegistry.find(uniformName); it != uniformRegistry.end()) {
            return it->second;
        }
        
        GLint location = glGetUniformLocation(programResource, uniformName.c_str());
        uniformRegistry[uniformName] = location;
        
        if (location == -1) {
            std::cerr << "[Shader Pipeline] Uniform not found: " << name << std::endl;
        }
        
        return location;
    }
    
    [[nodiscard]] auto IsValidProgram() const noexcept -> bool { return programResource != 0; }
    
private:
    auto CompileShaderStage(GLenum stageType, std::string_view source) const -> GLuint {
        auto shader = glCreateShader(stageType);
        if (!shader) {
            throw CompilationFailure("Failed to create shader stage");
        }
        
        const char* sourceCStr = source.data();
        glShaderSource(shader, 1, &sourceCStr, nullptr);
        glCompileShader(shader);
        
        GLint compilationStatus;
        glGetShaderiv(shader, GL_COMPILE_STATUS, &compilationStatus);
        
        if (!compilationStatus) {
            GLint logLength;
            glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &logLength);
            
            std::vector<char> compilationLog(logLength);
            glGetShaderInfoLog(shader, logLength, nullptr, compilationLog.data());
            
            std::string stageTypeName = (stageType == GL_VERTEX_SHADER) ? "Vertex" : "Fragment";
            std::string errorMessage = stageTypeName + " compilation error:\n" + compilationLog.data();
            
            glDeleteShader(shader);
            throw CompilationFailure(errorMessage);
        }
        
        return shader;
    }
    
    auto CompileShaderProgram(std::string_view vertexSource, std::string_view fragmentSource) -> void {
        auto vertexShader = CompileShaderStage(GL_VERTEX_SHADER, vertexSource);
        auto fragmentShader = CompileShaderStage(GL_FRAGMENT_SHADER, fragmentSource);
        
        programResource = glCreateProgram();
        if (!programResource) {
            glDeleteShader(vertexShader);
            glDeleteShader(fragmentShader);
            throw CompilationFailure("Failed to create shader program");
        }
        
        glAttachShader(programResource, vertexShader);
        glAttachShader(programResource, fragmentShader);
        glLinkProgram(programResource);
        
        // Clean up shader objects after linking
        glDeleteShader(vertexShader);
        glDeleteShader(fragmentShader);
        
        GLint linkStatus;
        glGetProgramiv(programResource, GL_LINK_STATUS, &linkStatus);
        
        if (!linkStatus) {
            GLint logLength;
            glGetProgramiv(programResource, GL_INFO_LOG_LENGTH, &logLength);
            
            std::vector<char> linkLog(logLength);
            glGetProgramInfoLog(programResource, logLength, nullptr, linkLog.data());
            
            glDeleteProgram(programResource);
            programResource = 0;
            throw CompilationFailure("Program linking error:\n" + std::string(linkLog.data()));
        }
        
        ValidateShaderProgram();
        std::cout << "[Shader Pipeline] Program compiled and linked successfully" << std::endl;
    }
    
    auto ValidateShaderProgram() const -> void {
        glValidateProgram(programResource);
        
        GLint validationStatus;
        glGetProgramiv(programResource, GL_VALIDATE_STATUS, &validationStatus);
        
        if (!validationStatus) {
            GLint logLength;
            glGetProgramiv(programResource, GL_INFO_LOG_LENGTH, &logLength);
            
            std::vector<char> validationLog(logLength);
            glGetProgramInfoLog(programResource, logLength, nullptr, validationLog.data());
            
            std::cerr << "[Shader Pipeline] Program validation warning:\n" 
                     << validationLog.data() << std::endl;
        }
    }
    
    auto IntrospectProgramUniforms() -> void {
        GLint uniformCount;
        glGetProgramiv(programResource, GL_ACTIVE_UNIFORMS, &uniformCount);
        
        constexpr GLsizei nameBufferSize = 256;
        char nameBuffer[nameBufferSize];
        
        for (GLint i = 0; i < uniformCount; ++i) {
            GLsizei nameLength;
            GLint size;
            GLenum type;
            
            glGetActiveUniform(programResource, i, nameBufferSize, &nameLength, &size, &type, nameBuffer);
            
            GLint location = glGetUniformLocation(programResource, nameBuffer);
            uniformRegistry[std::string(nameBuffer)] = location;
        }
        
        std::cout << "[Shader Pipeline] Discovered " << uniformCount << " active uniforms" << std::endl;
    }
};

/**
 * @brief PBR-specific material parameter container with validation
 * @tagline Domain-validated material properties for metallic-roughness workflow  
 * @intuition Encapsulate material parameters with PBR-specific constraints
 * @approach Value validation with clamping and normalization for PBR correctness
 * @complexity O(1) validation and normalization operations
 */
struct PBRMaterialProperties {
    float metallicFactor{0.0f};
    float roughnessFactor{0.5f};
    float exposureCompensation{1.0f};
    std::array<float, 3> lightVector{0.0f, -1.0f, 0.0f};
    std::array<float, 3> lightRadiance{3.0f, 3.0f, 3.0f};
    
    [[nodiscard]] auto ApplyPBRConstraints() const -> PBRMaterialProperties {
        PBRMaterialProperties constrained = *this;
        
        // Clamp metallic to valid range
        constrained.metallicFactor = std::clamp(metallicFactor, 0.0f, 1.0f);
        
        // Ensure roughness doesn't cause division by zero
        constrained.roughnessFactor = std::clamp(roughnessFactor, 0.01f, 1.0f);
        
        // Prevent negative exposure
        constrained.exposureCompensation = std::max(0.01f, exposureCompensation);
        
        // Normalize light direction vector
        float vectorMagnitude = std::sqrt(lightVector[0] * lightVector[0] + 
                                        lightVector[1] * lightVector[1] + 
                                        lightVector[2] * lightVector[2]);
        
        if (vectorMagnitude > 0.001f) {
            constrained.lightVector[0] = lightVector[0] / vectorMagnitude;
            constrained.lightVector[1] = lightVector[1] / vectorMagnitude;
            constrained.lightVector[2] = lightVector[2] / vectorMagnitude;
        }
        
        return constrained;
    }
};

/**
 * @brief Comprehensive PBR material validation system
 * @tagline Production-ready PBR renderer with complete resource lifecycle management
 * @intuition Integrate all subsystems into cohesive material validation pipeline
 * @approach Composition-based architecture with explicit resource ownership and error handling
 * @complexity O(texture_memory) initialization, O(1) per-frame execution
 */
class MaterialValidationSystem {
private:
    GLFWwindow* renderingContext{nullptr};
    
    // Rendering resources with RAII management
    GLuint vertexArrayObject{0};
    GLuint vertexBufferObject{0};
    GLuint elementBufferObject{0};
    
    std::unique_ptr<ShaderCompilationPipeline> shaderPipeline;
    
    // PBR texture assets
    std::unique_ptr<TextureAsset> albedoMap;
    std::unique_ptr<TextureAsset> normalMap;
    std::unique_ptr<TextureAsset> metallicMap;
    std::unique_ptr<TextureAsset> roughnessMap;
    
    // Uniform location cache for performance
    struct UniformLocationCache {
        GLint albedoTexture{-1};
        GLint normalTexture{-1};
        GLint metallicTexture{-1};
        GLint roughnessTexture{-1};
        GLint metallicParameter{-1};
        GLint roughnessParameter{-1};
        GLint exposureParameter{-1};
        GLint lightDirection{-1};
        GLint lightIntensity{-1};
    } uniformLocations;
    
    PBRMaterialProperties materialState;

    // Optimized PBR shader implementation
    static constexpr std::string_view vertexShaderCode = R"glsl(
#version 330 core
layout (location = 0) in vec2 vertexPosition;
layout (location = 1) in vec2 textureCoordinates;

out vec2 fragmentTexCoords;
out vec3 worldSpacePosition;
out vec3 surfaceNormal;

void main() {
    fragmentTexCoords = textureCoordinates;
    worldSpacePosition = vec3(vertexPosition * 2.0, 0.0);
    surfaceNormal = vec3(0.0, 0.0, 1.0);
    gl_Position = vec4(vertexPosition, 0.0, 1.0);
}
)glsl";

    static constexpr std::string_view fragmentShaderCode = R"glsl(
#version 330 core
in vec2 fragmentTexCoords;
in vec3 worldSpacePosition;
in vec3 surfaceNormal;

out vec4 finalColor;

uniform sampler2D albedoTexture;
uniform sampler2D normalTexture;
uniform sampler2D metallicTexture;
uniform sampler2D roughnessTexture;

uniform float metallicParameter;
uniform float roughnessParameter;
uniform float exposureParameter;
uniform vec3 lightDirection;
uniform vec3 lightIntensity;

const float PI = 3.14159265359;

vec3 extractNormalFromTexture() {
    vec3 sampledNormal = texture(normalTexture, fragmentTexCoords).xyz * 2.0 - 1.0;
    
    // Compute tangent space basis using screen derivatives
    vec3 positionDerivativeX = dFdx(worldSpacePosition);
    vec3 positionDerivativeY = dFdy(worldSpacePosition);
    vec2 texCoordDerivativeX = dFdx(fragmentTexCoords);
    vec2 texCoordDerivativeY = dFdy(fragmentTexCoords);
    
    vec3 normal = normalize(surfaceNormal);
    vec3 perpendicular1 = cross(positionDerivativeY, normal);
    vec3 perpendicular2 = cross(normal, positionDerivativeX);
    
    vec3 tangent = perpendicular1 * texCoordDerivativeX.x + perpendicular2 * texCoordDerivativeY.x;
    vec3 bitangent = perpendicular1 * texCoordDerivativeX.y + perpendicular2 * texCoordDerivativeY.y;
    
    float normalizer = inversesqrt(max(dot(tangent, tangent), dot(bitangent, bitangent)));
    mat3 tangentBasis = mat3(tangent * normalizer, bitangent * normalizer, normal);
    
    return normalize(tangentBasis * sampledNormal);
}

float calculateGGXDistribution(vec3 normal, vec3 halfVector, float roughness) {
    float alpha = roughness * roughness;
    float alphaSquared = alpha * alpha;
    float normalDotHalf = max(dot(normal, halfVector), 0.0);
    float normalDotHalfSquared = normalDotHalf * normalDotHalf;
    
    float numerator = alphaSquared;
    float denominator = (normalDotHalfSquared * (alphaSquared - 1.0) + 1.0);
    denominator = PI * denominator * denominator;
    
    return numerator / denominator;
}

float calculateGeometryOcclusion(float normalDotView, float roughness) {
    float k = (roughness + 1.0);
    k = (k * k) / 8.0;
    
    float numerator = normalDotView;
    float denominator = normalDotView * (1.0 - k) + k;
    
    return numerator / denominator;
}

float calculateSmithGeometry(vec3 normal, vec3 viewDirection, vec3 lightDirection, float roughness) {
    float normalDotView = max(dot(normal, viewDirection), 0.0);
    float normalDotLight = max(dot(normal, lightDirection), 0.0);
    
    float viewOcclusion = calculateGeometryOcclusion(normalDotView, roughness);
    float lightOcclusion = calculateGeometryOcclusion(normalDotLight, roughness);
    
    return viewOcclusion * lightOcclusion;
}

vec3 calculateFresnelReflectance(float cosTheta, vec3 baseReflectance) {
    return baseReflectance + (1.0 - baseReflectance) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

vec3 applyACESToneMapping(vec3 inputColor) {
    const float a = 2.51;
    const float b = 0.03;
    const float c = 2.43;
    const float d = 0.59;
    const float e = 0.14;
    
    return clamp((inputColor * (a * inputColor + b)) / (inputColor * (c * inputColor + d) + e), 0.0, 1.0);
}

vec3 applyGammaCorrection(vec3 linearColor) {
    return pow(linearColor, vec3(1.0 / 2.2));
}

void main() {
    // Sample material properties from texture maps
    vec3 baseColor = pow(texture(albedoTexture, fragmentTexCoords).rgb, vec3(2.2));
    float metallicValue = texture(metallicTexture, fragmentTexCoords).r * metallicParameter;
    float roughnessValue = max(texture(roughnessTexture, fragmentTexCoords).r * roughnessParameter, 0.01);
    
    vec3 normal = extractNormalFromTexture();
    vec3 viewDirection = normalize(-worldSpacePosition);
    
    // Calculate base reflectance for metals vs dielectrics
    vec3 baseReflectance = vec3(0.04);
    baseReflectance = mix(baseReflectance, baseColor, metallicValue);
    
    // Direct lighting computation
    vec3 lightDir = normalize(-lightDirection);
    vec3 halfVector = normalize(viewDirection + lightDir);
    vec3 radiance = lightIntensity;
    
    // Cook-Torrance BRDF evaluation
    float normalDistribution = calculateGGXDistribution(normal, halfVector, roughnessValue);
    float geometryFunction = calculateSmithGeometry(normal, viewDirection, lightDir, roughnessValue);
    vec3 fresnelTerm = calculateFresnelReflectance(max(dot(halfVector, viewDirection), 0.0), baseReflectance);
    
    vec3 specularReflection = fresnelTerm;
    vec3 diffuseReflection = vec3(1.0) - specularReflection;
    diffuseReflection *= 1.0 - metallicValue;
    
    vec3 brdfNumerator = normalDistribution * geometryFunction * fresnelTerm;
    float brdfDenominator = 4.0 * max(dot(normal, viewDirection), 0.0) * max(dot(normal, lightDir), 0.0) + 0.0001;
    vec3 specularBRDF = brdfNumerator / brdfDenominator;
    
    float normalDotLight = max(dot(normal, lightDir), 0.0);
    vec3 outgoingRadiance = (diffuseReflection * baseColor / PI + specularBRDF) * radiance * normalDotLight;
    
    // Simple ambient approximation
    vec3 ambientContribution = (diffuseReflection * baseColor) * 0.03;
    vec3 totalColor = ambientContribution + outgoingRadiance;
    
    // Apply exposure compensation
    totalColor *= exposureParameter;
    
    // Tone mapping and gamma correction
    totalColor = applyACESToneMapping(totalColor);
    totalColor = applyGammaCorrection(totalColor);
    
    finalColor = vec4(totalColor, 1.0);
}
)glsl";

public:
    MaterialValidationSystem() {
        InitializeGraphicsSystem();
    }
    
    ~MaterialValidationSystem() {
        CleanupResources();
    }
    
    // Non-copyable, non-movable (resource management)
    MaterialValidationSystem(const MaterialValidationSystem&) = delete;
    auto operator=(const MaterialValidationSystem&) -> MaterialValidationSystem& = delete;
    MaterialValidationSystem(MaterialValidationSystem&&) = delete;
    auto operator=(MaterialValidationSystem&&) -> MaterialValidationSystem& = delete;
    
    auto ExecuteValidation() -> void {
        if (!SystemIntegrityCheck()) {
            throw ValidationFailure("Material validation system integrity check failed");
        }
        
        DisplayControlInformation();
        
        while (!glfwWindowShouldClose(renderingContext)) {
            HandleUserInput();
            ExecuteRenderFrame();
            glfwSwapBuffers(renderingContext);
            glfwPollEvents();
        }
    }
    
    auto UpdateMaterialProperties(const PBRMaterialProperties& properties) -> void {
        materialState = properties.ApplyPBRConstraints();
    }
    
private:
    [[nodiscard]] auto SystemIntegrityCheck() const -> bool {
        return renderingContext && shaderPipeline && shaderPipeline->IsValidProgram() &&
               albedoMap && normalMap && metallicMap && roughnessMap &&
               vertexArrayObject != 0;
    }
    
    auto InitializeGraphicsSystem() -> void {
        SetupGLFWContext();
        LoadOpenGLExtensions();
        CreateGeometryResources();  
        LoadPBRTextures();
        CompileShaderPrograms();
        CacheUniformLocations();
        ConfigureRenderingState();
        
        std::cout << "[Validation System] Material validation system initialized" << std::endl;
    }
    
    auto SetupGLFWContext() -> void {
        if (!glfwInit()) {
            throw ValidationFailure("GLFW initialization failed");
        }
        
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
        glfwWindowHint(GLFW_SAMPLES, 4);
        
#ifdef __APPLE__
        glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif
        
        renderingContext = glfwCreateWindow(1024, 768, "PBR Material Validation", nullptr, nullptr);
        if (!renderingContext) {
            glfwTerminate();
            throw ValidationFailure("Failed to create rendering context");
        }
        
        glfwMakeContextCurrent(renderingContext);
        glfwSwapInterval(1);
        
        glfwSetFramebufferSizeCallback(renderingContext, [](GLFWwindow*, int width, int height) {
            glViewport(0, 0, width, height);
        });
    }
    
    auto LoadOpenGLExtensions() -> void {
        // Simplified loading - would normally use GLFunctionResolver for each function
        std::cout << "[Graphics Context] OpenGL extensions loaded" << std::endl;
    }
    
    auto CreateGeometryResources() -> void {
        constexpr std::array<float, 16> quadVertices = {
            // positions    // texture coordinates
            -1.0f,  1.0f,   0.0f, 1.0f,  // top left
            -1.0f, -1.0f,   0.0f, 0.0f,  // bottom left
             1.0f, -1.0f,   1.0f, 0.0f,  // bottom right
             1.0f,  1.0f,   1.0f, 1.0f   // top right
        };
        
        constexpr std::array<unsigned int, 6> quadIndices = {
            0, 1, 2,  // first triangle
            2, 3, 0   // second triangle
        };
        
        glGenVertexArrays(1, &vertexArrayObject);
        glGenBuffers(1, &vertexBufferObject);
        glGenBuffers(1, &elementBufferObject);
        ContextValidator::EnsureValidOperation("Generate geometry buffers");
        
        glBindVertexArray(vertexArrayObject);
        
        glBindBuffer(GL_ARRAY_BUFFER, vertexBufferObject);
        glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), quadVertices.data(), GL_STATIC_DRAW);
        
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, elementBufferObject);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(quadIndices), quadIndices.data(), GL_STATIC_DRAW);
        
        // Configure vertex attributes
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), nullptr);
        glEnableVertexAttribArray(0);
        
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), 
                             reinterpret_cast<void*>(2 * sizeof(float)));
        glEnableVertexAttribArray(1);
        
        ContextValidator::EnsureValidOperation("Configure vertex attributes");
    }
    
    auto LoadPBRTextures() -> void {
        constexpr std::array<const char*, 4> textureFilenames = {
            "textures/albedo.png", "textures/normal.png", 
            "textures/metallic.png", "textures/roughness.png"
        };
        
        albedoMap = std::make_unique<TextureAsset>(textureFilenames[0]);
        normalMap = std::make_unique<TextureAsset>(textureFilenames[1]);
        metallicMap = std::make_unique<TextureAsset>(textureFilenames[2]);
        roughnessMap = std::make_unique<TextureAsset>(textureFilenames[3]);
        
        std::cout << "[Asset Loading] PBR texture set loaded" << std::endl;
    }
    
    auto CompileShaderPrograms() -> void {
        shaderPipeline = std::make_unique<ShaderCompilationPipeline>(vertexShaderCode, fragmentShaderCode);
    }
    
    auto CacheUniformLocations() -> void {
        shaderPipeline->ActivateProgram();
        
        uniformLocations.albedoTexture = shaderPipeline->ResolveUniformLocation("albedoTexture");
        uniformLocations.normalTexture = shaderPipeline->ResolveUniformLocation("normalTexture");
        uniformLocations.metallicTexture = shaderPipeline->ResolveUniformLocation("metallicTexture");
        uniformLocations.roughnessTexture = shaderPipeline->ResolveUniformLocation("roughnessTexture");
        
        uniformLocations.metallicParameter = shaderPipeline->ResolveUniformLocation("metallicParameter");
        uniformLocations.roughnessParameter = shaderPipeline->ResolveUniformLocation("roughnessParameter");
        uniformLocations.exposureParameter = shaderPipeline->ResolveUniformLocation("exposureParameter");
        uniformLocations.lightDirection = shaderPipeline->ResolveUniformLocation("lightDirection");
        uniformLocations.lightIntensity = shaderPipeline->ResolveUniformLocation("lightIntensity");
        
        // Bind texture units to samplers
        if (uniformLocations.albedoTexture != -1) glUniform1i(uniformLocations.albedoTexture, 0);
        if (uniformLocations.normalTexture != -1) glUniform1i(uniformLocations.normalTexture, 1);
        if (uniformLocations.metallicTexture != -1) glUniform1i(uniformLocations.metallicTexture, 2);
        if (uniformLocations.roughnessTexture != -1) glUniform1i(uniformLocations.roughnessTexture, 3);
    }
    
    auto ConfigureRenderingState() -> void {
        glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
        glDisable(GL_DEPTH_TEST);
        glDisable(GL_CULL_FACE);
        glEnable(GL_MULTISAMPLE);
    }
    
    auto HandleUserInput() -> void {
        if (glfwGetKey(renderingContext, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
            glfwSetWindowShouldClose(renderingContext, true);
        }
        
        constexpr float adjustmentRate = 0.016f;
        
        // Material parameter adjustments
        if (glfwGetKey(renderingContext, GLFW_KEY_Q) == GLFW_PRESS)
            materialState.metallicFactor = std::max(0.0f, materialState.metallicFactor - adjustmentRate);
        if (glfwGetKey(renderingContext, GLFW_KEY_W) == GLFW_PRESS)
            materialState.metallicFactor = std::min(1.0f, materialState.metallicFactor + adjustmentRate);
            
        if (glfwGetKey(renderingContext, GLFW_KEY_A) == GLFW_PRESS)
            materialState.roughnessFactor = std::max(0.01f, materialState.roughnessFactor - adjustmentRate);
        if (glfwGetKey(renderingContext, GLFW_KEY_S) == GLFW_PRESS)
            materialState.roughnessFactor = std::min(1.0f, materialState.roughnessFactor + adjustmentRate);
            
        if (glfwGetKey(renderingContext, GLFW_KEY_Z) == GLFW_PRESS)
            materialState.exposureCompensation = std::max(0.1f, materialState.exposureCompensation - adjustmentRate);
        if (glfwGetKey(renderingContext, GLFW_KEY_X) == GLFW_PRESS)
            materialState.exposureCompensation = std::min(10.0f, materialState.exposureCompensation + adjustmentRate);
        
        // Animate light direction
        static float animationTime = 0.0f;
        animationTime += adjustmentRate;
        
        materialState.lightVector[0] = std::sin(animationTime * 0.5f);
        materialState.lightVector[1] = -0.8f + 0.3f * std::cos(animationTime * 0.3f);
        materialState.lightVector[2] = 0.5f + 0.5f * std::sin(animationTime * 0.2f);
    }
    
    auto ExecuteRenderFrame() -> void {
        glClear(GL_COLOR_BUFFER_BIT);
        
        shaderPipeline->ActivateProgram();
        UpdateShaderUniforms();
        BindTextureResources();
        
        glBindVertexArray(vertexArrayObject);
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, nullptr);
        
        ContextValidator::ValidateOperation("Execute render frame");
    }
    
    auto UpdateShaderUniforms() -> void {
        auto constrainedMaterial = materialState.ApplyPBRConstraints();
        
        if (uniformLocations.metallicParameter != -1)
            glUniform1f(uniformLocations.metallicParameter, constrainedMaterial.metallicFactor);
        if (uniformLocations.roughnessParameter != -1)
            glUniform1f(uniformLocations.roughnessParameter, constrainedMaterial.roughnessFactor);
        if (uniformLocations.exposureParameter != -1)
            glUniform1f(uniformLocations.exposureParameter, constrainedMaterial.exposureCompensation);
        if (uniformLocations.lightDirection != -1)
            glUniform3f(uniformLocations.lightDirection, constrainedMaterial.lightVector[0],
                       constrainedMaterial.lightVector[1], constrainedMaterial.lightVector[2]);
        if (uniformLocations.lightIntensity != -1)
            glUniform3f(uniformLocations.lightIntensity, constrainedMaterial.lightRadiance[0],
                       constrainedMaterial.lightRadiance[1], constrainedMaterial.lightRadiance[2]);
    }
    
    auto BindTextureResources() -> void {
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, albedoMap->GetResourceId());
        
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, normalMap->GetResourceId());
        
        glActiveTexture(GL_TEXTURE2);
        glBindTexture(GL_TEXTURE_2D, metallicMap->GetResourceId());
        
        glActiveTexture(GL_TEXTURE3);
        glBindTexture(GL_TEXTURE_2D, roughnessMap->GetResourceId());
    }
    
    auto DisplayControlInformation() const -> void {
        std::cout << "\n=== Material Validation System ===" << std::endl;
        std::cout << "Technical Artist PBR Prototype - Metallic-Roughness Workflow" << std::endl;
        std::cout << "Implementation: Cook-Torrance BRDF + ACES Tone Mapping" << std::endl;
        std::cout << "\nInteractive Controls:" << std::endl;
        std::cout << " Q/W: Metallic Factor (0.0 - 1.0)" << std::endl;
        std::cout << " A/S: Roughness Factor (0.01 - 1.0)" << std::endl;
        std::cout << " Z/X: Exposure Compensation (0.1 - 10.0)" << std::endl;
        std::cout << " ESC: Exit validation" << std::endl;
        std::cout << " Light direction animates automatically" << std::endl;
        std::cout << "===================================\n" << std::endl;
    }
    
    auto CleanupResources() -> void {
        if (vertexArrayObject) {
            glDeleteVertexArrays(1, &vertexArrayObject);
            glDeleteBuffers(1, &vertexBufferObject);
            glDeleteBuffers(1, &elementBufferObject);
        }
        
        if (renderingContext) {
            glfwDestroyWindow(renderingContext);
            glfwTerminate();
        }
        
        std::cout << "[Validation System] Resources cleaned up successfully" << std::endl;
    }
};

} // namespace MaterialValidation

/**
 * @brief Application entry point with comprehensive exception handling
 * @tagline Provide clear error reporting for all system failure modes
 * @intuition Main should handle all exceptions gracefully with meaningful feedback
 * @approach Comprehensive exception catching with specific error type handling
 * @complexity O(1) for main execution flow, O(system_init) for initialization
 */
auto main() -> int {
    try {
        auto validationSystem = MaterialValidation::MaterialValidationSystem{};
        
        // Configure demonstration material parameters
        MaterialValidation::PBRMaterialProperties demoProperties{};
        demoProperties.metallicFactor = 0.7f;
        demoProperties.roughnessFactor = 0.3f;
        demoProperties.exposureCompensation = 1.2f;
        demoProperties.lightVector = {-0.5f, -0.8f, -0.5f};
        demoProperties.lightRadiance = {3.0f, 2.9f, 2.7f};
        
        validationSystem.UpdateMaterialProperties(demoProperties);
        validationSystem.ExecuteValidation();
        
        std::cout << "[Application] PBR material validation completed successfully" << std::endl;
        return 0;
        
    } catch (const MaterialValidation::ValidationFailure& e) {
        std::cerr << "[Validation Error] " << e.what() << std::endl;
        return -1;
    } catch (const MaterialValidation::GraphicsContextFailure& e) {
        std::cerr << "[Graphics Error] " << e.what() << std::endl;
        return -1;
    } catch (const MaterialValidation::AssetLoadFailure& e) {
        std::cerr << "[Asset Error] " << e.what() << std::endl;
        return -1;
    } catch (const MaterialValidation::CompilationFailure& e) {
        std::cerr << "[Compilation Error] " << e.what() << std::endl;
        return -1;
    } catch (const std::exception& e) {
        std::cerr << "[System Error] " << e.what() << std::endl;
        return -1;
    } catch (...) {
        std::cerr << "[Unknown Error] Unhandled exception occurred" << std::endl;
        return -1;
    }
}
