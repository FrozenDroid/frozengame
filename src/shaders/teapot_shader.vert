#version 450

layout(binding = 0) buffer UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
} ubo;

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normals;
layout(location = 2) out vec3 normals_;
layout(location = 3) out vec3 frag_pos;


void main() {
    gl_Position = ubo.proj * ubo.view * ubo.model * vec4(position, 1.0);
    frag_pos = vec3(ubo.model * vec4(position, 1.0));
    normals_ = normals;
}
