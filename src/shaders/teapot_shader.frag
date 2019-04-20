#version 450

layout(binding = 1) buffer UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
} cube_ubo;

layout(binding = 2) buffer Vector3 {
    float x;
    float y;
    float z;
} cam_pos;

layout(location = 0) out vec4 f_color;
layout(location = 2) in vec3 normal;
layout(location = 3) in vec3 frag_pos;

void main() {
    float specularStrength = 0.5;
    vec3 view_pos = vec3(cam_pos.x, cam_pos.y, cam_pos.z);
    vec3 viewDir = normalize(view_pos - frag_pos);
    vec3 objectColor = vec3(1.0f, 0.5f, 0.31f);
    vec3 lightColor = vec3(1.0, 1.0, 1.0);
    float ambientStrength = 0.1;
    vec3 ambient = ambientStrength * lightColor;
    vec3 norm = normalize(normal);
    vec3 lightDir = normalize(vec3(0.0, -5.0, -10.0) - frag_pos);
    vec3 reflectDir = reflect(-lightDir, norm);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32);
    vec3 specular = specularStrength * spec * lightColor;
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = diff * lightColor;
    vec3 result = (ambient + diffuse + specular) * objectColor;
    f_color = vec4(result, 1.0);

}
