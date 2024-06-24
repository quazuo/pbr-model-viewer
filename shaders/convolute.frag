#version 450

#define PI 3.1415926535897932384626433832795

layout(location = 0) in vec3 localPosition;

layout(location = 0) out vec4 outColor;

layout(binding = 0) uniform samplerCube envmapSampler;

void main() {
    vec3 normal = normalize(localPosition);
    vec3 irradiance = vec3(0);
    float delta = 0.025;
    uint nSamples = 0;

    vec3 up = vec3(0.0, 1.0, 0.0);
    vec3 right = normalize(cross(up, normal));
    up = normalize(cross(normal, right));

    for (float phi = 0; phi < 2 * PI; phi += delta) {
        for (float theta = 0; theta < PI / 2; theta += delta) {
            // spherical to cartesian (in tangent space)
            vec3 tangentSample = vec3(
                sin(theta) * cos(phi),
                sin(theta) * sin(phi),
                cos(theta)
            );

            // tangent space to world
            vec3 sampleVec = tangentSample.x * right + tangentSample.y * up + tangentSample.z * normal;

            // add to riemann sum
            irradiance += texture(envmapSampler, sampleVec).rgb * cos(theta) * sin(theta);

            nSamples++;
        }
    }

    irradiance *= PI;
    irradiance /= float(nSamples);

    outColor = vec4(irradiance, 1.0);
}
