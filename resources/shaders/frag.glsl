#version 330
uniform vec3 camera_position;
in vec3 v_position;
in vec3 v_normal;
out vec3 f_diffuse;
out vec3 f_specular;
out float f_matte;
struct Material
{
    vec3 diffuse_albedo;
    vec3 specular_albedo;
    float shininess;
    float translucency;
    vec3 sigma_t;
};
uniform Material material;
struct DirLight
{
    vec3 direction;
    vec3 emission;
};
uniform sampler2D primary_directional_light_shadow_map;
uniform mat4 primary_directional_light_view_projection;
uniform float primary_directional_light_far;
uniform int num_directional_lights;
uniform DirLight directional_lights[16];
struct PointLight {
    vec3 position;
    vec3 emission;
    vec3 attenuation;
};
//uniform samplerCube primary_point_shadow_map; // point shadow can be handled only by using cubemaps, otherwise we need 6 passes to render
uniform int num_point_lights;
uniform PointLight point_lights[16];
void main() {
    vec3 position = v_position;
    vec3 normal = normalize(v_normal);
    vec3 view_dir = normalize(camera_position - position);
    f_diffuse = vec3(0.0);
    f_specular = vec3(0.0);
    f_matte = 1.0;
    if(num_directional_lights > 0)
    {
        DirLight light = directional_lights[0];
        vec3 light_dir = -normalize(light.direction);
        float diff = min(max(dot(normal, light_dir) + material.translucency, 0.0), 1.0);
        vec3 half_dir = normalize(light_dir + view_dir);
        float spec = pow(max(0.0, dot(normal, half_dir)), material.shininess);
        vec4 light_space_position =
            primary_directional_light_view_projection *
            vec4(position - 0.03 * normal, 1.0);
        light_space_position.xyz /= light_space_position.w;
        light_space_position = 0.5 * light_space_position + 0.5;
        float d1 = texture(primary_directional_light_shadow_map, light_space_position.xy).x;
        float d2 = light_space_position.z;
        float thickness = abs(max(d1 - d2 - 0.05, 0.0)) * primary_directional_light_far;
        vec3 extinction = exp(-material.sigma_t * thickness);
        f_specular += light.emission * material.specular_albedo * spec * (d2 > d1 - 0.005 ? 1.0 : 0.0);
        f_diffuse += light.emission * material.diffuse_albedo * diff * extinction;
    }
    for(int i = 1; i < num_directional_lights; i++)
    {
        DirLight light = directional_lights[i];
        vec3 light_dir = -normalize(light.direction);
        float diff = max(dot(normal, light_dir), 0.0);
        vec3 half_dir = normalize(light_dir + view_dir);
        float spec = pow(max(0.0, dot(normal, half_dir)), material.shininess);
        f_diffuse += light.emission * material.diffuse_albedo * diff;
        f_specular += light.emission * material.specular_albedo * spec;
    }
    for(int i = 0; i < num_point_lights; i++)
    {
        PointLight light = point_lights[i];
        vec3 light_dir = normalize(light.position - position);
        float diff = max(min(dot(normal, light_dir) + material.translucency, 1.0), 0.0);
        vec3 half_dir = normalize(light_dir + view_dir);
        float spec = pow(max(0.0, dot(normal, half_dir)), material.shininess);
        float distance = length(light.position - position);
        float attenuation = 1.0 / dot(light.attenuation, vec3(1.0, distance, distance * distance));
        f_diffuse += light.emission * material.diffuse_albedo * diff * attenuation;
        f_specular += light.emission * material.specular_albedo * spec * attenuation;
    }
}
