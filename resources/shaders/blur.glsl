#version 330
uniform sampler2D matte;
uniform sampler2D depth;
uniform sampler2D diffuse;
uniform vec2 direction;
uniform float camera_near;
uniform float camera_far;
uniform float camera_fov;
in vec2 f_texcoord;
out vec3 f_color;

#define KERNEL_WIDTH 21
#define DMFP 0.45922210812568665
#define WIDTH 0.03
vec4 kernel[KERNEL_WIDTH] = vec4[](
    vec4(0.017366673797369003, 0.01639840379357338, 0.01726832613348961, -10),
    vec4(0.01945168524980545, 0.018268978223204613, 0.01930919662117958, -9),
    vec4(0.02191878668963909, 0.020560288801789284, 0.02174455299973488, -8),
    vec4(0.025054914876818657, 0.023280955851078033, 0.024828949943184853, -7),
    vec4(0.028977666050195694, 0.026737049221992493, 0.028616489842534065, -6),
    vec4(0.03394278883934021, 0.031252164393663406, 0.033579643815755844, -5),
    vec4(0.04083738476037979, 0.03728603944182396, 0.040301624685525894, -4),
    vec4(0.05054687708616257, 0.04592210799455643, 0.049964789301157, -3),
    vec4(0.06614365428686142, 0.05967700481414795, 0.06523199379444122, -2),
    vec4(0.09568978101015091, 0.08559928834438324, 0.09418433159589767, -1),
    vec4(0.20013952255249023, 0.17682097852230072, 0.19647517800331116, 0),
    vec4(0.09568978101015091, 0.08559928834438324, 0.09418433159589767, 1),
    vec4(0.06614365428686142, 0.05967700481414795, 0.06523199379444122, 2),
    vec4(0.05054687708616257, 0.04592210799455643, 0.049964789301157, 3),
    vec4(0.04083738476037979, 0.03728603944182396, 0.040301624685525894, 4),
    vec4(0.03394278883934021, 0.031252164393663406, 0.033579643815755844, 5),
    vec4(0.028977666050195694, 0.026737049221992493, 0.028616489842534065, 6),
    vec4(0.025054914876818657, 0.023280955851078033, 0.024828949943184853, 7),
    vec4(0.02191878668963909, 0.020560288801789284, 0.02174455299973488, 8),
    vec4(0.01945168524980545, 0.018268978223204613, 0.01930919662117958, 9),
    vec4(0.017366673797369003, 0.01639840379357338, 0.01726832613348961, 10)
);

float linearize_depth(float z)
{
    z = 2.0 * z - 1.0;
    return 2.0 * camera_far * camera_near / (camera_far + camera_near - z * (camera_far - camera_near));
}

void main()
{
    vec3 color_m = texture(diffuse, f_texcoord).rgb;
    float depth_m = linearize_depth(texture(depth, f_texcoord).x);
    vec2 unit_step = normalize(direction) * DMFP * WIDTH / (2.0 * depth_m * tan(0.5 * radians(camera_fov)));
    vec3 color_blurred = vec3(0.0);
    for(int i = 0; i < KERNEL_WIDTH; i++)
    {
        vec2 texcoord = f_texcoord + unit_step * kernel[i].a;
        vec3 color_s = texture(diffuse, texcoord).rgb;
        float depth_s = linearize_depth(texture(depth, texcoord).x);
        float matte_s = texture(matte, texcoord).x;
        float s = min(1.0, abs(depth_m - depth_s) / (DMFP * (KERNEL_WIDTH >> 1)));
        color_s = mix(color_s, color_m, s);
        color_blurred += color_s * kernel[i].rgb;
    }
    f_color = color_blurred * texture(matte, f_texcoord).x;
}
