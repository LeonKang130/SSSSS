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

#define KERNEL_WIDTH 9
#define DMFP 0.18239612877368927
#define WIDTH 0.1
vec4 kernel[KERNEL_WIDTH] = vec4[](
    vec4(0.03557949140667915, 0.04551077261567116, 0.035560622811317444, -4),
    vec4(0.05146254226565361, 0.0610208660364151, 0.051441557705402374, -3),
    vec4(0.07988868653774261, 0.08702393621206284, 0.07990314811468124, -2),
    vec4(0.14081960916519165, 0.13919439911842346, 0.14077401161193848, -1),
    vec4(0.3844994008541107, 0.33450010418891907, 0.38464128971099854, 0),
    vec4(0.14081960916519165, 0.13919439911842346, 0.14077401161193848, 1),
    vec4(0.07988868653774261, 0.08702393621206284, 0.07990314811468124, 2),
    vec4(0.05146254226565361, 0.0610208660364151, 0.051441557705402374, 3),
    vec4(0.03557949140667915, 0.04551077261567116, 0.035560622811317444, 4)
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
