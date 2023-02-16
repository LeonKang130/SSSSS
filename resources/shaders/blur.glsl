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
#define DMFP 0.6763163208961487
#define WIDTH 0.02
vec4 kernel[KERNEL_WIDTH] = vec4[](
    vec4(0.018091047182679176, 0.018097402527928352, 0.0180776696652174, -10),
    vec4(0.020148461684584618, 0.020128294825553894, 0.02017524652183056, -9),
    vec4(0.022637534886598587, 0.022619809955358505, 0.022591140121221542, -8),
    vec4(0.025625525042414665, 0.025703100487589836, 0.025646934285759926, -7),
    vec4(0.029509371146559715, 0.02945631928741932, 0.02950415015220642, -6),
    vec4(0.03439522534608841, 0.03442388027906418, 0.03438130021095276, -5),
    vec4(0.04110262170433998, 0.04103775694966316, 0.04116227477788925, -4),
    vec4(0.050596028566360474, 0.05073437839746475, 0.050734903663396835, -3),
    vec4(0.06581652164459229, 0.06589063256978989, 0.06575163453817368, -2),
    vec4(0.09447832405567169, 0.0943569466471672, 0.09433413296937943, -1),
    vec4(0.19519869983196259, 0.1951030045747757, 0.19528129696846008, 0),
    vec4(0.09447832405567169, 0.0943569466471672, 0.09433413296937943, 1),
    vec4(0.06581652164459229, 0.06589063256978989, 0.06575163453817368, 2),
    vec4(0.050596028566360474, 0.05073437839746475, 0.050734903663396835, 3),
    vec4(0.04110262170433998, 0.04103775694966316, 0.04116227477788925, 4),
    vec4(0.03439522534608841, 0.03442388027906418, 0.03438130021095276, 5),
    vec4(0.029509371146559715, 0.02945631928741932, 0.02950415015220642, 6),
    vec4(0.025625525042414665, 0.025703100487589836, 0.025646934285759926, 7),
    vec4(0.022637534886598587, 0.022619809955358505, 0.022591140121221542, 8),
    vec4(0.020148461684584618, 0.020128294825553894, 0.02017524652183056, 9),
    vec4(0.018091047182679176, 0.018097402527928352, 0.0180776696652174, 10)
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
