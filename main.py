from typing import NamedTuple, List

import moderngl
import numpy as np
import tinyobjloader
from pyrr import Vector3, Matrix44
from luisa.window import Window
import dearpygui.dearpygui as dpg
from matplotlib import pyplot as plt
from camera import Camera

res = (800, 800)
ctx = moderngl.create_standalone_context()
camera = Camera(Vector3([0.0, 1.0, 8.0]), Vector3([0.0, 0.5, 0.0]), res, 20.0)


class Material(NamedTuple):
    diffuse_albedo: Vector3
    specular_albedo: Vector3
    shininess: float
    translucency: float
    sigma_t: Vector3


class DirLight(NamedTuple):
    direction: Vector3
    emission: Vector3


class PointLight(NamedTuple):
    position: Vector3
    emission: Vector3
    attenuation: Vector3


class SurfaceLight(NamedTuple):
    """
        Could be supported by using LTC, but not yet implemented.
        Must be quadrilateral.
    """
    vertices: np.ndarray
    emission: Vector3


directional_lights: List[DirLight] = []
point_lights: List[PointLight] = []
surface_lights: List[SurfaceLight] = []
model_file = "resources/objects/bunny.obj"
material = Material(Vector3([0.6, 0.6, 0.6]), Vector3([0.2, 0.2, 0.2]), 16.0, 0.3, Vector3([8.0, 8.0, 8.0]))
"""
Light format:
Light XXX
Point x y z
Emission r g b
Attenuation c l q

Light XXX
Directional x y z
Emission r g b
"""
with open("scene.sc", "r") as f:
    while line := f.readline():
        if line.startswith("Objects"):
            next_line = f.readline()
            model_file = next_line.strip()
            while (line := f.readline()) and (line := line.strip()):
                continue
        elif line.startswith("Lights"):
            print(line.strip())
            line = f.readline()
            if line.startswith("Directional"):
                direction = Vector3([float(coordinate) for coordinate in line.strip().split(' ')[1:]]).normalised
                emission = Vector3()
                while (line := f.readline()) and (line := line.strip()):
                    if line.startswith("Emission"):
                        emission = Vector3([float(channel) for channel in line.split(' ')[1:]])
                directional_lights.append(DirLight(direction, emission))
                print(directional_lights[-1])
            elif line.startswith("Point"):
                position = Vector3([float(coordinate) for coordinate in line.strip().split(' ')[1:]])
                emission, attenuation = Vector3(), Vector3([1.0, 0.0, 0.0])
                while (line := f.readline()) and (line := line.strip()):
                    if line.startswith("Emission"):
                        emission = Vector3([float(channel) for channel in line.split(' ')[1:]])
                    elif line.startswith("Attenuation"):
                        attenuation = Vector3([float(component) for component in line.split(' ')[1:]])
                point_lights.append(PointLight(position, emission, attenuation))
                print(point_lights[-1])
            elif line.startswith("Area"):
                filename = line.strip().split(' ')[1]
                reader = tinyobjloader.ObjReader()
                if not reader.ParseFromFile(filename):
                    exit(-1)
                vertices = np.array(reader.GetAttrib().vertices, dtype=np.float32).reshape(4, 3)
                emission = Vector3()
                while (line := f.readline()) and (line := line.strip()):
                    if line.startswith("Emission"):
                        emission = Vector3([float(channel) for channel in line.split(' ')[1:]])
                surface_lights.append(SurfaceLight(vertices, emission))
reader = tinyobjloader.ObjReader()
if not reader.ParseFromFile(model_file):
    exit(-1)
vertices = np.array(reader.GetAttrib().vertices, dtype=np.float32).reshape(-1, 3)
normal_vectors = np.array(reader.GetAttrib().normals, dtype=np.float32).reshape(-1, 3)
normals = np.zeros(vertices.shape, dtype=np.float32)
for shape in reader.GetShapes():
    for index in shape.mesh.indices:
        normals[index.vertex_index] = normal_vectors[index.normal_index]
triangles = np.array([index.vertex_index for shape in reader.GetShapes() for index in shape.mesh.indices])
"""
Render shadow maps for primary lights.
The first lights of each categories are considered primary(most important),
and back-lighting would only be calculated for these lights.
"""
vbo = ctx.buffer(vertices)
nbo = ctx.buffer(normals)
ibo = ctx.buffer(triangles)
shadow_caster = ctx.program(
    vertex_shader=
    """
        #version 330
        uniform mat4 light_view_projection;
        in vec3 in_position;
        void main()
        {
            gl_Position = light_view_projection * vec4(in_position, 1.0);
        }
    """,
    fragment_shader=
    """
    #version 330
    void main()
    {

    }
    """
)
shadow_vao = ctx.vertex_array(
    shadow_caster,
    [
        (vbo, '3f', "in_position"),
    ],
    index_buffer=ibo
)
primary_directional_light_shadow_map = ctx.depth_texture(res)
primary_directional_light_view_projection = Matrix44(dtype=np.float32)
primary_point_light_shadow_map = ctx.depth_texture(res)
primary_directional_light_near = 0.01
primary_directional_light_far = 100.0
if len(directional_lights) != 0:
    light = directional_lights[0]
    light_forward = light.direction.normalised
    light_right = light_forward.cross(Vector3([0.0, 1.0, 0.0]))
    if light_right.length < 1e-4:
        light_right = Vector3([1.0, 0.0, 0.0])
    else:
        light_right = light_right.normalised
    light_up = light_right.cross(light_forward)
    light_look_at = Matrix44.look_at(-10.0 * light.direction.normalised, Vector3([0.0, 0.0, 0.0]), light_up,
                                     dtype=np.float32)
    light_projection = Matrix44.orthogonal_projection(
        -10.0,
        10.0,
        -10.0,
        10.0,
        primary_directional_light_near,
        primary_directional_light_far,
        dtype=np.float32
    )
    primary_directional_light_view_projection = light_projection * light_look_at
    shadow_caster["light_view_projection"].write(primary_directional_light_view_projection.tobytes())
"""
Render diffuse and specular components into textures, and save matte stencil and depth texture for later processing.
"""
with open("resources/shaders/vert.glsl", "r") as vertex_shader:
    with open("resources/shaders/frag.glsl", "r") as fragment_shader:
        preprocessing_program = ctx.program(
            vertex_shader=vertex_shader.read(),
            fragment_shader=fragment_shader.read()
        )
preprocessing_program["mvp"].write(camera.mvp.astype('f4').tobytes())
preprocessing_program["camera_position"].write(camera.origin.astype('f4').tobytes())
preprocessing_program["material.diffuse_albedo"].write(material.diffuse_albedo.astype('f4').tobytes())
preprocessing_program["material.specular_albedo"].write(material.specular_albedo.astype('f4').tobytes())
preprocessing_program["material.shininess"] = material.shininess
preprocessing_program["material.translucency"] = material.translucency
preprocessing_program["material.sigma_t"].write(material.sigma_t.astype('f4').tobytes())
num_directional_lights = min(16, len(directional_lights))
preprocessing_program["num_directional_lights"] = num_directional_lights
preprocessing_program["primary_directional_light_shadow_map"] = 0
preprocessing_program["primary_directional_light_view_projection"].write(
    primary_directional_light_view_projection.tobytes())
preprocessing_program["primary_directional_light_far"] = primary_directional_light_far
for i in range(num_directional_lights):
    preprocessing_program[f"directional_lights[{i}].direction"].write(
        directional_lights[i].direction.astype('f4').tobytes())
    preprocessing_program[f"directional_lights[{i}].emission"].write(
        directional_lights[i].emission.astype('f4').tobytes())
num_point_lights = min(16, len(point_lights))
preprocessing_program["num_point_lights"] = num_point_lights
for i in range(num_point_lights):
    preprocessing_program[f"point_lights[{i}].position"].write(point_lights[i].position.astype('f4').tobytes())
    preprocessing_program[f"point_lights[{i}].emission"].write(point_lights[i].emission.astype('f4').tobytes())
    preprocessing_program[f"point_lights[{i}].attenuation"].write(point_lights[i].attenuation.astype('f4').tobytes())
render_vao = ctx.vertex_array(
    preprocessing_program,
    [
        (vbo, '3f', "in_position"),
        (nbo, '3f', "in_normal")
    ],
    index_buffer=ibo
)
diffuse, specular = ctx.texture(res, 3, dtype='f4'), ctx.texture(res, 3, dtype='f4')
matte = ctx.texture(res, 1, dtype='f4')
depth = ctx.depth_texture(res)
"""
Blur the diffuse component using the depth texture, matte stencil and original diffuse component which was rendered
using the Lambertian model.
"""
quad_vertices = np.array([
    [1.0, 1.0, 0.01],
    [-1.0, 1.0, 0.01],
    [-1.0, -1.0, 0.01],
    [1.0, 1.0, 0.01],
    [-1.0, -1.0, 0.01],
    [1.0, -1.0, 0.01]
], dtype=np.float32)
quad_vbo = ctx.buffer(quad_vertices.tobytes())
with open("resources/shaders/blur.glsl", "r") as blurring_shader:
    blurring_program = ctx.program(
        vertex_shader=
        """
        #version 330
        in vec3 in_position;
        out vec2 f_texcoord;
        void main()
        {
            gl_Position = vec4(in_position, 1.0);
            f_texcoord = in_position.xy * 0.5 + 0.5;
        }
        """,
        fragment_shader=blurring_shader.read()
    )
blurring_program["matte"] = 0
blurring_program["depth"] = 1
blurring_program["diffuse"] = 2
blurring_program["camera_near"] = camera.near
blurring_program["camera_far"] = camera.far
blurring_program["camera_fov"] = camera.fov
blurring_vao = ctx.simple_vertex_array(blurring_program, quad_vbo, 'in_position')
blurred_diffuse = ctx.texture(res, 3, dtype='f4')
"""
Combine the blurred diffuse component with the specular component.
"""
combining_program = ctx.program(
    vertex_shader=
    """
        #version 330
        in vec3 in_position;
        out vec2 f_texcoord;
        void main()
        {
            gl_Position = vec4(in_position, 1.0);
            gl_Position.xy = -gl_Position.xy;
            f_texcoord = in_position.xy * 0.5 + 0.5;
        }
    """,
    fragment_shader=
    """
        #version 330
        uniform sampler2D blurred_diffuse;
        uniform sampler2D specular;
        in vec2 f_texcoord;
        out vec4 f_color;
        vec3 aces_tone_mapping(vec3 color)
        {
            vec3 mapped = color * (2.51 * color + 0.03) / (color * (2.43 * color + 0.59) + 0.14);
            return max(vec3(0.0), min(vec3(1.0), mapped));
        }
        void main()
        {
            vec3 color = texture(blurred_diffuse, f_texcoord).rgb + texture(specular, f_texcoord).rgb;
            f_color = vec4(aces_tone_mapping(color), 1.0);
        }
    """
)
combining_program["blurred_diffuse"] = 0
combining_program["specular"] = 1
combining_vao = ctx.simple_vertex_array(combining_program, quad_vbo, 'in_position')
combined_texture = ctx.texture(res, 4, dtype='f4')
buffer = bytearray(res[0] * res[1] * 4 * 4)
arr = np.frombuffer(buffer, dtype=np.float32).reshape(res + (-1,))
w = Window("SSSSS", res, resizable=False, frame_rate=True)
w.set_background(arr, res)
dpg.draw_image("background", (0, 0), res, parent="viewport_draw")

def update():
    global diffuse, blurred_diffuse, combined_texture, buffer
    fbo = ctx.framebuffer(color_attachments=[], depth_attachment=primary_directional_light_shadow_map)
    with ctx.scope(fbo, moderngl.DEPTH_TEST):
        fbo.clear()
        shadow_vao.render()
    fbo.release()
    fbo = ctx.framebuffer(
        color_attachments=[
            diffuse, specular, matte
        ],
        depth_attachment=depth
    )
    scope = ctx.scope(
        fbo,
        moderngl.DEPTH_TEST,
        textures=[
            (primary_directional_light_shadow_map, 0),
            (primary_point_light_shadow_map, 1)
        ]
    )
    with scope:
        fbo.clear()
        render_vao.render()
    fbo.release()
    blurring_program["direction"] = (0.0, 1.0)
    fbo = ctx.framebuffer([blurred_diffuse])
    scope = ctx.scope(
        fbo,
        textures=[
            (matte, 0),
            (depth, 1),
            (diffuse, 2)
        ]
    )
    with scope:
        fbo.clear()
        blurring_vao.render()
    fbo.release()
    diffuse, blurred_diffuse = blurred_diffuse, diffuse
    fbo = ctx.framebuffer([blurred_diffuse])
    blurring_program["direction"] = (1.0, 0.0)
    scope = ctx.scope(
        fbo,
        textures=[
            (matte, 0),
            (depth, 1),
            (diffuse, 2)
        ]
    )
    with scope:
        fbo.clear()
        blurring_vao.render()
    fbo.release()
    fbo = ctx.framebuffer([combined_texture])
    scope = ctx.scope(
        fbo,
        textures=[
            (blurred_diffuse, 0),
            (specular, 1)
        ]
    )
    with scope:
        fbo.clear()
        combining_vao.render()
    fbo.release()
    combined_texture.read_into(buffer)
    w.update_frame_rate(dpg.get_frame_rate())


if __name__ == "__main__":
    w.run(update)
    plt.imsave("result.png", arr)
