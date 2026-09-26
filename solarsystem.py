from __future__ import annotations

from typing import Any
import time

import wgpu
from rendercanvas.glfw import RenderCanvas, loop

from camera2d import Camera2D
from transform import Transform
from quad import Quad
from disk import Disk
from node import Node
from shader import Shader
from pipeline import Pipeline
from scene import Scene
from renderer import Renderer
from engine import Engine
from texture import Texture
from textureset import TextureSet
from sampler import Sampler

canvas: RenderCanvas
device: wgpu.GPUDevice
context: Any
renderer: Renderer
camera: Camera2D
scene: Scene
last_t: float = 0.0

class SolarSystemEngine(Engine):
    def __init__(self, trf_venus: Transform, trf_earth_orbit: Transform, trf_earth_spin: Transform, trf_moon: Transform) -> None:
        self.trf_venus = trf_venus
        self.trf_earth_orbit = trf_earth_orbit
        self.trf_earth_spin = trf_earth_spin
        self.trf_moon = trf_moon

    def update(self, dt: float) -> None:
        # Translação de Vênus (mais rápido que a Terra)
        self.trf_venus.rotate(1.5 * dt, 0, 0, 1)

        # Translação da Terra em torno do Sol
        self.trf_earth_orbit.rotate(0.7 * dt, 0, 0, 1)

        # Rotação da Terra em torno do próprio eixo (Spin)
        self.trf_earth_spin.rotate(2.5 * dt, 0, 0, 1)

        # Translação da Lua em torno da Terra
        self.trf_moon.rotate(3.8 * dt, 0, 0, 1)

def initialize(device: wgpu.GPUDevice, target_format: str) -> None:
    global camera, scene

    # Câmera 2D centralizada em (0, 0)
    camera = Camera2D(-2.5, 2.5, -2.5, 2.5)

    # 1. Geometrias
    unit_disk = Disk(device, radius=1.0, n_segments=64)
    background_quad = Quad(device)

    # 2. Texturas e TextureSets
    tex_space = Texture(device, "color_tex", "../images/espaco.jpg")
    tex_sun = Texture(device, "color_tex", "../images/sol.png")
    tex_venus = Texture(device, "color_tex", "../images/venus.png")
    tex_earth = Texture(device, "color_tex", "../images/terra.png")
    tex_moon = Texture(device, "color_tex", "../images/lua.png")

    # Sampler compartilhado para o binding 1 do grupo 1
    shared_sampler = Sampler(device, "color_sampler")

    app_space = TextureSet([tex_space, shared_sampler])
    app_sun   = TextureSet([tex_sun, shared_sampler])
    app_venus = TextureSet([tex_venus, shared_sampler])
    app_earth = TextureSet([tex_earth, shared_sampler])
    app_moon  = TextureSet([tex_moon, shared_sampler])

    # ------------------ FUNDO (Espaço) ------------------
    # Quad mapeia [0, 1]x[0, 1], escalamos e transladamos para cobrir [-2.5, 2.5]
    trf_space = Transform()
    trf_space.translate(-2.5, -2.5, -0.9)
    trf_space.scale(5.0, 5.0, 1.0)
    node_space = Node(trf=trf_space, apps=[app_space], shps=[background_quad])

    # ------------------ SOL (Centro) ------------------
    trf_sun = Transform()
    trf_sun.scale(0.45, 0.45, 1.0)
    node_sun = Node(trf=trf_sun, apps=[app_sun], shps=[unit_disk])

    # ------------------ VÊNUS (Entre o Sol e a Terra) ------------------
    trf_venus_orbit = Transform()
    trf_venus_pos = Transform()
    trf_venus_pos.translate(0.9, 0.0, 0.0)

    trf_venus_scale = Transform()
    trf_venus_scale.scale(0.10, 0.10, 1.0)
    node_venus_body = Node(trf=trf_venus_scale, apps=[app_venus], shps=[unit_disk])

    node_venus = Node(trf=trf_venus_orbit, nodes=[
        Node(trf=trf_venus_pos, nodes=[node_venus_body])
    ])

    # ------------------ TERRA E LUA ------------------
    trf_earth_orbit = Transform()
    trf_earth_pos = Transform()
    trf_earth_pos.translate(1.7, 0.0, 0.0)

    # Rotação da Terra em torno do próprio eixo (Spin)
    trf_earth_spin = Transform()
    trf_earth_scale = Transform()
    trf_earth_scale.scale(0.16, 0.16, 1.0)
    node_earth_body = Node(trf=trf_earth_scale, apps=[app_earth], shps=[unit_disk])
    node_earth_spinning = Node(trf=trf_earth_spin, nodes=[node_earth_body])

    # Órbita da Lua: Filha direta de trf_earth_pos (NÃO herda o spin da Terra!)
    trf_moon_orbit = Transform()
    trf_moon_pos = Transform()
    trf_moon_pos.translate(0.32, 0.0, 0.0)

    trf_moon_scale = Transform()
    trf_moon_scale.scale(0.045, 0.045, 1.0)
    node_moon_body = Node(trf=trf_moon_scale, apps=[app_moon], shps=[unit_disk])
    node_moon = Node(trf=trf_moon_orbit, nodes=[
        Node(trf=trf_moon_pos, nodes=[node_moon_body])
    ])

    # Nó da Terra carrega a Terra girando e a órbita da Lua como irmãs
    node_earth = Node(trf=trf_earth_orbit, nodes=[
        Node(trf=trf_earth_pos, nodes=[node_earth_spinning, node_moon])
    ])

    # ------------------ PIPELINE E SHADER COM TEXTURA ------------------
    # Usando o shader textured 2D da biblioteca
    shader = Shader(device, "../shaders/2d/shader.wgsl")
    shader.set_vertex_buffers([
        {"array_stride": 2 * 4, "step_mode": "vertex",
         "attributes": [{"format": "float32x2", "offset": 0, "var_name": "pos"}]},
        {"array_stride": 2 * 4, "step_mode": "vertex",
         "attributes": [{"format": "float32x2", "offset": 0, "var_name": "uv"}]},
    ])
    pipeline = Pipeline(shader, target_format, depth_stencil=None)

    # Registra os texture sets no shader
    shader.add_texture_set(app_space)
    shader.add_texture_set(app_sun)
    shader.add_texture_set(app_venus)
    shader.add_texture_set(app_earth)
    shader.add_texture_set(app_moon)

    root = Node(pipeline, nodes=[node_space, node_sun, node_venus, node_earth])
    scene = Scene(root)
    scene.add_engine(SolarSystemEngine(trf_venus_orbit, trf_earth_orbit, trf_earth_spin, trf_moon_orbit))

def update(dt: float) -> None:
    scene.update(dt)

def draw() -> None:
    global last_t
    t = time.perf_counter()
    update(t - last_t)
    last_t = t

    target_texture = context.get_current_texture()
    renderer.render(target_texture, scene, camera)

def on_key(event: Any) -> None:
    if event["key"] == "q":
        canvas.close()

def main() -> None:
    global canvas, device, context, renderer, last_t

    canvas = RenderCanvas(size=(750, 750), title="Projeto 1: Mini-Sistema Solar 2D", update_mode="continuous", max_fps=60)
    adapter = wgpu.gpu.request_adapter_sync()
    device = adapter.request_device_sync()
    context = canvas.get_context("wgpu")

    target_format = context.get_preferred_format(device.adapter)
    context.configure(device=device, format=target_format)

    renderer = Renderer(device, clear_value=(0.0, 0.0, 0.0, 1.0))

    initialize(device, target_format)

    canvas.add_event_handler(on_key, "key_down")
    last_t = time.perf_counter()
    canvas.request_draw(draw)
    loop.run()

if __name__ == "__main__":
    main()