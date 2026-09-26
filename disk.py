from __future__ import annotations

from typing import TYPE_CHECKING
import math
import numpy as np
import wgpu
from shape import Shape

if TYPE_CHECKING:
    from state import State

class Disk(Shape):
    def __init__(self, device: wgpu.GPUDevice, radius: float = 1.0, n_segments: int = 64) -> None:
        coords = []
        texcoords = []
        indices = []

        # 1. Centro do disco (índice 0)
        coords.append([0.0, 0.0])
        texcoords.append([0.5, 0.5])

        # 2. Vértices do contorno circular
        for i in range(n_segments):
            theta = 2.0 * math.pi * i / n_segments
            cos_t = math.cos(theta)
            sin_t = math.sin(theta)

            coords.append([radius * cos_t, radius * sin_t])
            # Coordenadas UV: s no intervalo [0, 1] e t no intervalo [0, 1] invertido (+sin para baixo)
            texcoords.append([0.5 + 0.5 * cos_t, 0.5 + 0.5 * sin_t])

        # 3. Índices do leque de triângulos
        for i in range(1, n_segments + 1):
            next_i = 1 if i == n_segments else i + 1
            indices.extend([0, i, next_i])

        b_coords = np.array(coords, dtype=np.float32)
        b_texcoords = np.array(texcoords, dtype=np.float32)
        b_indices = np.array(indices, dtype=np.uint32)

        self.nind: int = len(indices)
        self.coord_vbo: wgpu.GPUBuffer = device.create_buffer_with_data(data=b_coords, usage=wgpu.BufferUsage.VERTEX)
        self.texcoord_vbo: wgpu.GPUBuffer = device.create_buffer_with_data(data=b_texcoords, usage=wgpu.BufferUsage.VERTEX)
        self.ibo: wgpu.GPUBuffer = device.create_buffer_with_data(data=b_indices, usage=wgpu.BufferUsage.INDEX)

    def draw(self, st: State) -> None:
        first_instance = st.get_shader().commit_matrix(st)
        st.render_pass.set_vertex_buffer(0, self.coord_vbo)
        st.render_pass.set_vertex_buffer(1, self.texcoord_vbo)
        st.render_pass.set_index_buffer(self.ibo, wgpu.IndexFormat.uint32)
        st.render_pass.draw_indexed(self.nind, 1, 0, 0, first_instance)