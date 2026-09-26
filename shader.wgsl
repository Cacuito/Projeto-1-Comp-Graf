struct Matrix {
  vertex: mat4x4<f32>,
}
@group(0) @binding(0) var<storage, read> matrix: array<Matrix>;

struct Global {
  projection: mat4x4<f32>,
}
@group(2) @binding(0) var<uniform> global: Global;

// Grupo 1: Textura no binding 0 e Sampler no binding 1
@group(1) @binding(0) var color_tex: texture_2d<f32>;
@group(1) @binding(1) var color_sampler: sampler;

struct VertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(
  @builtin(instance_index) instance_index: u32,
  @location(0) pos: vec2<f32>,
  @location(1) uv: vec2<f32>
) -> VertexOutput {
  var out: VertexOutput;
  out.position = global.projection * (matrix[instance_index].vertex * vec4<f32>(pos, 0.0, 1.0));
  out.uv = uv;
  return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
  return textureSample(color_tex, color_sampler, in.uv);
}