import tgpu from 'typegpu'
import { arrayOf, type v3f, vec2f, vec3f } from 'typegpu/data'

export const quadVertices = tgpu.const(arrayOf(vec2f, 6), [
  vec2f(-1, -1),
  vec2f(-1, 1),
  vec2f(1, 1),
  vec2f(1, 1),
  vec2f(1, -1),
  vec2f(-1, -1),
])

export const cubeVertices = tgpu.const(arrayOf(vec3f, 36), [
  // Front face
  vec3f(-1, -1, 1),
  vec3f(1, -1, 1),
  vec3f(1, 1, 1),
  vec3f(1, 1, 1),
  vec3f(-1, 1, 1),
  vec3f(-1, -1, 1),
  // Back face
  vec3f(1, -1, -1),
  vec3f(-1, -1, -1),
  vec3f(-1, 1, -1),
  vec3f(-1, 1, -1),
  vec3f(1, 1, -1),
  vec3f(1, -1, -1),
  // Top face
  vec3f(-1, 1, 1),
  vec3f(1, 1, 1),
  vec3f(1, 1, -1),
  vec3f(1, 1, -1),
  vec3f(-1, 1, -1),
  vec3f(-1, 1, 1),
  // Bottom face
  vec3f(-1, -1, -1),
  vec3f(1, -1, -1),
  vec3f(1, -1, 1),
  vec3f(1, -1, 1),
  vec3f(-1, -1, 1),
  vec3f(-1, -1, -1),
  // Right face
  vec3f(1, -1, 1),
  vec3f(1, -1, -1),
  vec3f(1, 1, -1),
  vec3f(1, 1, -1),
  vec3f(1, 1, 1),
  vec3f(1, -1, 1),
  // Left face
  vec3f(-1, -1, -1),
  vec3f(-1, -1, 1),
  vec3f(-1, 1, 1),
  vec3f(-1, 1, 1),
  vec3f(-1, 1, -1),
  vec3f(-1, -1, -1),
])

export function cubeVertex(idx: number, width: number, height: number): v3f {
  'use gpu'
  let pos = cubeVertices.$[idx].mul(0.5).add(vec3f(0, 0, 0.5))
  return pos.mul(vec3f(width, width, height))
}
