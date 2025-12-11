import {
  type m4x4f,
  mat4x4f,
  type v2f,
  type v3f,
  vec2f,
  vec4f,
} from 'typegpu/data'
import { cross, dot, normalize } from 'typegpu/std'

export const blending = {
  // premultiplied alpha
  normal: {
    color: {
      srcFactor: 'one',
      dstFactor: 'one-minus-src-alpha',
    },
    alpha: {
      srcFactor: 'one',
      dstFactor: 'one-minus-src-alpha',
    },
  },
} satisfies Record<string, GPUBlendState>

export function quadVert(idx: number): v2f {
  'use gpu'
  const quadVertices = [
    vec2f(-1, -1),
    vec2f(-1, 1),
    vec2f(1, 1),
    vec2f(1, 1),
    vec2f(1, -1),
    vec2f(-1, -1),
  ]
  return quadVertices[idx]
}

export function lookAt(eye: v3f, target: v3f, up: v3f): m4x4f {
  'use gpu'
  const zaxis = normalize(eye.sub(target))
  const xaxis = normalize(cross(normalize(up), zaxis))
  const yaxis = cross(zaxis, xaxis)

  return mat4x4f(
    vec4f(xaxis.x, yaxis.x, zaxis.x, 0.0),
    vec4f(xaxis.y, yaxis.y, zaxis.y, 0.0),
    vec4f(xaxis.z, yaxis.z, zaxis.z, 0.0),
    vec4f(-dot(xaxis, eye), -dot(yaxis, eye), -dot(zaxis, eye), 1.0),
  )
}
