import { type m4x4f, mat4x4f, type v3f, vec4f } from 'typegpu/data'
import { cross, dot, normalize } from 'typegpu/std'

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
