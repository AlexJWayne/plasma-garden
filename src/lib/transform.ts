import { type v2f, type v3f, vec2f, vec3f } from 'typegpu/data'
import { cos, sin } from 'typegpu/std'

export function rotate2d(p: v2f, theta: number) {
  'use gpu'
  const cosAngle = cos(theta)
  const sinAngle = sin(theta)
  return vec2f(
    p.x * cosAngle - p.y * sinAngle, //
    p.x * sinAngle + p.y * cosAngle,
  )
}

export function rotateX(p: v3f, theta: number) {
  'use gpu'
  const cosAngle = cos(theta)
  const sinAngle = sin(theta)
  return vec3f(
    p.x,
    p.y * cosAngle - p.z * sinAngle,
    p.y * sinAngle + p.z * cosAngle,
  )
}

export function rotateY(p: v3f, theta: number) {
  'use gpu'
  const cosAngle = cos(theta)
  const sinAngle = sin(theta)
  return vec3f(
    p.x * cosAngle + p.z * sinAngle,
    p.y,
    -p.x * sinAngle + p.z * cosAngle,
  )
}

export function rotateZ(p: v3f, theta: number) {
  'use gpu'
  const cosAngle = cos(theta)
  const sinAngle = sin(theta)
  return vec3f(
    p.x * cosAngle - p.y * sinAngle,
    p.x * sinAngle + p.y * cosAngle,
    p.z,
  )
}
