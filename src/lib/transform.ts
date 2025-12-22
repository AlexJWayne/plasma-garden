import { type v2f, vec2f } from 'typegpu/data'
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
