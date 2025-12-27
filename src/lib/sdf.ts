import type { v3f } from 'typegpu/data'
import { vec2f } from 'typegpu/data'
import {
  clamp,
  cos,
  dot,
  length,
  max,
  min,
  mul,
  sign,
  sin,
  sqrt,
} from 'typegpu/std'

export function sdTorus(p: v3f, r1: number, r2: number): number {
  'use gpu'
  const q = vec2f(length(p.xz) - r1, p.y)
  return length(q) - r2
}

export function sdCone(p: v3f, angle: number, h: number): number {
  'use gpu'
  const c = vec2f(sin(angle), cos(angle))

  const q = mul(h, vec2f(c.x / c.y, -1.0))

  const w = vec2f(length(p.xz), p.y)
  const a = w.sub(q.mul(clamp(dot(w, q) / dot(q, q), 0.0, 1.0)))
  const b = w.sub(q.mul(vec2f(clamp(w.x / q.x, 0.0, 1.0), 1.0)))
  const k = sign(q.y)
  const d = min(dot(a, a), dot(b, b))
  const s = max(k * (w.x * q.y - w.y * q.x), k * (w.y - q.y))
  return sqrt(d) * sign(s)
}

export function opIntersection(a: number, b: number): number {
  'use gpu'
  return max(a, b)
}
