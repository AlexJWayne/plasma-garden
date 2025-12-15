import type { v3f } from 'typegpu/data'
import { vec2f } from 'typegpu/data'
import { length } from 'typegpu/std'

export function sdTorus(p: v3f, r1: number, r2: number): number {
  'use gpu'
  const q = vec2f(length(p.xz) - r1, p.y)
  return length(q) - r2
}
