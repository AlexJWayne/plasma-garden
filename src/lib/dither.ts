import tgpu from 'typegpu'
import { type v2f, type v3f, vec3f } from 'typegpu/data'
import { dot, fract } from 'typegpu/std'

const ditherNoiseSeed = tgpu.const(
  vec3f,
  vec3f(0.06711056, 0.00583715, 52.9829189),
)

/**
 * Randomly add 1 step to color value to eliminate banding artifacts.
 *
 * Provide a color that is banding, and get a color that is less banding.
 */
export function dither(color: v3f, clipPos: v2f): v3f {
  'use gpu'
  const noise = fract(
    fract(dot(clipPos, ditherNoiseSeed.$.xy) * ditherNoiseSeed.$.z),
  )
  const dither = (noise - 0.5) / 255.0
  return color.add(vec3f(dither))
}
