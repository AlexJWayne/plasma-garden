import {
  type Infer,
  bool,
  f32,
  struct,
  type v3f,
  vec2f,
  vec3f,
} from 'typegpu/data'
import { length, normalize } from 'typegpu/std'

export const RayHit = struct({ isHit: bool, pos: vec3f })
export type RayHit = Infer<typeof RayHit>

export type SdSurface<T, R = number> = (p: v3f, arg: T) => R

export function createRaymarch<T>(
  sdSurface: SdSurface<T>,
  {
    maxSteps,
    maxDistance,
    epsilon,
  }: {
    maxSteps: number
    maxDistance: number
    epsilon: number
  },
): (cameraPos: v3f, worldPos: v3f, arg: T) => RayHit {
  maxDistance = f32(maxDistance)

  return function raymarch(cameraPos: v3f, worldPos: v3f, arg: T): RayHit {
    'use gpu'

    const triDiff = worldPos.sub(cameraPos)
    let totalDistance = length(triDiff)
    const rayDirection = normalize(triDiff)

    for (let i = 0; i < maxSteps; i++) {
      const point = cameraPos.add(rayDirection.mul(totalDistance))
      const distance = sdSurface(point, arg)

      if (distance < epsilon) return RayHit({ isHit: true, pos: point })
      if (distance > maxDistance) break

      totalDistance += distance
    }

    return RayHit({ isHit: false, pos: vec3f() })
  }
}

export function createCalcNormal<T>(
  sdSurface: SdSurface<T>,
  epsilon: number,
): SdSurface<T, v3f> {
  return function calcNormal(p: v3f, arg: T): v3f {
    'use gpu'
    const k = vec2f(1, -1)
    return normalize(
      k.xyy
        .mul(sdSurface(p.add(k.xyy.mul(epsilon)), arg))
        .add(
          k.yyx
            .mul(sdSurface(p.add(k.yyx.mul(epsilon)), arg))
            .add(
              k.yxy
                .mul(sdSurface(p.add(k.yxy.mul(epsilon)), arg))
                .add(k.xxx.mul(sdSurface(p.add(k.xxx.mul(epsilon)), arg))),
            ),
        ),
    )
  }
}
