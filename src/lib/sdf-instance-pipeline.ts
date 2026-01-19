import tgpu, { type TgpuBufferReadonly, type TgpuBufferUniform } from 'typegpu'
import {
  type BaseData,
  type Infer,
  type InferGPU,
  type WgslArray,
  type WgslStruct,
  bool,
  builtin,
  f32,
  interpolate,
  type m4x4f,
  struct,
  u32,
  type v3f,
  vec2f,
  vec3b,
  vec3f,
  vec4f,
} from 'typegpu/data'
import { length, normalize, select } from 'typegpu/std'

import type { CameraStruct } from '../components/game/camera'

import { cubeVertices } from './geometry'
import { Lighting, SurfaceColors, calcSurfaceLighting } from './lighting'
import { type SdSurface } from './raymarching'

export function createSdfSurfaceShaders() {}

export const AABB = struct({ min: vec3f, max: vec3f })
export type AABB = Infer<typeof AABB>

export const RayHit = struct({ isHit: bool, pos: vec3f })
export type RayHit = Infer<typeof RayHit>

const DEBUG_MISS = { color: vec4f(1, 0, 1, 1).mul(0.25), depth: 0 }

export function createSDFInstanceShaderProgram<T extends BaseData>({
  cameraBuffer,
  instanceBuffer,

  calcAABB,
  sdSurface,
  calcSurfaceColors,

  maxSteps,
  maxDistance,
  epsilon,
  epsilonNormal = epsilon,

  debug = false,
}: {
  cameraBuffer: TgpuBufferUniform<typeof CameraStruct>
  instanceBuffer: TgpuBufferReadonly<WgslArray<T>>

  calcAABB: (entity: Infer<T>) => AABB
  sdSurface: SdSurface<Infer<T>>
  calcSurfaceColors: (p: v3f, entity: Infer<T>) => SurfaceColors

  maxSteps: number
  maxDistance: number
  epsilon: number
  epsilonNormal: number

  debug?: boolean
}) {
  const vertexProgram = tgpu['~unstable'].vertexFn({
    in: {
      instanceIdx: builtin.instanceIndex,
      vertexIdx: builtin.vertexIndex,
    },
    out: {
      worldPos: vec3f,
      clipPos: builtin.position,
      instanceIdx: interpolate('flat', u32),
    },
  })(({ vertexIdx, instanceIdx }) => {
    const local = cubeVertices.$[vertexIdx]
    const instance = instanceBuffer.$[instanceIdx] as Infer<T>

    const aabb = calcAABB(instance)
    const worldPos = select(
      aabb.min,
      aabb.max,
      vec3b(local.x > 0, local.y > 0, local.z > 0),
    )

    const clipPos = cameraBuffer.$.viewMatrix.mul(vec4f(worldPos, 1))

    return {
      worldPos,
      clipPos,
      instanceIdx: instanceIdx,
    }
  })

  function raymarch(cameraPos: v3f, worldPos: v3f, arg: Infer<T>): RayHit {
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

  function calcNormal(p: v3f, instance: Infer<T>): v3f {
    'use gpu'
    const k = vec2f(1, -1)
    return normalize(
      k.xyy
        .mul(sdSurface(p.add(k.xyy.mul(epsilonNormal)), instance))
        .add(
          k.yyx
            .mul(sdSurface(p.add(k.yyx.mul(epsilonNormal)), instance))
            .add(
              k.yxy
                .mul(sdSurface(p.add(k.yxy.mul(epsilonNormal)), instance))
                .add(
                  k.xxx.mul(
                    sdSurface(p.add(k.xxx.mul(epsilonNormal)), instance),
                  ),
                ),
            ),
        ),
    )
  }

  const fragmentProgram = tgpu['~unstable'].fragmentFn({
    in: {
      worldPos: vec3f,
      instanceIdx: interpolate('flat', u32),
    },
    out: {
      color: vec4f,
      depth: builtin.fragDepth,
    },
  })(({ worldPos, instanceIdx }) => {
    'use gpu'

    const instance = instanceBuffer.$[instanceIdx] as Infer<T>
    const rayHit = raymarch(cameraBuffer.$.pos, worldPos, instance)

    if (!rayHit.isHit) {
      if (debug) return DEBUG_MISS
      return { color: vec4f(0), depth: 1 }
    }

    const color = calcSurfaceLighting(
      Lighting({
        cameraPos: cameraBuffer.$.pos,
        lightPos: cameraBuffer.$.playerPos,
        surfacePos: rayHit.pos,
        normal: calcNormal(rayHit.pos, instance),
        surface: calcSurfaceColors(rayHit.pos, instance),
      }),
    )

    const hitClipPos = cameraBuffer.$.viewMatrix.mul(vec4f(rayHit.pos, 1))
    return { color: vec4f(color, 1), depth: hitClipPos.z / hitClipPos.w }
  })

  return { vertexProgram, fragmentProgram }
}
