import {
  type Infer,
  f32,
  type m4x4f,
  struct,
  type v3f,
  vec3b,
  vec3f,
  vec4f,
} from 'typegpu/data'
import { select } from 'typegpu/std'

import type { CameraStruct } from '../components/game/camera'

import { cubeVertices } from './geometry'
import { Lighting, SurfaceColors, calcSurfaceLighting } from './lighting'
import { type SdSurface, createCalcNormal, createRaymarch } from './raymarching'

export function createSdfSurfaceShaders() {}

export const AABB = struct({ min: vec3f, max: vec3f })
export type AABB = Infer<typeof AABB>

export const EntityPositions = struct({
  local: vec3f,
  world: vec3f,
  clip: vec4f,
})
export type EntityPositions = Infer<typeof EntityPositions>

/** Returns the world and clip positions of a vertex index and an AABB. */
export function getVertexResult(
  cameraViewMatrix: m4x4f,
  aabb: AABB,
  vertexIdx: number,
): EntityPositions {
  'use gpu'

  const local = cubeVertices.$[vertexIdx]

  const world = select(
    aabb.min,
    aabb.max,
    vec3b(local.x > 0, local.y > 0, local.z > 0),
  )

  const clip = cameraViewMatrix.mul(vec4f(world, 1))

  return EntityPositions({ local, world, clip })
}

const FragmentResult = struct({ color: vec4f, depth: f32 })
export function createGetFragmentResult<T>({
  sdSurface,
  calcSurfaceColors,

  maxSteps,
  maxDistance,
  epsilon,
  epsilonNormal = epsilon,

  debug = false,
}: {
  sdSurface: SdSurface<T>
  calcSurfaceColors: (p: v3f, entity: T) => SurfaceColors

  maxSteps: number
  maxDistance: number
  epsilon: number
  epsilonNormal: number

  debug?: boolean
}) {
  const raymarch = createRaymarch(sdSurface, { maxSteps, maxDistance, epsilon })
  const calcNormal = createCalcNormal(sdSurface, epsilonNormal)

  return function getFragmentResult(
    cameraBuffer: Infer<typeof CameraStruct>,
    worldPos: v3f,
    entity: T,
  ) {
    'use gpu'

    const rayHit = raymarch(cameraBuffer.pos, worldPos, entity)

    if (!rayHit.isHit) {
      if (debug) {
        return FragmentResult({
          color: vec4f(1, 0, 1, 1).mul(0.25),
          depth: 0,
        })
      }

      return FragmentResult({ color: vec4f(0), depth: 1 })
    }

    const hitClipPos = cameraBuffer.viewMatrix.mul(vec4f(rayHit.pos, 1))

    const color = calcSurfaceLighting(
      Lighting({
        cameraPos: cameraBuffer.pos,
        lightPos: cameraBuffer.playerPos,
        surfacePos: rayHit.pos,
        normal: calcNormal(rayHit.pos, entity),
        surface: calcSurfaceColors(rayHit.pos, entity),
      }),
    )

    return FragmentResult({
      color: vec4f(color, 1),
      depth: hitClipPos.z / hitClipPos.w,
    })
  }
}
