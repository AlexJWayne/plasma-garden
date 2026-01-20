import tgpu, { type TgpuBufferReadonly, type TgpuBufferUniform } from 'typegpu'
import {
  type BaseData,
  type Infer,
  type WgslArray,
  bool,
  builtin,
  interpolate,
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
import type { World } from '../main'
import { presentationFormat, sampleCount } from '../setup-webgpu'

import { cubeVertices } from './geometry'
import { Lighting, SurfaceColors, calcSurfaceLighting } from './lighting'
import { createPipelinePerformanceCallback } from './pipeline-perf'
import { type SdSurface } from './raymarching'
import {
  blending,
  createColorAttachment,
  createDepthAttachment,
  depthStencil,
} from './web-gpu'

export function createSdfSurfaceShaders() {}

export const AABB = struct({ min: vec3f, max: vec3f })
export type AABB = Infer<typeof AABB>

export const RayHit = struct({ isHit: bool, pos: vec3f })
export type RayHit = Infer<typeof RayHit>

const DEBUG_MISS = { color: vec4f(1, 0, 1, 1).mul(0.25), depth: 0 }

export function createSDFInstancesRenderer<T extends BaseData>({
  name,
  world,

  cameraBuffer,
  instanceBuffer,

  writeBuffers,
  calcAABB,
  sdSurface,
  calcSurfaceColors,

  maxSteps,
  maxDistance,
  epsilon,
  epsilonNormal = epsilon,

  debug = false,
}: {
  /** The TypeGPU world instance. */
  world: World

  /** Name of the pipeline for debugging and performance tracking. */
  name: string

  /** The global camera buffer. */
  cameraBuffer: TgpuBufferUniform<typeof CameraStruct>

  /** The instance data. Should be an array of structs, one item per instance. */
  instanceBuffer: TgpuBufferReadonly<WgslArray<T>>

  /** Query the world's entities and write to `instanceBuffer`. Return the number of instances to render. */
  writeBuffers: () => number

  /** Calculate the axis-aligned bounding box for an entity instance. */
  calcAABB: (entity: Infer<T>) => AABB

  /** The signed distance function for the surface. Returns the distance from point to surface. */
  sdSurface: SdSurface<Infer<T>>

  /** Calculate the surface colors (diffuse, specular, emissive, etc.) at a given point on the surface. */
  calcSurfaceColors: (p: v3f, entity: Infer<T>) => SurfaceColors

  /** Maximum number of raymarching steps. */
  maxSteps: number

  /** Maximum distance to march before considering a ray as missed. */
  maxDistance: number

  /** Distance threshold for considering a ray as having hit the surface. */
  epsilon: number

  /** Distance threshold for ray hits in normal calculation. */
  epsilonNormal: number

  /** Enable debug visualization. This shows all AABBs for all rendered instances. */
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

  const pipeline = world.root['~unstable']
    .withVertex(vertexProgram)
    .withFragment(fragmentProgram, {
      color: { format: presentationFormat, blend: blending.normal },
    })
    .withDepthStencil(depthStencil)
    .withPrimitive({ topology: 'triangle-list', cullMode: 'back' })
    .withMultisample({ count: sampleCount })
    .createPipeline()
    .withPerformanceCallback(createPipelinePerformanceCallback(name))

  return function render() {
    const count = writeBuffers()
    pipeline
      .withColorAttachment({ color: createColorAttachment(world) })
      .withDepthStencilAttachment(createDepthAttachment(world))
      .draw(cubeVertices.$.length, count)
  }
}
