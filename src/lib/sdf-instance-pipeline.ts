import tgpu, { type TgpuBuffer, type ValidateBufferSchema } from 'typegpu'
import {
  type BaseData,
  type Infer,
  type WgslArray,
  arrayOf,
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

import type { World } from '../main'
import { presentationFormat, sampleCount } from '../setup-webgpu'

import { cubeVertices } from './geometry'
import { Lighting, SurfaceColors, calcSurfaceLighting } from './lighting'
import { createPipelinePerformanceCallback } from './pipeline-perf'
import {
  blending,
  createColorAttachment,
  createDepthAttachment,
  depthStencil,
} from './web-gpu'

type RaymarchingConfig<T extends BaseData> = {
  /** Calculate the axis-aligned bounding box for an entity instance. */
  calcAABB: (entity: Infer<T>, elapsed: number) => AABB

  /** The signed distance function for the surface. Returns the distance from point to surface. */
  sdSurface: SdSurface<Infer<T>>

  /** Calculate the surface colors (diffuse, specular, emissive, etc.) at a given point on the surface. */
  calcSurfaceColors: (
    p: v3f,
    entity: Infer<T>,
    elapsed: number,
  ) => SurfaceColors

  /** Optional post-processing function for normals. Defaults to identity. */
  postProcessNormal?: (normal: v3f, p: v3f, entity: Infer<T>) => v3f

  /** Maximum number of raymarching steps. */
  maxSteps: number

  /** Maximum distance to march before considering a ray as missed. */
  maxDistance: number

  /** Distance threshold for considering a ray as having hit the surface. */
  epsilon: number

  /** Distance threshold for ray hits in normal calculation. Defaults to epsilon. */
  epsilonNormal?: number

  /** Enable debug visualization. This shows all AABBs for all rendered instances. */
  debug?: boolean
}

export type SdSurface<T, R = number> = (p: v3f, arg: T, elapsed: number) => R

export const AABB = struct({ min: vec3f, max: vec3f })
export type AABB = Infer<typeof AABB>

export const RayHit = struct({ isHit: bool, pos: vec3f })
export type RayHit = Infer<typeof RayHit>

const DEBUG_MISS = { color: vec4f(1, 0, 1, 1).mul(0.25), depth: 0 }

export function createSDFInstancesRenderer(world: World, name: string) {
  return {
    withBuffer<T extends BaseData>(
      instanceStruct: T,
      instanceCapacity: number,
      writeBuffers: (buffer: TgpuBuffer<WgslArray<T>>) => number,
    ) {
      return {
        withRaymarching(config: RaymarchingConfig<T>) {
          return {
            createRenderer() {
              const {
                calcAABB,
                sdSurface,
                calcSurfaceColors,
                maxSteps,
                maxDistance,
                epsilon,
              } = config

              const epsilonNormal = config.epsilonNormal ?? epsilon
              const postProcessNormal =
                config.postProcessNormal ??
                ((n) => {
                  'use gpu'
                  return vec3f(n)
                })
              const debug = config.debug ?? false

              const instanceBuffer = world.root
                .createBuffer(
                  arrayOf(
                    // @ts-expect-error ???
                    instanceStruct,
                    instanceCapacity,
                  ) as ValidateBufferSchema<WgslArray<T>>,
                )
                // @ts-expect-error ???
                .$usage('storage')

              // @ts-expect-error ???
              const instanceBufferReadonly = instanceBuffer.as('readonly')
              const cameraBuffer = world.camera.buffer.as('uniform')
              const timeBuffer = world.time.buffer.as('uniform')

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
                const instance = instanceBufferReadonly.$[
                  instanceIdx
                ] as Infer<T>

                const aabb = calcAABB(instance, timeBuffer.$.elapsed)
                const worldPos = select(
                  aabb.min,
                  aabb.max,
                  vec3b(local.x > 0, local.y > 0, local.z > 0),
                )

                const clipPos = cameraBuffer.$.viewMatrix.mul(
                  vec4f(worldPos, 1),
                )

                return {
                  worldPos,
                  clipPos,
                  instanceIdx: instanceIdx,
                }
              })

              const raymarch = createRaymarch({
                world,
                sdSurface,
                maxSteps,
                maxDistance,
                epsilon,
              })

              const calcNormal = createCalcNormal({
                world,
                sdSurface,
                epsilonNormal,
              })

              const fragmentProgram = createFragmentProgram({
                world,
                instanceBufferReadonly,
                raymarch,
                calcNormal,
                postProcessNormal,
                calcSurfaceColors,
                debug,
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
                .withPerformanceCallback(
                  createPipelinePerformanceCallback(name),
                )

              return function render() {
                const count = writeBuffers(instanceBuffer)
                if (count === 0) return

                pipeline
                  .withColorAttachment({ color: createColorAttachment(world) })
                  .withDepthStencilAttachment(createDepthAttachment(world))
                  .draw(cubeVertices.$.length, count)
              }
            },
          }
        },
      }
    },
  }
}

function createRaymarch<T extends BaseData>({
  world,
  sdSurface,
  maxSteps,
  maxDistance,
  epsilon,
}: {
  world: World
  sdSurface: SdSurface<Infer<T>>
  maxSteps: number
  maxDistance: number
  epsilon: number
}) {
  const timeBuffer = world.time.buffer.as('uniform')
  return function raymarch(
    cameraPos: v3f,
    worldPos: v3f,
    arg: Infer<T>,
  ): RayHit {
    'use gpu'

    const triDiff = worldPos.sub(cameraPos)
    let totalDistance = length(triDiff)
    const rayDirection = normalize(triDiff)

    for (let i = 0; i < maxSteps; i++) {
      const point = cameraPos.add(rayDirection.mul(totalDistance))
      const distance = sdSurface(point, arg, timeBuffer.$.elapsed)

      if (distance < epsilon) return RayHit({ isHit: true, pos: point })
      if (distance > maxDistance) break

      totalDistance += distance
    }

    return RayHit({ isHit: false, pos: vec3f() })
  }
}

function createCalcNormal<T extends BaseData>({
  world,
  sdSurface,
  epsilonNormal,
}: {
  world: World
  sdSurface: SdSurface<Infer<T>>
  epsilonNormal: number
}) {
  const timeBuffer = world.time.buffer.as('uniform')
  return function calcNormal(p: v3f, instance: Infer<T>): v3f {
    'use gpu'
    const k = vec2f(1, -1)
    return normalize(
      k.xyy
        .mul(
          sdSurface(
            p.add(k.xyy.mul(epsilonNormal)),
            instance,
            timeBuffer.$.elapsed,
          ),
        )
        .add(
          k.yyx
            .mul(
              sdSurface(
                p.add(k.yyx.mul(epsilonNormal)),
                instance,
                timeBuffer.$.elapsed,
              ),
            )
            .add(
              k.yxy
                .mul(
                  sdSurface(
                    p.add(k.yxy.mul(epsilonNormal)),
                    instance,
                    timeBuffer.$.elapsed,
                  ),
                )
                .add(
                  k.xxx.mul(
                    sdSurface(
                      p.add(k.xxx.mul(epsilonNormal)),
                      instance,
                      timeBuffer.$.elapsed,
                    ),
                  ),
                ),
            ),
        ),
    )
  }
}

function createFragmentProgram<T extends BaseData>({
  world,
  instanceBufferReadonly,
  raymarch,
  calcNormal,
  postProcessNormal,
  calcSurfaceColors,
  debug,
}: {
  world: World
  instanceBufferReadonly: ReturnType<TgpuBuffer<WgslArray<T>>['as']>
  raymarch: (cameraPos: v3f, worldPos: v3f, arg: Infer<T>) => RayHit
  calcNormal: (p: v3f, instance: Infer<T>) => v3f
  postProcessNormal: (normal: v3f, p: v3f, entity: Infer<T>) => v3f
  calcSurfaceColors: (
    p: v3f,
    entity: Infer<T>,
    elapsed: number,
  ) => SurfaceColors
  debug: boolean
}) {
  const cameraBuffer = world.camera.buffer.as('uniform')
  const timeBuffer = world.time.buffer.as('uniform')

  return tgpu['~unstable'].fragmentFn({
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

    const instance = instanceBufferReadonly.$[instanceIdx] as Infer<T>
    const rayHit = raymarch(cameraBuffer.$.pos, worldPos, instance)

    if (!rayHit.isHit) {
      if (debug) return DEBUG_MISS
      return { color: vec4f(0), depth: 1 }
    }

    const normal = postProcessNormal(
      calcNormal(rayHit.pos, instance),
      rayHit.pos,
      instance,
    )

    const color = calcSurfaceLighting(
      Lighting({
        cameraPos: cameraBuffer.$.pos,
        lightPos: cameraBuffer.$.playerPos,
        surfacePos: rayHit.pos,
        normal,
        surface: calcSurfaceColors(
          rayHit.pos,
          instance,
          timeBuffer.$.elapsed,
        ),
      }),
    )

    const hitClipPos = cameraBuffer.$.viewMatrix.mul(vec4f(rayHit.pos, 1))
    return {
      color: vec4f(color, 1),
      depth: hitClipPos.z / hitClipPos.w,
    }
  })
}
