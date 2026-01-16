import {
  opSmoothDifference,
  opSmoothUnion,
  opUnion,
  sdBox3d,
  sdCapsule,
  sdSphere,
} from '@typegpu/sdf'
import { addEntity, query, set } from 'bitecs'
import tgpu, { type TgpuBufferUniform } from 'typegpu'
import {
  type Infer,
  builtin,
  f32,
  struct,
  type v3f,
  vec2f,
  vec3f,
  vec4f,
} from 'typegpu/data'
import { abs, atan2, clamp, fract, max, sin, smoothstep } from 'typegpu/std'

import { createInstanceBuffer } from '../../lib/buffers'
import { easeInCubic, easeInSine, easeOutSine } from '../../lib/ease'
import { cubeVertex, cubeVertices } from '../../lib/geometry'
import { hsl2rgb } from '../../lib/hsl'
import { Lighting, Surface, calcSurfaceLighting } from '../../lib/lighting'
import { createPipelinePerformanceCallback } from '../../lib/pipeline-perf'
import { createCalcNormal, createRaymarch } from '../../lib/raymarching'
import { remap } from '../../lib/remap'
import { sdCone } from '../../lib/sdf'
import { rotate2d, rotateX } from '../../lib/transform'
import {
  blending,
  createColorAttachment,
  createDepthAttachment,
  depthStencil,
} from '../../lib/web-gpu'
import type { World } from '../../main'
import { presentationFormat, sampleCount } from '../../setup-webgpu'
import type { TimeStruct } from '../../time'
import { type CameraStruct, worldToClipSpace } from '../game/camera'
import { GridPosition, getRandomEmptyGridPosition } from '../general/grid'
import { Lifetime, getLifetimeCompletion } from '../general/lifetime'

const DEBUG = false

type Mushroom = {
  height: number
  lobes: number
  stemRadius: number
  capRadius: number
}
const Mushroom = [] as Mushroom[]

const MushroomStruct = struct({
  pos: vec3f,
  height: f32,
  lobes: f32,
  stemRadius: f32,
  capRadius: f32,
  alpha: f32,
  completion: f32,
})
type MushroomStruct = Infer<typeof MushroomStruct>

const SPAWN_RATE = 0.05

export function createMushroom(world: World) {
  const gridPosition = getRandomEmptyGridPosition(world)
  if (!gridPosition) return

  const eid = addEntity(
    world,
    set(GridPosition, gridPosition),
    set(Lifetime, Math.random() * 30 + 15),
    Mushroom,
  )
  Mushroom[eid] = {
    height: Math.random() * 1.5 + 0.5,
    lobes: Math.floor(Math.random() * 8) + 3,
    stemRadius: Math.random() * 0.04 + 0.04,
    capRadius: Math.random() * 0.7 + 0.3,
  }
}

export function spawnMushroomsSystem(world: World) {
  if (Math.random() < SPAWN_RATE) createMushroom(world)
}

export function createRenderMushroomSystem(world: World) {
  const [mushroomsBuffer, mushroomsLayout] = createInstanceBuffer(
    world,
    MushroomStruct,
    1000,
  )

  const pipeline = world.root['~unstable']
    .withVertex(
      createVertexProgram(world.camera.buffer.as('uniform')),
      mushroomsLayout.attrib,
    )
    .withFragment(
      createFragmentProgram(
        world.time.buffer.as('uniform'),
        world.camera.buffer.as('uniform'),
      ),
      { color: { format: presentationFormat, blend: blending.normal } },
    )
    .withDepthStencil(depthStencil)
    .withPrimitive({ topology: 'triangle-list', cullMode: 'back' })
    .withMultisample({ count: sampleCount })
    .createPipeline()
    .with(mushroomsLayout, mushroomsBuffer)
    .withPerformanceCallback(createPipelinePerformanceCallback('mushrooms'))

  function render(world: World) {
    const mushrooms = query(world, [Mushroom, GridPosition])
    if (mushrooms.length === 0) return

    mushroomsBuffer.writePartial(
      [...mushrooms].map((eid, idx) => {
        const completion = getLifetimeCompletion(world, eid)
        const growth = easeOutSine(completion)
        return {
          idx,
          value: {
            pos: vec3f(GridPosition[eid], 0),
            height: Mushroom[eid].height,
            lobes: Mushroom[eid].lobes,
            stemRadius: Mushroom[eid].stemRadius,
            capRadius: Mushroom[eid].capRadius,
            alpha: clamp(remap(growth, 0.9, 1, 1, 0), 0, 1),
            completion,
          },
        }
      }),
    )

    pipeline
      .withColorAttachment({ color: createColorAttachment(world) })
      .withDepthStencilAttachment(createDepthAttachment(world))
      .draw(cubeVertices.$.length, mushrooms.length)
  }

  return render
}

function createVertexProgram(
  cameraBuffer: TgpuBufferUniform<typeof CameraStruct>,
) {
  return tgpu['~unstable'].vertexFn({
    in: {
      idx: builtin.vertexIndex,
      pos: vec2f,
      height: f32,
      lobes: f32,
      stemRadius: f32,
      capRadius: f32,
      alpha: f32,
      completion: f32,
    },
    out: {
      localPos: vec3f,
      worldPos: vec3f,
      clipPos: builtin.position,

      entityPos: vec3f,
      height: f32,
      lobes: f32,
      stemRadius: f32,
      capRadius: f32,
      alpha: f32,
      completion: f32,
    },
  })(
    ({ idx, pos, height, lobes, stemRadius, capRadius, alpha, completion }) => {
      const growth = easeOutSine(completion)

      const scale = max(stemRadius * growth, capRadius * growth) * 1.5
      let localPos = cubeVertex(idx, scale, height * growth)

      const entityPos = vec3f(pos, 0)
      const worldPos = localPos.add(entityPos)
      const clipPos = worldToClipSpace(cameraBuffer.$, worldPos)

      return {
        localPos,
        worldPos,
        clipPos,

        entityPos,
        height,
        lobes,
        stemRadius,
        capRadius,
        alpha,
        completion,
      }
    },
  )
}

function createFragmentProgram(
  timeBuffer: TgpuBufferUniform<typeof TimeStruct>,
  cameraBuffer: TgpuBufferUniform<typeof CameraStruct>,
) {
  const MAX_DISTANCE = f32(100)
  const MAX_STEPS = 50
  const EPSILON = 0.001

  const main = tgpu['~unstable'].fragmentFn({
    in: {
      worldPos: vec3f,

      entityPos: vec3f,
      height: f32,
      lobes: f32,
      stemRadius: f32,
      capRadius: f32,
      alpha: f32,
      completion: f32,
    },
    out: {
      color: vec4f,
      depth: builtin.fragDepth,
    },
  })(
    ({
      worldPos,
      entityPos,
      height,
      lobes,
      stemRadius,
      capRadius,
      alpha,
      completion,
    }) => {
      const mushroom = MushroomStruct({
        pos: entityPos,
        height,
        lobes,
        stemRadius,
        capRadius,
        alpha,
        completion,
      })

      const hit = raymarch(cameraBuffer.$.pos, worldPos, mushroom)

      if (DEBUG && !hit.isHit)
        return { color: vec4f(1, 0, 1, 1).mul(0.25), depth: 0 }
      if (!hit.isHit) return { color: vec4f(0), depth: 1 }

      const hitClipPos = cameraBuffer.$.viewMatrix.mul(vec4f(hit.pos, 1))

      const normal = calcNormal(hit.pos, entityPos, mushroom)

      const litColor = calcSurfaceLighting(
        Lighting({
          cameraPos: cameraBuffer.$.pos,
          lightPos: cameraBuffer.$.playerPos,
          surfacePos: hit.pos,
          normal: normal,
          surface: calcSurfaceColors(hit.pos, mushroom),
        }),
      )

      return {
        color: vec4f(litColor, 1),
        depth: hitClipPos.z / hitClipPos.w,
      }
    },
  )

  function scene(p: v3f, mushroom: MushroomStruct): number {
    'use gpu'
    const localP = p.sub(mushroom.pos)

    const growth = easeInSine(mushroom.completion)
    const stemRadius = mushroom.stemRadius * growth
    const capRadius = mushroom.capRadius * growth
    const height = mushroom.height * growth

    const stem = opSmoothUnion(
      sdCapsule(
        localP,
        vec3f(0, 0, stemRadius),
        vec3f(0, 0, height - stemRadius),
        stemRadius,
      ),
      sdCone(
        rotateX(localP.sub(vec3f(0, 0, height - 0.2)), -Math.PI / 2),
        0.15,
        height - 0.2,
      ),
      0.15,
    )

    const capCenter = localP.sub(vec3f(0, 0, height - capRadius))
    let cap = sdSphere(capCenter, capRadius)
    cap = opSmoothDifference(
      cap,
      sdBox3d(localP, vec3f(1, 1, height - capRadius * 0.4)),
      0.05,
    )

    const expiryGrowth = clamp(
      remap(mushroom.completion, 0.65, f32(1), f32(0), f32(1)),
      f32(0),
      f32(1),
    )
    const expiry = sdCapsule(
      p,
      mushroom.pos.sub(vec3f(0, 0, 100)),
      mushroom.pos.add(
        vec3f(0, 0, -capRadius + (capRadius + height * 1.25) * expiryGrowth),
      ),
      capRadius * expiryGrowth,
    )

    // return expiry
    // return opUnion(stem, cap)
    return opSmoothDifference(opUnion(stem, cap), expiry, 0.1)
  }

  const raymarch = createRaymarch(scene, {
    maxSteps: MAX_STEPS,
    maxDistance: MAX_DISTANCE,
    epsilon: EPSILON,
  })

  const calcNormalBase = createCalcNormal(scene, EPSILON)

  function calcNormal(p: v3f, entityPos: v3f, mushroom: MushroomStruct): v3f {
    'use gpu'
    let normal = calcNormalBase(p, mushroom)

    const angle = atan2(p.y - entityPos.y, p.x - entityPos.x)
    normal = vec3f(
      rotate2d(normal.xy, sin(angle * mushroom.lobes) * 0.3),
      normal.z,
    )
    return normal
  }

  function calcSurfaceColors(hitPos: v3f, mushroom: MushroomStruct): Surface {
    'use gpu'

    const baseHueShift =
      (fract(mushroom.pos.x * 1.1577 + mushroom.pos.y * 3.7193) * 2 - 1) * 0.1
    const baseColor = hsl2rgb(vec3f(0.66 + baseHueShift, 0.4, 0.4))

    const angle = atan2(hitPos.y - mushroom.pos.y, hitPos.x - mushroom.pos.x)

    const glowZFactor = easeInCubic(
      hitPos.z / (mushroom.height * mushroom.completion),
    )
    const glowZ = glowZFactor * (mushroom.height * mushroom.completion)

    let glowValue =
      (glowZ + sin(angle * mushroom.lobes) * 0.01) * 2 +
      timeBuffer.$.elapsed * 0.1 +
      mushroom.pos.x +
      mushroom.pos.y
    glowValue = abs(fract(glowValue) - 0.5) * 2

    let glowWhiteValue = smoothstep(0.95, 1, glowValue)
    glowValue = smoothstep(0.8, 1, glowValue)

    const glowHue = fract(
      hitPos.z +
        mushroom.pos.x * 3.77 +
        mushroom.pos.y * 5.37 +
        timeBuffer.$.elapsed * 0.1,
    )
    const glowColor = hsl2rgb(vec3f(glowHue, 1, 0.6))
      .mul(glowValue)
      .add(vec3f(glowWhiteValue))
      .mul(easeInCubic(hitPos.z / mushroom.height))

    return Surface({
      diffuse: baseColor,
      specular: vec3f(1, 0.5, 0),
      emissive: glowColor,
      shininess: f32(64),
    })
  }

  return main
}
