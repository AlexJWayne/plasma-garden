import { perlin3d, randf } from '@typegpu/noise'
import { opExtrudeY, opSmoothUnion, sdLine } from '@typegpu/sdf'
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
import { abs, saturate, sin, smoothstep } from 'typegpu/std'

import { createInstanceBuffer } from '../../lib/buffers'
import { cubeVertex, cubeVertices } from '../../lib/geometry'
import { hsl2rgb } from '../../lib/hsl'
import { Lighting, Surface, calcSurfaceLighting } from '../../lib/lighting'
import { createPipelinePerformanceCallback } from '../../lib/pipeline-perf'
import { randomRange } from '../../lib/random'
import { createCalcNormal, createRaymarch } from '../../lib/raymarching'
import { sdLink } from '../../lib/sdf'
import { rotate2d } from '../../lib/transform'
import {
  blending,
  createColorAttachment,
  createDepthAttachment,
  depthStencil,
} from '../../lib/web-gpu'
import type { World } from '../../main'
import { presentationFormat, sampleCount } from '../../setup-webgpu'
import { type CameraStruct, worldToClipSpace } from '../game/camera'
import { GridPosition, getRandomEmptyGridPosition } from '../general/grid'
import { Lifetime, getLifetimeCompletion } from '../general/lifetime'

const DEBUG = false

type Kelp = {
  height: number
  twist: number
  seed: number
}
const Kelp = [] as Kelp[]

const KelpStruct = struct({
  entityPos: vec3f,
  height: f32,
  twist: f32,
  growth: f32,
  seed: f32,
})
type KelpStruct = Infer<typeof KelpStruct>

const SPAWN_RATE = 0.05

export function createKelp(world: World): void {
  const gridPosition = getRandomEmptyGridPosition(world)
  if (!gridPosition) return

  const eid = addEntity(
    world,
    set(GridPosition, gridPosition),
    set(Lifetime, randomRange(10, 30)),
    Kelp,
  )
  Kelp[eid] = {
    height: randomRange(3, 6),
    twist: randomRange(0.75, 2) * (Math.random() > 0.5 ? 1 : -1),
    seed: Math.random(),
  }
  console.log(Kelp[eid].twist)
}

export function spawnKelpSystem(world: World): void {
  if (Math.random() < SPAWN_RATE) createKelp(world)
}

export function createRenderKelpSystem(world: World) {
  const [kelpsBuffer, kelpsLayout] = createInstanceBuffer(
    world,
    KelpStruct,
    1000,
  )

  const { vertexProgram, fragmentProgram } = createShaderProgram(
    world.camera.buffer.as('uniform'),
  )

  const pipeline = world.root['~unstable']
    .withVertex(vertexProgram, kelpsLayout.attrib)
    .withFragment(fragmentProgram, {
      color: { format: presentationFormat, blend: blending.normal },
    })
    .withDepthStencil(depthStencil)
    .withPrimitive({ topology: 'triangle-list', cullMode: 'back' })
    .withMultisample({ count: sampleCount })
    .createPipeline()
    .with(kelpsLayout, kelpsBuffer)
    .withPerformanceCallback(createPipelinePerformanceCallback('kelps'))

  function render(world: World) {
    const kelps = query(world, [Kelp, GridPosition, Lifetime])
    if (kelps.length === 0) return

    kelpsBuffer.writePartial(
      [...kelps].map((eid, idx) => ({
        idx,
        value: {
          entityPos: vec3f(GridPosition[eid], 0),
          height: Kelp[eid].height,
          growth: getLifetimeCompletion(world, eid),
          twist: Kelp[eid].twist,
          seed: Kelp[eid].seed,
        },
      })),
    )

    pipeline
      .withColorAttachment({ color: createColorAttachment(world) })
      .withDepthStencilAttachment(createDepthAttachment(world))
      .draw(cubeVertices.$.length, kelps.length)
  }

  return render
}

function createShaderProgram(
  cameraBuffer: TgpuBufferUniform<typeof CameraStruct>,
) {
  const vertexProgram = tgpu['~unstable'].vertexFn({
    in: {
      idx: builtin.vertexIndex,
      entityPos: vec3f,
      height: f32,
      growth: f32,
      twist: f32,
      seed: f32,
    },
    out: {
      localPos: vec3f,
      worldPos: vec3f,
      clipPos: builtin.position,

      entityPos: vec3f,
      height: f32,
      growth: f32,
      seed: f32,
      twist: f32,
    },
  })(({ idx, entityPos, height, growth, twist, seed }) => {
    const localPos = cubeVertex(idx, 0.9 * (1 - growth), height * growth)
    const worldPos = localPos.add(vec3f(entityPos))
    const clipPos = worldToClipSpace(cameraBuffer.$, worldPos)
    return {
      localPos,
      worldPos,
      clipPos,

      entityPos,
      height,
      growth,
      seed,
      twist,
    }
  })

  function sdSurface(p: v3f, kelp: KelpStruct): number {
    'use gpu'

    const twistedP = calcTwistedP(p, kelp)
    const narrowness = 1 - kelp.growth
    const height = kelp.height * kelp.growth

    const linkThickness = 0.03
    const radius = narrowness * 0.4 - linkThickness

    const membrane2d =
      sdLine(
        twistedP.xz,
        vec2f(0, -radius),
        vec2f(0, height - radius - linkThickness),
      ) - radius
    const membrane = opExtrudeY(twistedP, membrane2d, 0.01)

    const membraneCenter = height * 0.5 - radius - linkThickness
    const centeredP = twistedP.sub(vec3f(0, 0, membraneCenter))
    const offsetP = vec3f(centeredP.xzy)
    const border = sdLink(offsetP, height * 0.5, radius, linkThickness)

    return opSmoothUnion(membrane, border, 0.07)
  }

  const raymarch = createRaymarch(sdSurface, {
    maxSteps: 100,
    maxDistance: 100,
    epsilon: 0.003,
  })
  const calcNormal = createCalcNormal(sdSurface, 0.01)

  const fragmentProgram = tgpu['~unstable'].fragmentFn({
    in: {
      localPos: vec3f,
      worldPos: vec3f,
      clipPos: builtin.position,

      entityPos: vec3f,
      height: f32,
      growth: f32,
      twist: f32,
      seed: f32,
    },
    out: {
      color: vec4f,
      depth: builtin.fragDepth,
    },
  })(({ worldPos, entityPos, height, growth, twist, seed }) => {
    randf.seed(seed)
    const kelp = KelpStruct({ entityPos, height, growth, twist, seed })

    const hit = raymarch(cameraBuffer.$.pos, worldPos, kelp)

    if (DEBUG && !hit.isHit)
      return { color: vec4f(1, 0, 1, 1).mul(0.25), depth: 0 }

    if (!hit.isHit) return { color: vec4f(0), depth: 1 }

    const hitClipPos = cameraBuffer.$.viewMatrix.mul(vec4f(hit.pos, 1))

    const color = calcSurfaceLighting(
      Lighting({
        cameraPos: cameraBuffer.$.pos,
        lightPos: cameraBuffer.$.playerPos,
        surfacePos: hit.pos,
        normal: calcNormal(hit.pos, kelp),
        surface: calcSurfaceColors(calcTwistedP(hit.pos, kelp), kelp),
      }),
    )

    return {
      color: vec4f(color, 1),
      depth: hitClipPos.z / hitClipPos.w,
    }
  })

  function calcSurfaceColors(twistedP: v3f, kelp: KelpStruct): Surface {
    'use gpu'

    const p = twistedP.xz

    const amplitude = sin((p.y - kelp.growth * 2) * 20) * 0.025
    const waveD = abs(p.x - amplitude)
    const waveGlow = smoothstep(0.1, 0, waveD) ** 8

    const noise = perlin3d.sample(
      vec3f(
        p.x, //
        p.y - kelp.growth * 5,
        kelp.growth * 0.25,
      ).mul(12),
    )

    return Surface({
      diffuse: hsl2rgb(
        vec3f(
          0.33, //
          0.4 + waveD * 0.5,
          0.08 + waveD,
        ),
      ),
      specular: hsl2rgb(vec3f(0.5, 0.5, saturate(noise))),
      emissive: hsl2rgb(
        vec3f(
          0.6,
          0.7,
          saturate(waveGlow + smoothstep(0.0, 0.7, noise)) ** 4,
        ).mul(0.8),
      ),
      shininess: f32(32),
    })
  }

  function calcTwistedP(p: v3f, kelp: KelpStruct): v3f {
    'use gpu'

    const localP = p.sub(kelp.entityPos)
    const twistAngle = localP.z * kelp.twist + (Math.PI / 2) * kelp.seed

    return vec3f(rotate2d(localP.xy, twistAngle), localP.z)
  }

  return { vertexProgram, fragmentProgram }
}
