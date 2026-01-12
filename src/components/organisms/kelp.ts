import { perlin3d } from '@typegpu/noise'
import { opExtrudeY, sdLine } from '@typegpu/sdf'
import { addEntity, query, set } from 'bitecs'
import tgpu, { type TgpuBufferUniform } from 'typegpu'
import {
  type Infer,
  bool,
  builtin,
  f32,
  struct,
  type v3f,
  vec2f,
  vec3f,
  vec4f,
} from 'typegpu/data'
import { abs, length, normalize, saturate, sin, smoothstep } from 'typegpu/std'

import { createInstanceBuffer } from '../../lib/buffers'
import { cubeVertex, cubeVertices } from '../../lib/geometry'
import { hsl2rgb } from '../../lib/hsl'
import { Lighting, Surface, calcSurfaceLighting } from '../../lib/lighting'
import { createPipelinePerformanceCallback } from '../../lib/pipeline-perf'
import { randomRange } from '../../lib/random'
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

type Kelp = {
  height: number
}
const Kelp = [] as Kelp[]

const KelpStruct = struct({
  entityPos: vec3f,
  height: f32,
  growth: f32,
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
  Kelp[eid] = { height: randomRange(3, 6) }
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

  const pipeline = world.root['~unstable']
    .withVertex(
      createVertexProgram(world.camera.buffer.as('uniform')),
      kelpsLayout.attrib,
    )
    .withFragment(createFragmentProgram(world.camera.buffer.as('uniform')), {
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
          height: Kelp[eid].height,
          entityPos: vec3f(GridPosition[eid], 0),
          growth: getLifetimeCompletion(world, eid),
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

function createVertexProgram(
  cameraBuffer: TgpuBufferUniform<typeof CameraStruct>,
) {
  return tgpu['~unstable'].vertexFn({
    in: {
      idx: builtin.vertexIndex,
      entityPos: vec3f,
      height: f32,
      growth: f32,
    },
    out: {
      localPos: vec3f,
      worldPos: vec3f,
      clipPos: builtin.position,

      entityPos: vec3f,
      height: f32,
      growth: f32,
    },
  })(({ idx, entityPos, height, growth }) => {
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
    }
  })
}

function createFragmentProgram(
  cameraBuffer: TgpuBufferUniform<typeof CameraStruct>,
) {
  const MAX_DISTANCE = f32(100)
  const MAX_STEPS = 100
  const EPSILON = 0.003

  const Hit = struct({ hit: bool, pos: vec3f, twistedP: vec3f })
  type Hit = Infer<typeof Hit>

  // Handy primes for randomy feeling things.
  const prime3k = Math.sqrt(3) * 1000
  const prime7k = Math.sqrt(7) * 1000
  const prime17k = Math.sqrt(17) * 1000

  const main = tgpu['~unstable'].fragmentFn({
    in: {
      localPos: vec3f,
      worldPos: vec3f,
      clipPos: builtin.position,

      entityPos: vec3f,
      height: f32,
      growth: f32,
    },
    out: {
      color: vec4f,
      depth: builtin.fragDepth,
    },
  })(({ worldPos, entityPos, height, growth }) => {
    const kelp = KelpStruct({ entityPos, height, growth })

    const hit = raymarch(worldPos, kelp)
    if (!hit.hit) return { color: vec4f(0), depth: 1 }

    const hitClipPos = cameraBuffer.$.viewMatrix.mul(vec4f(hit.pos, 1))

    const color = calcSurfaceLighting(
      Lighting({
        cameraPos: cameraBuffer.$.pos,
        lightPos: cameraBuffer.$.playerPos,
        surfacePos: hit.pos,
        normal: calcNormal(hit.pos, kelp),
        surface: calcSurfaceColors(hit.twistedP, kelp),
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
          0.6 + waveD * 0.5,
          0.1 + waveD,
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

  function raymarch(worldPos: v3f, kelp: KelpStruct): Hit {
    'use gpu'

    const triDiff = worldPos.sub(cameraBuffer.$.pos)
    let totalDistance = length(triDiff)
    const rayDirection = normalize(triDiff)

    for (let i = 0; i < MAX_STEPS; i++) {
      const point = cameraBuffer.$.pos.add(rayDirection.mul(totalDistance))
      const distance = scene(point, kelp)

      if (distance < EPSILON) {
        const twistedP = calcTwistedP(point, kelp)
        return Hit({ hit: true, pos: point, twistedP })
      }
      if (distance > MAX_DISTANCE) break

      totalDistance += distance
    }

    return Hit({ hit: false, pos: vec3f(), twistedP: vec3f() })
  }

  function calcTwistedP(p: v3f, kelp: KelpStruct): v3f {
    'use gpu'
    const localP = p.sub(kelp.entityPos)
    const twistAngle =
      localP.z + //
      kelp.entityPos.x * prime7k +
      kelp.entityPos.y * prime17k +
      kelp.height * prime3k +
      kelp.entityPos.x * kelp.entityPos.y * 0.123 // breaks symmetry

    return vec3f(rotate2d(localP.xy, twistAngle), localP.z)
  }

  function scene(p: v3f, kelp: KelpStruct): number {
    'use gpu'

    const twistedP = calcTwistedP(p, kelp)
    const narrowness = 1 - kelp.growth
    const height = kelp.height * kelp.growth

    const radius = narrowness * 0.4

    const flatRibbonD =
      sdLine(
        twistedP.xz,
        vec2f(0, -radius), //
        vec2f(0, height - radius),
      ) - radius

    const center = opExtrudeY(twistedP, flatRibbonD, 0.01)
    return center
  }

  function calcNormal(p: v3f, kelp: KelpStruct): v3f {
    'use gpu'
    const h = EPSILON
    const k = vec2f(1, -1)
    return normalize(
      k.xyy
        .mul(scene(p.add(k.xyy.mul(h)), kelp))
        .add(
          k.yyx
            .mul(scene(p.add(k.yyx.mul(h)), kelp))
            .add(
              k.yxy
                .mul(scene(p.add(k.yxy.mul(h)), kelp))
                .add(k.xxx.mul(scene(p.add(k.xxx.mul(h)), kelp))),
            ),
        ),
    )
  }

  return main
}
