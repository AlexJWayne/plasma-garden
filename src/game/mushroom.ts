import {
  opSmoothDifference,
  opSmoothUnion,
  opUnion,
  sdBox3d,
  sdCapsule,
  sdSphere,
} from '@typegpu/sdf'
import { addEntity, query, removeEntity } from 'bitecs'
import tgpu, { type TgpuBufferUniform } from 'typegpu'
import {
  type Infer,
  bool,
  builtin,
  f32,
  struct,
  type v2f,
  type v3f,
  vec2f,
  vec3f,
  vec4f,
} from 'typegpu/data'
import {
  abs,
  atan2,
  clamp,
  dot,
  fract,
  length,
  max,
  normalize,
  pow,
  reflect,
  sin,
  smoothstep,
} from 'typegpu/std'

import { createInstanceBuffer } from '../lib/buffers'
import { easeInCubic, easeInExpo, easeOutSine } from '../lib/ease'
import { cubeVertices } from '../lib/geometry'
import { hsl2rgb } from '../lib/hsl'
import { remap } from '../lib/remap'
import { sdCone } from '../lib/sdf'
import { rotate2d, rotateX } from '../lib/transform'
import {
  blending,
  createColorAttachment,
  createDepthAttachment,
  depthStencilTransparent,
} from '../lib/web-gpu'
import type { World } from '../main'
import { presentationFormat, sampleCount } from '../setup-webgpu'

import { CAMERA_OFFSET, type CameraStruct } from './camera'
import { Lifetime, Position, getLifetimeCompletion } from './components'
import { TimeStruct } from './time'

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
})
type MushroomStruct = Infer<typeof MushroomStruct>

export function createMushroom(
  world: World,
  pos: v2f,
  height: number,
  lobes: number,
) {
  const eid = addEntity(world, Position, Mushroom, Lifetime)
  Position[eid] = pos
  Mushroom[eid] = {
    height,
    lobes,
    stemRadius: Math.random() * 0.04 + 0.04,
    capRadius: Math.random() * 0.2 + 0.3,
  }
  Lifetime[eid] = {
    bornAt: world.time.elapsed,
    duration: Math.random() * 30 + 15,
  }
}

export function spawnMushroomsSystem(world: World) {
  if (Math.random() < 0.2) {
    createMushroom(
      world,
      vec2f(
        (Math.random() * 2 - 1) * 10, //
        (Math.random() * 2 - 1) * 10,
      ),
      Math.random() * 1.5 + 0.5,
      Math.floor(Math.random() * 8) + 3,
    )
  }
}

export function expireMushroomsSystem(world: World) {
  const mushrooms = query(world, [Mushroom, Lifetime])
  for (const eid of mushrooms) {
    const { bornAt, duration } = Lifetime[eid]
    if (world.time.elapsed > bornAt + duration) removeEntity(world, eid)
  }
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
      {
        color: {
          format: presentationFormat,
          blend: blending.normal,
        },
      },
    )
    .withDepthStencil(depthStencilTransparent)
    .withPrimitive({ topology: 'triangle-list', cullMode: 'back' })
    .withMultisample({ count: sampleCount })
    .createPipeline()
    .with(mushroomsLayout, mushroomsBuffer)

  function render(world: World) {
    const mushrooms = query(world, [Mushroom, Position])
    if (mushrooms.length === 0) return

    const cameraPos = world.camera.target.current.add(CAMERA_OFFSET)
    const sortedMushrooms = [...mushrooms].sort((a, b) => {
      const distA = length(cameraPos.sub(vec3f(Position[a], 0)))
      const distB = length(cameraPos.sub(vec3f(Position[b], 0)))
      return distB - distA
    })

    mushroomsBuffer.writePartial(
      sortedMushrooms.map((eid, idx) => {
        const growth = easeOutSine(getLifetimeCompletion(world, eid))
        return {
          idx,
          value: {
            pos: vec3f(Position[eid], 0),
            height: Mushroom[eid].height * growth,
            lobes: Mushroom[eid].lobes,
            stemRadius: Mushroom[eid].stemRadius * growth,
            capRadius: Mushroom[eid].capRadius * growth,
            alpha: clamp(remap(growth, 0.9, 1, 1, 0), 0, 1),
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
    },
  })(({ idx, pos, height, lobes, stemRadius, capRadius, alpha }) => {
    const scale = max(stemRadius, capRadius) * 1.5
    let localPos = cubeVertices.$[idx].mul(0.5)
    localPos.z += 0.5
    localPos = localPos.mul(vec3f(scale, scale, height))

    const entityPos = vec3f(pos, 0)
    const worldPos = localPos.add(entityPos)
    const clipPos = cameraBuffer.$.viewMatrix.mul(vec4f(worldPos, 1))
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
    }
  })
}

function createFragmentProgram(
  timeBuffer: TgpuBufferUniform<typeof TimeStruct>,
  cameraBuffer: TgpuBufferUniform<typeof CameraStruct>,
) {
  const MAX_DISTANCE = f32(100)
  const MAX_STEPS = 100
  const EPSILON = 0.0001

  const Hit = struct({ hit: bool, pos: vec3f })
  type Hit = Infer<typeof Hit>

  const main = tgpu['~unstable'].fragmentFn({
    in: {
      worldPos: vec3f,

      entityPos: vec3f,
      height: f32,
      lobes: f32,
      stemRadius: f32,
      capRadius: f32,
      alpha: f32,
    },
    out: { color: vec4f, depth: builtin.fragDepth },
  })(({ worldPos, entityPos, height, lobes, stemRadius, capRadius, alpha }) => {
    const mushroom = MushroomStruct({
      pos: entityPos,
      height,
      lobes,
      stemRadius,
      capRadius,
      alpha,
    })

    const hit = raymarch(worldPos, entityPos, mushroom)

    if (DEBUG && !hit.hit)
      return { color: vec4f(1, 0, 1, 1).mul(0.25), depth: 0 }
    if (!hit.hit) return { color: vec4f(0), depth: 1 }

    const hitClipPos = cameraBuffer.$.viewMatrix.mul(vec4f(hit.pos, 1))

    const normal = calcNormal(hit.pos, entityPos, mushroom)
    const diffuseValue = calcLighting(normal, hit.pos)
    const color = calcColor(diffuseValue, 0.01, hit.pos, mushroom)

    return {
      color: vec4f(color, 1).mul(alpha),
      depth: hitClipPos.z / hitClipPos.w,
    }
  })

  function raymarch(
    worldPos: v3f,
    entityPos: v3f,
    mushroom: MushroomStruct,
  ): Hit {
    'use gpu'

    const triDiff = worldPos.sub(cameraBuffer.$.pos)
    let totalDistance = length(triDiff)
    const rayDirection = normalize(triDiff)

    for (let i = 0; i < MAX_STEPS; i++) {
      const point = cameraBuffer.$.pos.add(rayDirection.mul(totalDistance))
      const distance = scene(point, entityPos, mushroom)

      if (distance < EPSILON) return Hit({ hit: true, pos: point })
      if (distance > MAX_DISTANCE) break

      totalDistance += distance
    }

    return Hit({ hit: false, pos: vec3f() })
  }

  function scene(p: v3f, entityPos: v3f, mushroom: MushroomStruct): number {
    'use gpu'
    const localP = p.sub(entityPos)

    const stalk = opSmoothUnion(
      sdCapsule(
        localP,
        vec3f(0, 0, mushroom.stemRadius),
        vec3f(0, 0, mushroom.height - mushroom.stemRadius),
        mushroom.stemRadius,
      ),
      sdCone(
        rotateX(localP.sub(vec3f(0, 0, mushroom.height - 0.2)), -Math.PI / 2),
        0.15,
        mushroom.height - 0.2,
      ),
      0.15,
    )

    const capCenter = localP.sub(
      vec3f(0, 0, mushroom.height - mushroom.capRadius),
    )
    let cap = sdSphere(capCenter, mushroom.capRadius)
    cap = opSmoothDifference(
      cap,
      sdBox3d(localP, vec3f(1, 1, mushroom.height - mushroom.capRadius * 0.4)),
      0.05,
    )

    return opUnion(stalk, cap)
  }

  function calcNormal(p: v3f, entityPos: v3f, mushroom: MushroomStruct): v3f {
    'use gpu'
    const h = EPSILON
    const k = vec2f(1, -1)
    let normal = normalize(
      k.xyy
        .mul(scene(p.add(k.xyy.mul(h)), entityPos, mushroom))
        .add(
          k.yyx
            .mul(scene(p.add(k.yyx.mul(h)), entityPos, mushroom))
            .add(
              k.yxy
                .mul(scene(p.add(k.yxy.mul(h)), entityPos, mushroom))
                .add(
                  k.xxx.mul(scene(p.add(k.xxx.mul(h)), entityPos, mushroom)),
                ),
            ),
        ),
    )

    const angle = atan2(p.y - entityPos.y, p.x - entityPos.x)
    normal = vec3f(
      rotate2d(normal.xy, sin(angle * mushroom.lobes) * 0.3),
      normal.z,
    )
    return normal
  }

  function calcLighting(normal: v3f, hitPos: v3f): number {
    'use gpu'
    const lightPos = cameraBuffer.$.targetPos
    const lightDistance = lightPos.sub(hitPos)
    const lightDir = normalize(lightDistance)
    const diffuse =
      max(dot(normal, lightDir), 0) *
      remap(length(lightDistance), f32(2), f32(15), f32(1), f32(0))
    const viewDir = normalize(cameraBuffer.$.pos.sub(hitPos))
    const specular = pow(
      max(dot(reflect(lightDir.mul(-1), normal), viewDir), 0),
      32,
    )
    return diffuse * 0.5 + specular
  }

  function calcColor(
    diffuse: number,
    ambient: number,
    hitPos: v3f,
    mushroom: MushroomStruct,
  ): v3f {
    'use gpu'

    const baseColor = vec3f(0.2, 0.3, 0.7)
    const diffuseColor = baseColor.mul(ambient + (1 - ambient) * diffuse)

    const angle = atan2(hitPos.y - mushroom.pos.y, hitPos.x - mushroom.pos.x)

    const glowZFactor = easeInExpo(hitPos.z / mushroom.height)
    const glowZ = glowZFactor * mushroom.height

    let glowValue =
      (glowZ + sin(angle * mushroom.lobes) * 0.01) * 0.7 +
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

    return diffuseColor.add(glowColor)
  }

  return main
}
