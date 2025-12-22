import {
  opSmoothDifference,
  opUnion,
  sdBox3d,
  sdCapsule,
  sdSphere,
} from '@typegpu/sdf'
import { addEntity, query } from 'bitecs'
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
import { fract, length, normalize } from 'typegpu/std'

import { createInstanceBuffer } from '../lib/buffers'
import { cubeVertices } from '../lib/geometry'
import {
  blending,
  createColorAttachment,
  createDepthAttachment,
  depthStencil,
} from '../lib/web-gpu'
import type { World } from '../main'
import { presentationFormat, sampleCount } from '../setup-webgpu'

import type { CameraStruct } from './camera'
import { Position } from './components'
import { TimeStruct } from './time'

const DEBUG = false

type Mushroom = {
  height: number
}
const Mushroom = [] as Mushroom[]

const MushroomStruct = struct({
  pos: vec2f,
  height: f32,
})

export function createMushroom(world: World, pos: v2f, height: number) {
  const eid = addEntity(world, Position, Mushroom)
  Position[eid] = pos
  Mushroom[eid] = { height }
}

export function createRenderMushroomSystem(world: World) {
  const [mushroomsBuffer, mushroomsLayout] = createInstanceBuffer(
    world,
    MushroomStruct,
    100,
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
    .withDepthStencil(depthStencil)
    .withPrimitive({ topology: 'triangle-list', cullMode: 'back' })
    .withMultisample({ count: sampleCount })
    .createPipeline()
    .with(mushroomsLayout, mushroomsBuffer)

  function render(world: World) {
    const mushrooms = query(world, [Mushroom, Position])
    if (mushrooms.length === 0) return

    mushroomsBuffer.writePartial(
      [...mushrooms].map((eid, idx) => ({
        idx,
        value: {
          pos: Position[eid],
          height: Mushroom[eid].height,
        },
      })),
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
    },
    out: {
      localPos: vec3f,
      worldPos: vec3f,
      clipPos: builtin.position,
      entityPos: vec3f,
      height: f32,
    },
  })(({ idx, pos, height }) => {
    let localPos = cubeVertices.$[idx].mul(0.5)
    localPos.z += 0.5
    localPos.z *= height

    const entityPos = vec3f(pos, 0)
    const worldPos = localPos.add(entityPos)
    const clipPos = cameraBuffer.$.viewMatrix.mul(vec4f(worldPos, 1))
    return {
      localPos,
      worldPos,
      height,
      clipPos,
      entityPos,
    }
  })
}

function createFragmentProgram(
  timeBuffer: TgpuBufferUniform<typeof TimeStruct>,
  cameraBuffer: TgpuBufferUniform<typeof CameraStruct>,
) {
  const Hit = struct({ hit: bool, pos: vec3f })
  type Hit = Infer<typeof Hit>

  const main = tgpu['~unstable'].fragmentFn({
    in: {
      localPos: vec3f,
      worldPos: vec3f,
      entityPos: vec3f,
      height: f32,
    },
    out: { color: vec4f, depth: builtin.fragDepth },
  })(({ localPos, worldPos, entityPos, height }) => {
    const hit = raymarch(worldPos, entityPos, height)

    if (DEBUG && !hit.hit)
      return { color: vec4f(1, 0, 1, 1).mul(0.25), depth: 0 }
    if (!hit.hit) return { color: vec4f(0), depth: 1 }

    const hitClipPos = cameraBuffer.$.viewMatrix.mul(vec4f(hit.pos, 1))
    return {
      color: vec4f(
        fract(vec3f(hit.pos.z + timeBuffer.$.elapsed * 0.05).mul(10)),
        1,
      ),
      depth: hitClipPos.z / hitClipPos.w,
    }
  })

  function raymarch(worldPos: v3f, entityPos: v3f, height: number): Hit {
    'use gpu'

    const MAX_DISTANCE = f32(100)
    const MAX_STEPS = 100
    const EPSILON = 0.0001

    const triDiff = worldPos.sub(cameraBuffer.$.pos)
    let totalDistance = length(triDiff)
    const rayDirection = normalize(triDiff)

    for (let i = 0; i < MAX_STEPS; i++) {
      const point = cameraBuffer.$.pos.add(rayDirection.mul(totalDistance))
      const distance = scene(point, entityPos, height)

      if (distance < EPSILON) return Hit({ hit: true, pos: point })
      if (distance > MAX_DISTANCE) break

      totalDistance += distance
    }

    return Hit({ hit: false, pos: vec3f() })
  }

  function scene(p: v3f, entityPos: v3f, height: number): number {
    'use gpu'
    const stalkR = 0.05
    const localP = p.sub(entityPos)

    const stalk = sdCapsule(
      localP,
      vec3f(0, 0, stalkR),
      vec3f(0, 0, height - stalkR),
      stalkR,
    )

    const capR = 0.3
    const capCenter = localP.sub(vec3f(0, 0, height - capR))
    let cap = sdSphere(capCenter, capR)
    cap = opSmoothDifference(
      cap,
      sdBox3d(localP, vec3f(1, 1, height - capR)),
      0.05,
    )

    return opUnion(stalk, cap)
  }

  return main
}
