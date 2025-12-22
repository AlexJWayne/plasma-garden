import { addEntity, query } from 'bitecs'
import tgpu, { type TgpuBufferUniform } from 'typegpu'
import {
  builtin,
  f32,
  struct,
  type v2f,
  vec2f,
  vec3f,
  vec4f,
} from 'typegpu/data'
import { abs, fract } from 'typegpu/std'

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
import type { TimeStruct } from './time'

type Mushroom = {
  height: number
}
const Mushroom = [] as Mushroom[]

const MushroomStruct = struct({
  pos: vec2f,
  height: f32,
})

export function createMushroom(world: World, pos: v2f) {
  const eid = addEntity(world, Position, Mushroom)
  Position[eid] = pos
  Mushroom[eid] = { height: 1.5 }
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
    .withFragment(createFragmentProgram(world.time.buffer.as('uniform')), {
      format: presentationFormat,
      blend: blending.normal,
    })
    .withDepthStencil(depthStencil)
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
      .withColorAttachment(createColorAttachment(world))
      .withDepthStencilAttachment(createDepthAttachment(world))
      .draw(36, mushrooms.length)
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
      height: f32,
    },
  })(({ idx, pos, height }) => {
    let localPos = cubeVertices.$[idx].mul(0.5)
    localPos.z += 0.5
    localPos.z *= height

    const worldPos = localPos.add(vec3f(pos, 0))
    const clipPos = cameraBuffer.$.viewMatrix.mul(vec4f(worldPos, 1))
    return {
      localPos,
      worldPos,
      height,
      clipPos,
    }
  })
}

function createFragmentProgram(
  timeBuffer: TgpuBufferUniform<typeof TimeStruct>,
) {
  return tgpu['~unstable'].fragmentFn({
    in: {
      localPos: vec3f,
      worldPos: vec3f,
      height: f32,
    },
    out: vec4f,
  })(({ localPos, height }) => {
    let color = vec3f(
      abs(localPos.x),
      abs(localPos.y),
      abs(localPos.z / height),
    )

    color = fract(color.add(vec3f(timeBuffer.$.elapsed * 0.1)))
    return vec4f(color, 1)
  })
}
