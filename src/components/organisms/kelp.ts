import { addEntity, query, set } from 'bitecs'
import tgpu, { type TgpuBufferUniform } from 'typegpu'
import { type Infer, builtin, f32, struct, vec3f, vec4f } from 'typegpu/data'

import { createInstanceBuffer } from '../../lib/buffers'
import { cubeVertex, cubeVertices } from '../../lib/geometry'
import { createPipelinePerformanceCallback } from '../../lib/pipeline-perf'
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
  pos: vec3f,
  height: f32,
  growth: f32,
})
type KelpStruct = Infer<typeof KelpStruct>

const SPAWN_RATE = 0.05

export function createKelp(world: World) {
  const gridPosition = getRandomEmptyGridPosition(world)
  if (!gridPosition) return

  const eid = addEntity(
    world,
    set(GridPosition, gridPosition),
    set(Lifetime, Math.random() * 10 + 5),
    Kelp,
  )
  Kelp[eid] = { height: Math.random() * 3 + 3 }
}

export function spawnKelpSystem(world: World) {
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
    .withFragment(createFragmentProgram(), {
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
          pos: vec3f(GridPosition[eid], 0),
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
      pos: vec3f,
      height: f32,
      growth: f32,
    },
    out: {
      localPos: vec3f,
      worldPos: vec3f,
      clipPos: builtin.position,
    },
  })(({ idx, pos, height, growth }) => {
    const localPos = cubeVertex(idx, 0.9 * (1 - growth), height * growth)
    const worldPos = localPos.add(vec3f(pos))
    const clipPos = worldToClipSpace(cameraBuffer.$, worldPos)
    return {
      localPos,
      worldPos,
      clipPos,
    }
  })
}

function createFragmentProgram() {
  return tgpu['~unstable'].fragmentFn({
    in: {
      localPos: vec3f,
      worldPos: vec3f,
      clipPos: builtin.position,
    },
    out: {
      color: vec4f,
    },
  })(() => {
    const color = vec4f(0.0, 0.5, 0.0, 1.0)
    return { color }
  })
}
