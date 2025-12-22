import { type TgpuBufferUniform, tgpu } from 'typegpu'
import { builtin, vec2f, vec3f, vec4f } from 'typegpu/data'
import { abs, discard, fract, min, smoothstep } from 'typegpu/std'

import { quadVertices } from '../lib/geometry'
import {
  createColorAttachment,
  createDepthAttachment,
  depthStencil,
} from '../lib/web-gpu'
import type { World } from '../main'
import { presentationFormat, sampleCount } from '../setup-webgpu'

import { CameraStruct } from './camera'

export function createRenderBackgroundSystem(world: World) {
  const renderPipeline = world.root['~unstable']
    .withVertex(createVertexProgram(world.camera.buffer.as('uniform')), {})
    .withFragment(createFragmentProgram(), { format: presentationFormat })
    .withDepthStencil(depthStencil)
    .withMultisample({ count: sampleCount })
    .createPipeline()

  function render(world: World) {
    renderPipeline
      .withColorAttachment({
        ...createColorAttachment(world),
        loadOp: 'clear',
      })
      .withDepthStencilAttachment({
        ...createDepthAttachment(world),
        depthLoadOp: 'clear',
        depthClearValue: 1,
      })
      .draw(quadVertices.$.length)
  }

  return render
}

function createVertexProgram(
  cameraBuffer: TgpuBufferUniform<typeof CameraStruct>,
) {
  return tgpu['~unstable'].vertexFn({
    in: { idx: builtin.vertexIndex },
    out: {
      pos: builtin.position,
      uv: vec2f,
    },
  })(({ idx }) => {
    const uv = quadVertices.$[idx]
    return {
      pos: cameraBuffer.$.viewMatrix.mul(vec4f(uv.mul(10), 0, 1)),
      uv,
    }
  })
}

function createFragmentProgram() {
  return tgpu['~unstable'].fragmentFn({
    in: { uv: vec2f },
    out: vec4f,
  })(({ uv }) => {
    const grid = fract(uv.mul(10))
    const line = abs(grid.sub(0.5))
    const d = min(line.x, line.y)
    const v = smoothstep(0.1, 0, d)
    if (v < 0.75) discard()

    return vec4f(vec3f(0.2), 1)
  })
}
