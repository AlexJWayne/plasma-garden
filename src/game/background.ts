import { type TgpuBufferUniform, tgpu } from 'typegpu'
import { builtin, vec2f, vec3f, vec4f } from 'typegpu/data'
import { abs, discard, fract, min, smoothstep } from 'typegpu/std'

import { quadVertices } from '../lib/geometry'
import type { World } from '../main'
import { depthFormat, presentationFormat, sampleCount } from '../setup-webgpu'

import { CameraStruct } from './camera'

export function createRenderBackgroundSystem(world: World) {
  const renderPipeline = world.root['~unstable']
    .withVertex(createVertexProgram(world.camera.buffer.as('uniform')), {})
    .withFragment(createFragmentProgram(), { format: presentationFormat })
    .withDepthStencil({
      format: depthFormat,
      depthWriteEnabled: true,
      depthCompare: 'less',
    })
    .withMultisample({ count: sampleCount })
    .createPipeline()

  function render(world: World) {
    renderPipeline
      .withColorAttachment({
        view: world.colorTexture.createView(),
        resolveTarget: world.ctx.getCurrentTexture().createView(),
        loadOp: 'clear',
        storeOp: 'store',
      })
      .withDepthStencilAttachment({
        view: world.depthTexture.createView(),
        depthLoadOp: 'clear',
        depthStoreOp: 'store',
        depthClearValue: 1.0,
      })
      .draw(6)
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
      pos: cameraBuffer.$.viewMatrix.mul(vec4f(uv, 0, 1)),
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
