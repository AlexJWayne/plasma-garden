import { quadVert } from '../lib-gpu'
import type { World } from '../main'
import { presentationFormat } from '../setup-webgpu'
import { CameraStruct } from './camera'
import { type TgpuBufferUniform, tgpu } from 'typegpu'
import { builtin, vec2f, vec3f, vec4f } from 'typegpu/data'
import { abs, fract, min, smoothstep } from 'typegpu/std'

export function createRenderBackgroundSystem(world: World) {
  const renderPipeline = world.root['~unstable']
    .withVertex(createVertexProgram(world.camera.buffer.as('uniform')), {})
    .withFragment(createFragmentProgram(), { format: presentationFormat })
    .createPipeline()

  function render(world: World) {
    renderPipeline
      .withColorAttachment({
        view: world.ctx.getCurrentTexture().createView(),
        loadOp: 'clear',
        storeOp: 'store',
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
    const uv = quadVert(idx)
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
    return vec4f(vec3f(v * 0.2), 1)
  })
}
