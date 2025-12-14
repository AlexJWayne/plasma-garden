import { addEntity, query } from 'bitecs'
import tgpu, { type TgpuBufferUniform } from 'typegpu'
import {
  arrayOf,
  builtin,
  struct,
  type v2f,
  vec2f,
  vec3f,
  vec4f,
} from 'typegpu/data'
import { length, smoothstep } from 'typegpu/std'

import { quadVertices } from '../lib/geometry'
import { blending } from '../lib/web-gpu'
import type { World } from '../main'
import { depthFormat, presentationFormat } from '../setup-webgpu'

import type { CameraStruct } from './camera'
import { Bullet, Position, Velocity } from './components'

const MAX_BULLETS = 10000

const BulletStruct = struct({
  pos: vec2f,
})

export function createBullet(
  world: World,
  position: v2f,
  velocity: v2f,
  damage: number,
): number {
  const eid = addEntity(world, Bullet, Position, Velocity)
  Position[eid] = position
  Velocity[eid] = velocity
  Bullet[eid] = { damage }
  return eid
}

export function createRenderBulletSystem(world: World) {
  const bulletsBuffer = world.root
    .createBuffer(arrayOf(BulletStruct, MAX_BULLETS))
    .$usage('vertex', 'storage')

  const bulletsLayout = tgpu.vertexLayout(
    (n: number) => arrayOf(BulletStruct, n),
    'instance',
  )

  const renderPipeline = world.root['~unstable']
    .withVertex(createVertexProgram(world.camera.buffer.as('uniform')), {
      pos: bulletsLayout.attrib.pos,
    })
    .withFragment(createFragmentProgram(), {
      format: presentationFormat,
      blend: blending.normal,
    })
    .withDepthStencil({
      format: depthFormat,
      depthWriteEnabled: true,
      depthCompare: 'less',
    })
    .createPipeline()
    .with(bulletsLayout, bulletsBuffer)

  function render(world: World) {
    const bullets = query(world, [Bullet, Position])
    if (bullets.length === 0) return

    bulletsBuffer.writePartial(
      [...bullets].map((eid, i) => ({
        idx: i,
        value: { pos: Position[eid] },
      })),
    )

    renderPipeline
      .withColorAttachment({
        view: world.ctx.getCurrentTexture().createView(),
        loadOp: 'load',
        storeOp: 'store',
      })
      .withDepthStencilAttachment({
        view: world.depthTexture.createView(),
        depthLoadOp: 'load',
        depthStoreOp: 'store',
      })
      .draw(6, bullets.length)
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
    },
    out: {
      pos: builtin.position,
      uv: vec2f,
    },
  })(({ idx, pos }) => {
    const uv = quadVertices.$[idx]
    const outPos = vec3f(uv.mul(0.01).add(pos), 0.05)
    return {
      pos: cameraBuffer.$.viewMatrix.mul(vec4f(outPos, 1)),
      uv,
    }
  })
}

function createFragmentProgram() {
  return tgpu['~unstable'].fragmentFn({
    in: { uv: vec2f },
    out: vec4f,
  })(({ uv }) => {
    let v = smoothstep(1, 0.6, length(uv))
    return vec4f(vec3f(1, 0, 0).mul(v), v)
  })
}
