import { blending, quadVert } from '../lib-gpu'
import type { World } from '../main'
import { presentationFormat } from '../setup-webgpu'
import { CameraStruct } from './camera'
import { Position, Velocity } from './physics'
import { addComponent, addEntity, query } from 'bitecs'
import tgpu, { type TgpuBufferUniform } from 'typegpu'
import { builtin, struct, vec2f, vec3f, vec4f } from 'typegpu/data'
import { length, pow, smoothstep } from 'typegpu/std'

export const Player = {}

const PlayerStruct = struct({
  position: vec2f,
})

export function createPlayerEntity(world: World) {
  const eid = addEntity(world)
  addComponent(world, eid, Player)

  addComponent(world, eid, Position)
  Position.set(eid, 0, 0)

  addComponent(world, eid, Velocity)
  Velocity.set(
    eid,
    (Math.random() * 2 - 1) * 0.01,
    (Math.random() * 2 - 1) * 0.01,
  )
}

export function createRenderPlayerSystem(world: World) {
  const playerBuffer = world.root.createBuffer(PlayerStruct).$usage('uniform')

  const renderPipeline = world.root['~unstable']
    .withVertex(
      createVertexProgram(
        world.camera.buffer.as('uniform'),
        playerBuffer.as('uniform'),
      ),
      {},
    )
    .withFragment(createFragmentProgram(), {
      format: presentationFormat,
      blend: blending.normal,
    })
    .createPipeline()

  function render(world: World) {
    const player = query(world, [Player, Position])[0]

    playerBuffer.write({
      position: vec2f(Position.x[player], Position.y[player]),
    })

    renderPipeline
      .withColorAttachment({
        view: world.ctx.getCurrentTexture().createView(),
        loadOp: 'load',
        storeOp: 'store',
      })
      .draw(6)
  }

  return render
}

function createVertexProgram(
  cameraBuffer: TgpuBufferUniform<typeof CameraStruct>,
  playerBuffer: TgpuBufferUniform<typeof PlayerStruct>,
) {
  return tgpu['~unstable'].vertexFn({
    in: { idx: builtin.vertexIndex },
    out: {
      pos: builtin.position,
      uv: vec2f,
    },
  })(({ idx }) => {
    const uv = quadVert(idx)
    const pos = vec3f(uv.mul(0.1).add(playerBuffer.$.position), 0)
    return {
      pos: cameraBuffer.$.viewMatrix.mul(vec4f(pos, 1)),
      uv,
    }
  })
}

function createFragmentProgram() {
  return tgpu['~unstable'].fragmentFn({
    in: { uv: vec2f },
    out: vec4f,
  })(({ uv }) => {
    let v = smoothstep(1, 0.1, length(uv))
    v = pow(v, 5)
    return vec4f(vec3f(0.6, 0.6, 1).mul(v), v)
  })
}
