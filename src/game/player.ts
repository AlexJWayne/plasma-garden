import { blending, quadVert } from '../lib-gpu'
import type { World } from '../main'
import { presentationFormat } from '../setup-webgpu'
import { CameraStruct } from './camera'
import { Drag, Player, Position, Velocity } from './components'
import { addComponent, addEntity, query } from 'bitecs'
import tgpu, { type TgpuBufferUniform } from 'typegpu'
import { builtin, struct, vec2f, vec3f, vec4f } from 'typegpu/data'
import { length, normalize, pow, smoothstep } from 'typegpu/std'

const PlayerStruct = struct({
  position: vec2f,
})

export function createPlayerEntity(world: World) {
  const eid = addEntity(world)
  addComponent(world, eid, Player)

  addComponent(world, eid, Position)
  Position.set(eid, 0, 0)

  addComponent(world, eid, Velocity)
  Velocity.set(eid, 0, 0)
  Velocity.maxSpeed[eid] = 4

  addComponent(world, eid, Drag)
  Drag[eid] = 4
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

export function applyMovementInputToPlayer(world: World) {
  const force = 3

  let direction = vec2f(0)

  if (world.input.isDown('arrowright')) direction.x += 1
  if (world.input.isDown('arrowleft')) direction.x -= 1
  if (world.input.isDown('arrowup')) direction.y += 1
  if (world.input.isDown('arrowdown')) direction.y -= 1

  direction = length(direction) > 0 ? normalize(direction) : direction

  const player = query(world, [Player, Velocity])[0]
  const maxSpeed = Velocity.maxSpeed[player]
  const accel = direction.mul(force * world.delta)
  let vel = Velocity.asVec2f(player).add(accel)
  if (length(vel) > maxSpeed) vel = normalize(vel).mul(maxSpeed)

  Velocity.set(player, vel.x, vel.y)
}
