import { addComponent, addEntity, query } from 'bitecs'
import tgpu, { type TgpuBufferUniform } from 'typegpu'
import { builtin, struct, vec2f, vec3f, vec4f } from 'typegpu/data'
import { length, normalize, pow, smoothstep } from 'typegpu/std'

import { blending, quadVert } from '../lib-gpu'
import type { World } from '../main'
import { presentationFormat } from '../setup-webgpu'

import { CameraStruct } from './camera'
import {
  Acceleration,
  Drag,
  MaxSpeed,
  Player,
  Position,
  Velocity,
} from './components'

const PlayerStruct = struct({
  position: vec2f,
})

export function createPlayerEntity(world: World) {
  const eid = addEntity(
    world,
    Player,
    Position,
    Velocity,
    Acceleration,
    MaxSpeed,
    Drag,
  )

  Position[eid] = vec2f(0)
  Velocity[eid] = vec2f(0)
  Acceleration[eid] = vec2f(0)
  MaxSpeed[eid] = 2
  Drag[eid] = 1

  return eid
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
      position: Position[player],
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
  const force = 200

  let direction = vec2f(0)
  if (world.input.isDirectionDown('right')) direction.x += 1
  if (world.input.isDirectionDown('left')) direction.x -= 1
  if (world.input.isDirectionDown('up')) direction.y += 1
  if (world.input.isDirectionDown('down')) direction.y -= 1

  direction = length(direction) > 0 ? normalize(direction) : direction

  const player = query(world, [Player, Velocity, Acceleration])[0]
  Acceleration[player] = direction.mul(force * world.delta)
}
