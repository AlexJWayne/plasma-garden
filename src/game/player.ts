import { addEntity, query } from 'bitecs'
import tgpu, { type TgpuBufferUniform } from 'typegpu'
import { builtin, struct, vec2f, vec3f, vec4f } from 'typegpu/data'
import { length, normalize, pow, smoothstep } from 'typegpu/std'

import { cubeVertices, quadVertices } from '../lib/geometry'
import { blending } from '../lib/web-gpu'
import type { World } from '../main'
import { depthFormat, presentationFormat } from '../setup-webgpu'

import { CameraStruct } from './camera'
import {
  Acceleration,
  Drag,
  MaxSpeed,
  Player,
  Position,
  Velocity,
} from './components'

const SIZE = 0.2

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
    .withDepthStencil({
      format: depthFormat,
      depthWriteEnabled: true,
      depthCompare: 'less',
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
      .withDepthStencilAttachment({
        view: world.depthTexture.createView(),
        depthLoadOp: 'load',
        depthStoreOp: 'store',
      })
      .draw(cubeVertices.value.length)
  }

  return render
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

function createVertexProgram(
  cameraBuffer: TgpuBufferUniform<typeof CameraStruct>,
  playerBuffer: TgpuBufferUniform<typeof PlayerStruct>,
) {
  return tgpu['~unstable'].vertexFn({
    in: { idx: builtin.vertexIndex },
    out: { localPos: vec3f, worldPos: vec3f, clipPos: builtin.position },
  })(({ idx }) => {
    const localPos = cubeVertices.$[idx]
    const worldPos = localPos
      .mul(SIZE / 2)
      .add(vec3f(playerBuffer.$.position, SIZE / 2))
    const clipPos = cameraBuffer.$.viewMatrix.mul(vec4f(worldPos, 1))
    return { localPos, worldPos, clipPos }
  })
}

function createFragmentProgram() {
  return tgpu['~unstable'].fragmentFn({
    in: { localPos: vec3f, worldPos: vec3f },
    out: vec4f,
  })(({ localPos, worldPos }) => {
    return vec4f(localPos.div(2).add(0.5), 1)
  })
}
