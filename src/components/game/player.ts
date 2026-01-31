import { sdSphere } from '@typegpu/sdf'
import { addEntity, query } from 'bitecs'
import { type Infer, arrayOf, f32, struct, vec2f, vec3f } from 'typegpu/data'
import { length, normalize } from 'typegpu/std'

import { SurfaceColors } from '../../lib/lighting'
import {
  AABB,
  createSDFInstancesRenderer,
} from '../../lib/sdf-instance-pipeline'
import type { World } from '../../main'
import {
  Acceleration,
  Drag,
  MaxSpeed,
  Position,
  Velocity,
} from '../general/physics'

export const Player = {}

export const SIZE = 0.05
export const PLAYER_HEIGHT = f32(3)

const PlayerStruct = struct({
  position: vec3f,
})
type PlayerStruct = Infer<typeof PlayerStruct>

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
  Drag[eid] = 0.5

  return eid
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
  Acceleration[player] = direction.mul(force * world.time.delta)
}

export function createRenderPlayerSystem(world: World) {
  const playerBuffer = world.root
    .createBuffer(arrayOf(PlayerStruct, 1))
    .$usage('storage')

  return createSDFInstancesRenderer({
    name: 'Player',
    world,

    cameraBuffer: world.camera.buffer.as('uniform'),
    instanceBuffer: playerBuffer.as('readonly'),
    timeBuffer: world.time.buffer.as('uniform'),

    writeBuffers: () => {
      const players = query(world, [Player, Position])
      if (players.length === 0) return 0

      const player = players[0]

      playerBuffer.write([{ position: vec3f(Position[player], PLAYER_HEIGHT) }])

      return players.length
    },

    calcAABB: (player) => {
      'use gpu'
      const halfSize = SIZE / 2
      return AABB({
        min: player.position.sub(vec3f(halfSize)),
        max: player.position.add(vec3f(halfSize)),
      })
    },

    sdSurface: (p, player) => {
      'use gpu'
      return sdSphere(p.sub(player.position), SIZE / 2)
    },

    calcSurfaceColors: (_hitPos, _player, _elapsed) => {
      'use gpu'
      return SurfaceColors({
        diffuse: vec3f(0.9, 0.9, 0.9),
        specular: vec3f(0.5),
        emissive: vec3f(1),
        shininess: f32(64),
      })
    },

    maxSteps: 60,
    maxDistance: 20,
    epsilon: 0.0001,
    epsilonNormal: 0.01,
    debug: false,
  })
}
