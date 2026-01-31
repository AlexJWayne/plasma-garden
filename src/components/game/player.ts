import { opSmoothUnion, sdSphere } from '@typegpu/sdf'
import { addEntity, query } from 'bitecs'
import { type Infer, arrayOf, f32, struct, vec2f, vec3f } from 'typegpu/data'
import { length, normalize, sin } from 'typegpu/std'

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
      const halfSize = SIZE * 1.7
      return AABB({
        min: player.position.sub(vec3f(halfSize)),
        max: player.position.add(vec3f(halfSize)),
      })
    },

    sdSurface: (p, player, elapsed) => {
      'use gpu'
      const position = p.sub(player.position)
      const radius = SIZE / 2

      let d = sdSphere(position, radius)

      for (let i = 0; i < 12; i++) {
        const idx = f32(i)

        const time = elapsed * 0.6
        const offset = vec3f(
          sin((time + 10) * (0.5 + idx * 0.08)),
          sin((time + 20) * (0.6 + idx * 0.1)),
          sin((time + 30) * (0.7 + idx * 0.12)),
        ).mul(0.05)

        d = opSmoothUnion(d, sdSphere(position.add(offset), radius * 0.8), 0.04)
      }
      return d
    },

    calcSurfaceColors: (hitPos, player, _elapsed) => {
      'use gpu'
      const closeness = 1 - length(hitPos.sub(player.position)) / 0.14
      return SurfaceColors({
        diffuse: vec3f(0),
        specular: vec3f(0),
        emissive: vec3f(closeness),
        shininess: f32(0),
      })
    },

    maxSteps: 60,
    maxDistance: 10,
    epsilon: 0.001,
    epsilonNormal: 0.01,
    debug: false,
  })
}
