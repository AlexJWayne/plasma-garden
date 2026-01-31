import { opSmoothUnion, sdSphere } from '@typegpu/sdf'
import { addEntity, query } from 'bitecs'
import {
  type Infer,
  type Vec3f,
  arrayOf,
  f32,
  struct,
  type v3f,
  vec2f,
  vec3f,
} from 'typegpu/data'
import { length, max, min, normalize, sin } from 'typegpu/std'

import { easeOutCubic, easeOutSine } from '../../lib/ease'
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

const WIGGLER_COUNT = 12

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

    calcAABB: (_player, elapsed) => {
      'use gpu'

      let minPt = vec3f(1000)
      let maxPt = vec3f(-1000)
      for (let i = 0; i < WIGGLER_COUNT; i++) {
        const offset = getWiggleOffset(elapsed, i)
        minPt = min(minPt, offset)
        maxPt = max(maxPt, offset)
      }

      return AABB({
        min: minPt.add(_player.position).sub(0.05),
        max: maxPt.add(_player.position).add(0.05),
      })
    },

    sdSurface: (p, player, elapsed) => {
      'use gpu'
      const position = p.sub(player.position)
      const radius = SIZE / 2

      let d = sdSphere(position, radius * 1.3)

      for (let i = 0; i < WIGGLER_COUNT; i++) {
        const offset = getWiggleOffset(elapsed, i)
        d = opSmoothUnion(d, sdSphere(position.add(offset), radius * 0.7), 0.04)
      }
      return d
    },

    calcSurfaceColors: (hitPos, player, _elapsed) => {
      'use gpu'
      const closeness = easeOutCubic(
        1 - length(hitPos.sub(player.position)) / 0.14,
      )
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
    epsilonNormal: 1,
    debug: false,
  })
}

function getWiggleOffset(elapsed: number, idxInt: number): v3f {
  'use gpu'
  const idx = f32(idxInt)
  const time = elapsed * 0.6
  return vec3f(
    sin((time + 10) * (0.5 + idx * 0.08)),
    sin((time + 20) * (0.6 + idx * 0.1)),
    sin((time + 30) * (0.7 + idx * 0.12)),
  ).mul(0.055)
}
