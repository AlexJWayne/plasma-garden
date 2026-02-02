import { perlin3d } from '@typegpu/noise'
import { opSmoothUnion, sdSphere } from '@typegpu/sdf'
import { addEntity, query } from 'bitecs'
import {
  type Infer,
  f32,
  struct,
  type v3f,
  vec2f,
  vec3f,
} from 'typegpu/data'
import { clamp, length, max, min, normalize, sin } from 'typegpu/std'

import { easeOutCubic } from '../../lib/ease'
import { hsl2rgb } from '../../lib/hsl'
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
export const PLAYER_HEIGHT = f32(2)
const WIGGLER_COUNT = 20

const PlayerStruct = struct({
  position: vec3f,
  velocity: vec3f,
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
  return createSDFInstancesRenderer(world, 'Player')
    .withBuffer(PlayerStruct, 1, (buffer) => {
      const players = query(world, [Player, Position])
      if (players.length === 0) return 0

      const player = players[0]

      buffer.write([
        {
          position: vec3f(Position[player], PLAYER_HEIGHT),
          velocity: vec3f(Velocity[player], 0),
        },
      ])

      return players.length
    })
    .withRaymarching({
      calcAABB: (player, elapsed) => {
        'use gpu'

        let minPt = vec3f(-0.02)
        let maxPt = vec3f(0.02)
        for (let i = 0; i < WIGGLER_COUNT; i++) {
          const offset = getWiggleOffset(player, elapsed, i).mul(-1)
          minPt = min(minPt, offset)
          maxPt = max(maxPt, offset)
        }

        return AABB({
          min: minPt.add(player.position).sub(0.03),
          max: maxPt.add(player.position).add(0.03),
        })
      },

      sdSurface: (p, player, elapsed) => {
        'use gpu'
        const position = p.sub(player.position)
        const radius = SIZE / 2

        let d = sdSphere(position, radius * 1.3)

        for (let i = 0; i < WIGGLER_COUNT; i++) {
          const offset = getWiggleOffset(player, elapsed, i)
          d = opSmoothUnion(d, sdSphere(position.add(offset), radius * 0.3), 0.04)
        }
        return d
      },

      calcSurfaceColors: (hitPos, player, elapsed) => {
        'use gpu'

        const noise =
          perlin3d.sample(
            hitPos
              .sub(player.position)
              .add(elapsed * 0.0)
              .mul(70),
          ) *
            0.1 +
          0.02

        const value = clamp(
          easeOutCubic(1 - length(hitPos.sub(player.position)) / 0.2) + noise,
          0.8,
          1,
        )

        return SurfaceColors({
          diffuse: vec3f(0),
          specular: vec3f(0),
          emissive: vec3f(hsl2rgb(vec3f(0.6, 0.8, value))),
          shininess: f32(0),
        })
      },

      maxSteps: 60,
      maxDistance: 10,
      epsilon: 0.001,
      epsilonNormal: 1,
      debug: false,
    })
    .createRenderer()
}

function getWiggleOffset(
  player: PlayerStruct,
  elapsed: number,
  idxInt: number,
): v3f {
  'use gpu'
  const idx = f32(idxInt)
  const time = elapsed * 0.6

  const wiggleOffset = vec3f(
    sin((time + 10) * (0.5 + idx * 0.08)),
    sin((time + 20) * (0.6 + idx * 0.1)),
    sin((time + 30) * (0.7 + idx * 0.12)),
  ).mul(0.04)

  const velocityOffset = player.velocity.mul((0.1 * idx) / f32(WIGGLER_COUNT))

  return wiggleOffset.add(velocityOffset)
}
