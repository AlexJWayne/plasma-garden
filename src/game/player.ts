import { addEntity, query } from 'bitecs'
import { f32, vec2f } from 'typegpu/data'
import { length, normalize } from 'typegpu/std'

import type { World } from '../main'

import {
  Acceleration,
  Drag,
  MaxSpeed,
  Player,
  Position,
  Velocity,
} from './components'

export const SIZE = 0.05
export const PLAYER_HEIGHT = f32(3)

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
