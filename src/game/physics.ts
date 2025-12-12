import { query } from 'bitecs'
import { length, normalize } from 'typegpu/std'

import type { World } from '../main'

import { Drag, MaxSpeed, Position, Velocity } from './components'

export function moveSystem(world: World) {
  for (const eid of query(world, [Position, Velocity])) {
    Position[eid] = Position[eid].add(Velocity[eid].mul(world.delta))
  }
}

export function bounceOffBoundariesSystem(world: World) {
  for (const eid of query(world, [Position, Velocity])) {
    const pos = Position[eid]
    const vel = Velocity[eid]

    if (pos.x > 1) {
      vel.x = -Math.abs(vel.x)
      pos.x = 1
    }
    if (pos.y > 1) {
      vel.y = -Math.abs(vel.y)
      pos.y = 1
    }
    if (pos.x < -1) {
      vel.x = Math.abs(vel.x)
      pos.x = -1
    }
    if (pos.y < -1) {
      vel.y = Math.abs(vel.y)
      pos.y = -1
    }
  }
}

export function applyDragSystem(world: World) {
  for (const eid of query(world, [Position, Velocity, Drag])) {
    Velocity[eid] = Velocity[eid].mul(1 - Drag[eid] * world.delta)
  }
}

export function applyMaxSpeedSystem(world: World) {
  for (const eid of query(world, [Velocity, MaxSpeed])) {
    const vel = Velocity[eid]
    const maxSpeed = MaxSpeed[eid]
    if (length(vel) > maxSpeed) Velocity[eid] = normalize(vel).mul(maxSpeed)
  }
}
