import { query } from 'bitecs'
import { length, normalize } from 'typegpu/std'

import type { World } from '../main'

import { Acceleration, Drag, MaxSpeed, Position, Velocity } from './components'

export function physicsSystem(world: World) {
  bounceOffBoundariesSystem(world)
  accelerateSystem(world)
  moveByVelocitySystem(world)
  applyMaxSpeedSystem(world)
  applyDragSystem(world)
}

const BOUNDARY = 10
function bounceOffBoundariesSystem(world: World) {
  for (const eid of query(world, [Position, Velocity])) {
    const pos = Position[eid]
    const vel = Velocity[eid]

    if (pos.x > BOUNDARY) {
      vel.x = -Math.abs(vel.x)
      pos.x = BOUNDARY
    }
    if (pos.y > BOUNDARY) {
      vel.y = -Math.abs(vel.y)
      pos.y = BOUNDARY
    }
    if (pos.x < -BOUNDARY) {
      vel.x = Math.abs(vel.x)
      pos.x = -BOUNDARY
    }
    if (pos.y < -BOUNDARY) {
      vel.y = Math.abs(vel.y)
      pos.y = -BOUNDARY
    }
  }
}

function accelerateSystem(world: World) {
  for (const eid of query(world, [Velocity, Acceleration])) {
    Velocity[eid] = Velocity[eid].add(Acceleration[eid].mul(world.delta))
  }
}

function moveByVelocitySystem(world: World) {
  for (const eid of query(world, [Position, Velocity])) {
    Position[eid] = Position[eid].add(Velocity[eid].mul(world.delta))
  }
}

function applyMaxSpeedSystem(world: World) {
  for (const eid of query(world, [Velocity, MaxSpeed])) {
    const vel = Velocity[eid]
    const maxSpeed = MaxSpeed[eid]
    if (length(vel) > maxSpeed) Velocity[eid] = normalize(vel).mul(maxSpeed)
  }
}

function applyDragSystem(world: World) {
  for (const eid of query(world, [Position, Velocity, Drag])) {
    Velocity[eid] = Velocity[eid].mul(1 - Drag[eid] * world.delta)
  }
}
