import { query } from 'bitecs'
import type { v2f } from 'typegpu/data'
import { length, normalize } from 'typegpu/std'

import type { World } from '../../main'

export const Position = [] as v2f[]
export const Velocity = [] as v2f[]
export const Acceleration = [] as v2f[]
export const Drag = [] as number[]
export const MaxSpeed = [] as number[]

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
    Velocity[eid] = Velocity[eid].add(Acceleration[eid].mul(world.time.delta))
  }
}

function moveByVelocitySystem(world: World) {
  for (const eid of query(world, [Position, Velocity])) {
    Position[eid] = Position[eid].add(Velocity[eid].mul(world.time.delta))
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
    Velocity[eid] = Velocity[eid].mul(1 - Drag[eid] * world.time.delta)
  }
}
