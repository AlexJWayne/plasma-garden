import type { World } from '../main'
import { Drag, Position, Velocity } from './components'
import { query } from 'bitecs'

export function moveSystem(world: World) {
  for (const eid of query(world, [Position, Velocity])) {
    Position.add(
      eid,
      Velocity.x[eid] * world.delta,
      Velocity.y[eid] * world.delta,
    )
  }
}

export function bounceOffBoundariesSystem(world: World) {
  for (const eid of query(world, [Position, Velocity])) {
    const px = Position.x[eid]
    const py = Position.y[eid]
    const vx = Velocity.x[eid]
    const vy = Velocity.y[eid]

    if (px > 1) {
      Velocity.x[eid] = -Math.abs(vx)
      Position.x[eid] = 1
    }
    if (py > 1) {
      Velocity.y[eid] = -Math.abs(vy)
      Position.y[eid] = 1
    }
    if (px < -1) {
      Velocity.x[eid] = Math.abs(vx)
      Position.x[eid] = -1
    }
    if (py < -1) {
      Velocity.y[eid] = Math.abs(vy)
      Position.y[eid] = -1
    }
  }
}

export function applyDragSystem(world: World) {
  for (const eid of query(world, [Position, Velocity, Drag])) {
    const vx = Velocity.x[eid]
    const vy = Velocity.y[eid]
    const drag = Drag[eid]

    Velocity.set(
      eid,
      vx - vx * drag * world.delta,
      vy - vy * drag * world.delta,
    )
  }
}
