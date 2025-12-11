import type { World } from '../main'
import { Drag, Position, Velocity } from './components'
import { query } from 'bitecs'
import { vec2f } from 'typegpu/data'

export function moveSystem(world: World) {
  for (const eid of query(world, [Position, Velocity])) {
    Position[eid] = Position[eid].add(
      vec2f(
        Velocity.x[eid] * world.delta, //
        Velocity.y[eid] * world.delta,
      ),
    )
  }
}

export function bounceOffBoundariesSystem(world: World) {
  for (const eid of query(world, [Position, Velocity])) {
    const pos = Position[eid]

    const vx = Velocity.x[eid]
    const vy = Velocity.y[eid]

    if (pos.x > 1) {
      Velocity.x[eid] = -Math.abs(vx)
      pos.x = 1
    }
    if (pos.y > 1) {
      Velocity.y[eid] = -Math.abs(vy)
      pos.y = 1
    }
    if (pos.x < -1) {
      Velocity.x[eid] = Math.abs(vx)
      pos.x = -1
    }
    if (pos.y < -1) {
      Velocity.y[eid] = Math.abs(vy)
      pos.y = -1
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
