import type { World } from '../main'
import { query } from 'bitecs'

const createVec2Component = () => {
  const Component = {
    x: new Float32Array(10000),
    y: new Float32Array(10000),

    set(eid: number, x: number, y: number) {
      Component.x[eid] = x
      Component.y[eid] = y
    },

    add(eid: number, x: number, y: number) {
      Component.x[eid] += x
      Component.y[eid] += y
    },
  }

  return Component
}

export const Position = createVec2Component()
export const Velocity = createVec2Component()

export function moveSystem(world: World) {
  for (const eid of query(world, [Position, Velocity])) {
    Position.add(eid, Velocity.x[eid], Velocity.y[eid])
  }
}

export function bounceOffBoundariesSystem(world: World) {
  for (const eid of query(world, [Position, Velocity])) {
    const px = Position.x[eid]
    const py = Position.y[eid]
    const vx = Velocity.x[eid]
    const vy = Velocity.y[eid]

    if (px > 1) Velocity.x[eid] = -Math.abs(vx)
    if (py > 1) Velocity.y[eid] = -Math.abs(vy)
    if (px < -1) Velocity.x[eid] = Math.abs(vx)
    if (py < -1) Velocity.y[eid] = Math.abs(vy)
  }
}
