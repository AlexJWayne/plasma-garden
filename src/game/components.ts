import type { EntityId } from 'bitecs'
import { type v2f } from 'typegpu/data'
import { clamp } from 'typegpu/std'

import type { World } from '../main'

// General
export const Position = [] as v2f[]
export const Velocity = [] as v2f[]
export const Acceleration = [] as v2f[]
export const Drag = [] as number[]
export const MaxSpeed = [] as number[]

export const Lifetime = [] as {
  bornAt: number
  duration: number
}[]

export function getLifetimeCompletion(world: World, eid: EntityId) {
  const { bornAt, duration } = Lifetime[eid]
  const completion = (world.time.elapsed - bornAt) / duration
  return clamp(completion, 0, 1)
}

// Game Objects
export const Player = {}
export const Camera = {}
export const Bullet = [] as { damage: number }[]
