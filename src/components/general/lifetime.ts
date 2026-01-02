import { type EntityId, observe, onSet } from 'bitecs'
import { clamp } from 'typegpu/std'

import type { World } from '../../main'

export type Lifetime = {
  bornAt: number
  duration: number
}
export const LifetimeStore = [] as Lifetime[]
export const Lifetime = LifetimeStore as Readonly<typeof LifetimeStore>

export function observeLifetimes(world: World) {
  observe(world, onSet(Lifetime), (eid, duration: number) => {
    LifetimeStore[eid] = { bornAt: world.time.elapsed, duration }
  })
}

export function getLifetimeCompletion(world: World, eid: EntityId) {
  const { bornAt, duration } = Lifetime[eid]
  const completion = (world.time.elapsed - bornAt) / duration
  return clamp(completion, 0, 1)
}
