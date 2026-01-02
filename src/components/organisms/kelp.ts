import { addEntity, query, removeEntity, set } from 'bitecs'
import { type Infer, f32, struct, vec3f } from 'typegpu/data'

import type { World } from '../../main'
import { GridPosition, getRandomEmptyGridPosition } from '../general/grid'
import { Lifetime } from '../general/lifetime'

type Kelp = {
  height: number
}
const Kelp = [] as Kelp[]

const KelpStruct = struct({
  pos: vec3f,
  height: f32,
})
type KelpStruct = Infer<typeof KelpStruct>

const SPAWN_RATE = 0.05

export function createKelp(world: World) {
  const gridPosition = getRandomEmptyGridPosition(world)
  if (!gridPosition) return

  const eid = addEntity(
    world,
    set(GridPosition, gridPosition),
    set(Lifetime, Math.random() * 10 + 5),
    Kelp,
  )
  Kelp[eid] = { height: Math.random() * 4 + 2 }
}

export function spawnKelpSystem(world: World) {
  if (Math.random() < SPAWN_RATE) createKelp(world)
}

export function expireKelpSystem(world: World) {
  const mushrooms = query(world, [Kelp, Lifetime, GridPosition])
  for (const eid of mushrooms) {
    const { bornAt, duration } = Lifetime[eid]
    if (world.time.elapsed > bornAt + duration) {
      removeEntity(world, eid)
    }
  }
}
