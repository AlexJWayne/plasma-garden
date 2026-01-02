import {
  type World,
  addEntity,
  observe,
  onRemove,
  onSet,
  query,
  removeEntity,
} from 'bitecs'
import type { v2i } from 'typegpu/data'
import { vec2i } from 'typegpu/data'

const EXTENTS = {
  min: vec2i(-10, -10),
  max: vec2i(10, 10),
}

const GridPositionStore = [] as v2i[]
export const GridPosition = GridPositionStore as Readonly<
  typeof GridPositionStore
>

const EmptyGridPosition = [] as v2i[]

export function observeGrid(world: World): void {
  observe(world, onSet(GridPosition), (eid, params: v2i) => {
    GridPositionStore[eid] = params
    takeEmptyGridPosition(world, params)
  })
  observe(world, onRemove(GridPosition), (eid) => {
    releaseGridPosition(world, GridPosition[eid])
  })
}

export function createGridPositions(world: World): void {
  for (let y = EXTENTS.min.y; y <= EXTENTS.max.y; y++) {
    for (let x = EXTENTS.min.x; x <= EXTENTS.max.x; x++) {
      const eid = addEntity(world, EmptyGridPosition)
      EmptyGridPosition[eid] = vec2i(x, y)
    }
  }
}

export function takeEmptyGridPosition(world: World, pos: v2i): v2i | null {
  const emptyPositions = query(world, [EmptyGridPosition])
  if (emptyPositions.length === 0) return null

  for (const eid of emptyPositions) {
    const position = EmptyGridPosition[eid]
    if (position.x === pos.x && position.y === pos.y) {
      removeEntity(world, eid)
      return position
    }
  }

  throw new Error(`Position ${pos} is not empty`)
}

export function releaseGridPosition(world: World, pos: v2i): void {
  const eid = addEntity(world, EmptyGridPosition)
  EmptyGridPosition[eid] = pos
}

export function getRandomEmptyGridPosition(world: World): v2i | null {
  const emptyPositions = query(world, [EmptyGridPosition])
  if (emptyPositions.length === 0) return null

  const index = Math.floor(Math.random() * emptyPositions.length)
  const eid = emptyPositions[index]
  return EmptyGridPosition[eid]
}
