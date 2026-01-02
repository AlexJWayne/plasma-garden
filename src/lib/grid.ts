import { type World, query } from 'bitecs'
import type { v2i } from 'typegpu/data'
import { vec2i } from 'typegpu/data'

const EXTENTS = {
  min: vec2i(-10, -10),
  max: vec2i(10, 10),
}
const SIZE = EXTENTS.max.sub(EXTENTS.min).add(1)
const TOTAL_POSITIONS = SIZE.x * SIZE.y

export const GridPosition = [] as v2i[]

// TODO: This is terribly slow
export function getRandomEmptyGridPosition(world: World): v2i | null {
  const occupiedEntities = query(world, [GridPosition])

  // Create a Set of occupied positions for O(1) lookup
  const occupiedPositions = new Set<string>()
  for (const eid of occupiedEntities) {
    const pos = GridPosition[eid]
    occupiedPositions.add(`${pos.x},${pos.y}`)
  }

  const emptyPositions = TOTAL_POSITIONS - occupiedPositions.size
  if (emptyPositions === 0) return null

  // Generate random positions until we find an empty one
  const maxAttempts = Math.min(emptyPositions * 10, 500)
  for (let i = 0; i < maxAttempts; i++) {
    const x = Math.floor(Math.random() * SIZE.x) + EXTENTS.min.x
    const y = Math.floor(Math.random() * SIZE.y) + EXTENTS.min.y
    const key = `${x},${y}`

    if (!occupiedPositions.has(key)) return vec2i(x, y)
  }

  return null
}
