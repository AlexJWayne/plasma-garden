import { setKeyStateSystem } from '../input'
import { createGridPositions, observeGrid } from '../lib/grid'
import type { World } from '../main'
import { listenForResize } from '../setup-webgpu'

import { createRenderBackgroundSystem } from './background'
import { positionCameraSystem } from './camera'
import { observeLifetimes } from './lifetime'
import {
  createRenderMushroomSystem,
  expireMushroomsSystem,
  spawnMushroomsSystem,
} from './mushroom'
import { physicsSystem } from './physics'
import { applyMovementInputToPlayer, createPlayerEntity } from './player'
import { createRenderPlayerSystem } from './player-renderer'
import { updateWorldTimeSystem } from './time'

export function startGame(world: World) {
  listenForResize(world)

  observeGrid(world)
  observeLifetimes(world)

  createGridPositions(world)
  createPlayerEntity(world)

  const renderPlayerSystem = createRenderPlayerSystem(world)
  const renderBackgroundSystem = createRenderBackgroundSystem(world)
  const renderMushroomSystem = createRenderMushroomSystem(world)

  function tick(timeMs: number) {
    updateWorldTimeSystem(world, timeMs)

    applyMovementInputToPlayer(world)
    physicsSystem(world)

    spawnMushroomsSystem(world)
    expireMushroomsSystem(world)

    setKeyStateSystem()

    positionCameraSystem(world)
    renderBackgroundSystem(world)
    renderPlayerSystem(world)
    renderMushroomSystem(world)

    requestAnimationFrame(tick)
  }

  requestAnimationFrame(tick)
}
