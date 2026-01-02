import { setKeyStateSystem } from '../input'
import { createGridPositions } from '../lib/grid'
import type { World } from '../main'
import { listenForResize } from '../setup-webgpu'

import { createRenderBackgroundSystem } from './background'
import { createRenderBulletSystem } from './bullet'
import { positionCameraSystem } from './camera'
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

  createGridPositions(world)
  createPlayerEntity(world)

  const renderPlayerSystem = createRenderPlayerSystem(world)
  const renderBackgroundSystem = createRenderBackgroundSystem(world)
  const renderBulletSystem = createRenderBulletSystem(world)
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
    renderBulletSystem(world)
    renderMushroomSystem(world)

    requestAnimationFrame(tick)
  }

  requestAnimationFrame(tick)
}
