import { vec2f } from 'typegpu/data'

import { setKeyStateSystem } from '../input'
import type { World } from '../main'

import { createRenderBackgroundSystem } from './background'
import { createRenderBulletSystem } from './bullet'
import { positionCameraSystem } from './camera'
import { createMushroom, createRenderMushroomSystem } from './mushroom'
import { physicsSystem } from './physics'
import { applyMovementInputToPlayer, createPlayerEntity } from './player'
import { createRenderPlayerSystem } from './player-renderer'

export function startGame(world: World) {
  createPlayerEntity(world)
  createMushroom(world, vec2f(-1, -1))

  const renderPlayerSystem = createRenderPlayerSystem(world)
  const renderBackgroundSystem = createRenderBackgroundSystem(world)
  const renderBulletSystem = createRenderBulletSystem(world)
  const renderMushroomSystem = createRenderMushroomSystem(world)

  let lastTimeMs = 0
  function tick(timeMs: number) {
    world.delta = (timeMs - lastTimeMs) / 1000
    world.time = timeMs / 1000
    lastTimeMs = timeMs

    applyMovementInputToPlayer(world)
    physicsSystem(world)

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
