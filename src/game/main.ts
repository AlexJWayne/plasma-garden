import { vec2f } from 'typegpu/data'

import { setKeyStateSystem } from '../input'
import type { World } from '../main'
import { listenForResize } from '../setup-webgpu'

import { createRenderBackgroundSystem } from './background'
import { createRenderBulletSystem } from './bullet'
import { positionCameraSystem } from './camera'
import { createMushroom, createRenderMushroomSystem } from './mushroom'
import { physicsSystem } from './physics'
import { applyMovementInputToPlayer, createPlayerEntity } from './player'
import { createRenderPlayerSystem } from './player-renderer'
import { updateWorldTimeSystem } from './time'

export function startGame(world: World) {
  listenForResize(world)

  createPlayerEntity(world)

  createMushroom(world, vec2f(-1, -1), 1.5)
  createMushroom(world, vec2f(1, -1), 2)
  createMushroom(world, vec2f(2, 0), 1)
  createMushroom(world, vec2f(-1, -2), 3)

  const renderPlayerSystem = createRenderPlayerSystem(world)
  const renderBackgroundSystem = createRenderBackgroundSystem(world)
  const renderBulletSystem = createRenderBulletSystem(world)
  const renderMushroomSystem = createRenderMushroomSystem(world)

  function tick(timeMs: number) {
    updateWorldTimeSystem(world, timeMs)

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
