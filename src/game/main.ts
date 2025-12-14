import { vec2f } from 'typegpu/data'
import { normalize } from 'typegpu/std'

import { setKeyStateSystem } from '../input'
import type { World } from '../main'

import { createRenderBackgroundSystem } from './background'
import { createBullet, createRenderBulletSystem } from './bullet'
import { positionCameraSystem } from './camera'
import { physicsSystem } from './physics'
import { applyMovementInputToPlayer, createPlayerEntity } from './player'
import { createRenderPlayerSystem } from './player-renderer'

export function startGame(world: World) {
  createPlayerEntity(world)

  const renderPlayerSystem = createRenderPlayerSystem(world)
  const renderBackgroundSystem = createRenderBackgroundSystem(world)
  const renderBulletSystem = createRenderBulletSystem(world)

  let lastTime = 0
  function tick(time: number) {
    world.delta = (time - lastTime) / 1000
    lastTime = time

    applyMovementInputToPlayer(world)
    physicsSystem(world)

    setKeyStateSystem()

    positionCameraSystem(world)
    renderBackgroundSystem(world)
    renderPlayerSystem(world)
    renderBulletSystem(world)

    requestAnimationFrame(tick)
  }

  requestAnimationFrame(tick)

  // TEMP
  for (let i = 0; i < 10; i++) {
    createBullet(
      world,
      vec2f(0, 0),
      normalize(vec2f(Math.random() * 2 - 1, Math.random() * 2 - 1)).mul(0.2),
      10,
    )
  }
}
