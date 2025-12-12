import { keyboardInputSystem } from '../input'
import type { World } from '../main'

import { createRenderBackgroundSystem } from './background'
import { positionCameraSystem } from './camera'
import {
  applyDragSystem,
  applyMaxSpeedSystem,
  bounceOffBoundariesSystem,
  moveSystem,
} from './physics'
import {
  applyMovementInputToPlayer,
  createPlayerEntity,
  createRenderPlayerSystem,
} from './player'

export function startGame(world: World) {
  createPlayerEntity(world)

  const renderPlayerSystem = createRenderPlayerSystem(world)
  const renderBackgroundSystem = createRenderBackgroundSystem(world)

  let lastTime = 0
  function tick(time: number) {
    world.delta = (time - lastTime) / 1000
    lastTime = time

    applyMovementInputToPlayer(world)
    moveSystem(world)
    applyDragSystem(world)
    applyMaxSpeedSystem(world)
    bounceOffBoundariesSystem(world)

    keyboardInputSystem()

    positionCameraSystem(world)
    renderBackgroundSystem(world)
    renderPlayerSystem(world)

    requestAnimationFrame(tick)
  }

  requestAnimationFrame(tick)
}
