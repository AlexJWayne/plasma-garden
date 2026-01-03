import { createWorld } from 'bitecs'

import { createRenderBackgroundSystem } from './components/game/background'
import { setupCamera } from './components/game/camera'
import { positionCameraSystem } from './components/game/camera'
import {
  applyMovementInputToPlayer,
  createPlayerEntity,
} from './components/game/player'
import { createRenderPlayerSystem } from './components/game/player-renderer'
import { createGridPositions, observeGrid } from './components/general/grid'
import {
  killExpiredLifetimesSystem,
  observeLifetimes,
} from './components/general/lifetime'
import { physicsSystem } from './components/general/physics'
import {
  createRenderKelpSystem,
  spawnKelpSystem,
} from './components/organisms/kelp'
import {
  createRenderMushroomSystem,
  spawnMushroomsSystem,
} from './components/organisms/mushroom'
import { setupInput } from './input'
import { setKeyStateSystem } from './input'
import { setupWebgpu } from './setup-webgpu'
import { listenForResize } from './setup-webgpu'
import { setupTime } from './time'
import { updateWorldTimeSystem } from './time'

const { canvas, ctx, root, depthTexture, colorTexture } = await setupWebgpu()

const world = createWorld({
  canvas,
  ctx,
  root,
  depthTexture,
  colorTexture,
  camera: setupCamera(root),
  input: setupInput(),
  time: setupTime(root),
})
export type World = typeof world

export function startGame(world: World) {
  listenForResize(world)

  observeGrid(world)
  observeLifetimes(world)

  createGridPositions(world)
  createPlayerEntity(world)

  const renderPlayerSystem = createRenderPlayerSystem(world)
  const renderBackgroundSystem = createRenderBackgroundSystem(world)
  const renderMushroomSystem = createRenderMushroomSystem(world)
  const renderKelpSystem = createRenderKelpSystem(world)

  function tick(timeMs: number) {
    updateWorldTimeSystem(world, timeMs)

    applyMovementInputToPlayer(world)
    physicsSystem(world)

    spawnMushroomsSystem(world)
    spawnKelpSystem(world)

    killExpiredLifetimesSystem(world)

    setKeyStateSystem()

    positionCameraSystem(world)
    renderBackgroundSystem(world)
    renderPlayerSystem(world)
    renderMushroomSystem(world)
    renderKelpSystem(world)

    requestAnimationFrame(tick)
  }

  requestAnimationFrame(tick)
}

startGame(world)
