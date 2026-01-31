import { createWorld } from 'bitecs'

import {
  createBackgroundEntity,
  createRenderBackgroundSystem,
} from './components/game/background'
import { setupCamera } from './components/game/camera'
import { positionCameraSystem } from './components/game/camera'
import {
  applyMovementInputToPlayer,
  createPlayerEntity,
  createRenderPlayerSystem,
} from './components/game/player'
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
import { clearScreenSystem } from './lib/clear-screen'
import { setupWebgpu } from './setup-webgpu'
import { listenForResize } from './setup-webgpu'
import { createRenderWorldSystem } from './systems/render-world'
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
  createBackgroundEntity(world)

  const renderWorldSystem = createRenderWorldSystem(world)

  function tick(timeMs: number) {
    updateWorldTimeSystem(world, timeMs)

    applyMovementInputToPlayer(world)
    physicsSystem(world)

    spawnMushroomsSystem(world)
    spawnKelpSystem(world)

    killExpiredLifetimesSystem(world)

    setKeyStateSystem()

    positionCameraSystem(world)

    renderWorldSystem()

    requestAnimationFrame(tick)
  }

  requestAnimationFrame(tick)
}

startGame(world)
