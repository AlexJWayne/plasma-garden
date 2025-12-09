import { createRenderBackgroundSystem } from './game-objects/background'
import { positionCameraSystem, setupCamera } from './game-objects/camera'
import { bounceOffBoundariesSystem, moveSystem } from './game-objects/physics'
import {
  createPlayerEntity,
  createRenderPlayerSystem,
} from './game-objects/player'
import { setupWebgpu } from './setup-webgpu'
import { createWorld } from 'bitecs'

const { canvas, ctx, root } = await setupWebgpu()
const camera = setupCamera(root)

const world = createWorld({
  canvas,
  ctx,
  root,
  camera,
})
export type World = typeof world

createPlayerEntity(world)

const renderPlayerSystem = createRenderPlayerSystem(world)
const renderBackgroundSystem = createRenderBackgroundSystem(world)

function tick() {
  moveSystem(world)
  bounceOffBoundariesSystem(world)

  positionCameraSystem(world)
  renderBackgroundSystem(world)
  renderPlayerSystem(world)

  requestAnimationFrame(tick)
}

requestAnimationFrame(tick)
