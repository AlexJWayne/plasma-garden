import { createWorld } from 'bitecs'

import { setupCamera } from './game/camera'
import { startGame } from './game/main'
import { setupTime } from './game/time'
import { setupInput } from './input'
import { setupWebgpu } from './setup-webgpu'

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

startGame(world)
