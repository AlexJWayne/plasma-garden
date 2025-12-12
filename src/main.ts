import { createWorld } from 'bitecs'

import { setupCamera } from './game/camera'
import { startGame } from './game/main'
import { setupInput } from './input'
import { setupWebgpu } from './setup-webgpu'

const { canvas, ctx, root } = await setupWebgpu()

const world = createWorld({
  canvas,
  ctx,
  root,
  camera: setupCamera(root),
  input: setupInput(),
  delta: 0,
})
export type World = typeof world

startGame(world)
