import { query } from 'bitecs'
import { type TgpuRoot } from 'typegpu'
import { mat4x4f, struct, vec3f } from 'typegpu/data'
import { mat4 } from 'wgpu-matrix'

import { lookAt } from '../lib/matrix'
import type { World } from '../main'

import { Player, Position } from './components'
import { PLAYER_HEIGHT } from './player'

const FOV = 45 * (Math.PI / 180)
const NEAR = 0.1
const FAR = 100

export const CAMERA_OFFSET = vec3f(0, -1, 1).mul(1.5)
const UP = vec3f(0, 0, 1)

export const CameraStruct = struct({
  viewMatrix: mat4x4f,
  pos: vec3f,
  targetPos: vec3f,
})

export function setupCamera(root: TgpuRoot) {
  const buffer = root.createBuffer(CameraStruct).$usage('uniform')
  const target = { current: vec3f(0, 0, PLAYER_HEIGHT) }
  return { buffer, target }
}

export function positionCameraSystem(world: World) {
  const { camera } = world

  const player = query(world, [Player, Position])[0]
  const playerPos = vec3f(Position[player], PLAYER_HEIGHT)

  camera.target.current = camera.target.current.add(
    playerPos.sub(camera.target.current).mul(3 * world.time.delta),
  )
  const pos = camera.target.current.add(CAMERA_OFFSET)

  const projection = mat4.perspective(
    FOV,
    world.canvas.width / world.canvas.height,
    NEAR,
    FAR,
    mat4x4f(),
  )

  const viewMatrix = projection.mul(lookAt(pos, camera.target.current, UP))
  camera.buffer.write({
    viewMatrix,
    pos,
    targetPos: camera.target.current,
  })
}
