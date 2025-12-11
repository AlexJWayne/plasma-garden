import { lookAt } from '../lib-gpu'
import type { World } from '../main'
import { Player, Position } from './components'
import { query } from 'bitecs'
import { type TgpuRoot } from 'typegpu'
import { mat4x4f, struct, vec3f } from 'typegpu/data'
import { mat4 } from 'wgpu-matrix'

export const projection = mat4.perspective(
  (60 * Math.PI) / 180, // fov
  1, // aspect
  0.1, // near
  100, // far
  mat4x4f(),
)

const offset = vec3f(0, -0.5, 1)
const up = vec3f(0, 1, 0)

export const CameraStruct = struct({ viewMatrix: mat4x4f })

export function setupCamera(root: TgpuRoot) {
  const buffer = root.createBuffer(CameraStruct).$usage('uniform')
  let target = { current: vec3f() }
  return { buffer, target }
}

export function positionCameraSystem(world: World) {
  const { camera } = world

  const player = query(world, [Player, Position])[0]
  const playerPos = Position.asVec3f(player)

  camera.target.current = camera.target.current.add(
    playerPos.sub(camera.target.current).mul(10 * world.delta),
  )

  const viewMatrix = projection.mul(
    lookAt(camera.target.current.add(offset), camera.target.current, up),
  )
  camera.buffer.write({ viewMatrix })
}
