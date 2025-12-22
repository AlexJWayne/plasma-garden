import {
  opSmoothUnion,
  opUnion,
  sdBoxFrame3d,
  sdCapsule,
  sdSphere,
} from '@typegpu/sdf'
import { query } from 'bitecs'
import tgpu, { type TgpuBufferUniform } from 'typegpu'
import {
  type Infer,
  bool,
  builtin,
  f32,
  mat4x4f,
  struct,
  type v3f,
  vec2f,
  vec3f,
  vec4f,
} from 'typegpu/data'
import { abs, fract, length, normalize, sin } from 'typegpu/std'
import { mat4 } from 'wgpu-matrix'

import { cubeVertices } from '../lib/geometry'
import {
  blending,
  createColorAttachment,
  createDepthAttachment,
  depthStencil,
} from '../lib/web-gpu'
import type { World } from '../main'
import { presentationFormat, sampleCount } from '../setup-webgpu'

import type { CameraStruct } from './camera'
import { Player, Position, Velocity } from './components'
import { SIZE } from './player'
import type { TimeStruct } from './time'

const DEBUG = false

const PlayerStruct = struct({
  position: vec2f,
  transform: mat4x4f,
  inverseTransform: mat4x4f,
})
type PlayerStruct = Infer<typeof PlayerStruct>

export function createRenderPlayerSystem(world: World) {
  const playerBuffer = world.root.createBuffer(PlayerStruct).$usage('uniform')

  const renderPipeline = world.root['~unstable']
    .withVertex(
      createVertexProgram(
        world.camera.buffer.as('uniform'),
        playerBuffer.as('uniform'),
      ),
      {},
    )
    .withFragment(
      createFragmentProgram(
        world.time.buffer.as('uniform'),
        world.camera.buffer.as('uniform'),
        playerBuffer.as('uniform'),
      ),
      { color: { format: presentationFormat, blend: blending.normal } },
    )
    .withDepthStencil(depthStencil)
    .withPrimitive({ topology: 'triangle-list', cullMode: 'back' })
    .withMultisample({ count: sampleCount })
    .createPipeline()

  function render(world: World) {
    const player = query(world, [Player, Position, Velocity])[0]
    const vel = Velocity[player]

    const transform = mat4x4f
      .translation(vec3f(Position[player], 0))
      .mul(mat4x4f.rotationZ(Math.atan2(vel.y, vel.x)))

    playerBuffer.write({
      position: Position[player],
      transform,
      inverseTransform: mat4.invert(transform, mat4x4f()),
    })

    renderPipeline
      .withColorAttachment({ color: createColorAttachment(world) })
      .withDepthStencilAttachment(createDepthAttachment(world))
      .draw(cubeVertices.value.length)
  }

  return render
}

function createVertexProgram(
  cameraBuffer: TgpuBufferUniform<typeof CameraStruct>,
  playerBuffer: TgpuBufferUniform<typeof PlayerStruct>,
) {
  return tgpu['~unstable'].vertexFn({
    in: { idx: builtin.vertexIndex },
    out: {
      worldPos: vec3f,
      clipPos: builtin.position,
    },
  })(({ idx }) => {
    const localPos = cubeVertices.$[idx].mul(SIZE / 2)
    const worldPos = playerBuffer.$.transform.mul(vec4f(localPos, 1))
    const clipPos = cameraBuffer.$.viewMatrix.mul(worldPos)
    return {
      worldPos: worldPos.xyz,
      clipPos,
    }
  })
}

function createFragmentProgram(
  timeBuffer: TgpuBufferUniform<typeof TimeStruct>,
  cameraBuffer: TgpuBufferUniform<typeof CameraStruct>,
  playerBuffer: TgpuBufferUniform<typeof PlayerStruct>,
) {
  return tgpu['~unstable'].fragmentFn({
    in: { worldPos: vec3f },
    out: { color: vec4f, depth: builtin.fragDepth },
  })(({ worldPos }) => {
    const hit = raymarch(
      cameraBuffer.$.pos,
      worldPos,
      playerBuffer.$,
      timeBuffer.$.elapsed,
    )

    if (DEBUG && !hit.hit) return { color: vec4f(1, 0, 1, 1), depth: 0 }
    if (!hit.hit) return { color: vec4f(0), depth: 1 }

    const hitClipPos = cameraBuffer.$.viewMatrix.mul(vec4f(hit.pos, 1))

    return {
      color: vec4f(
        vec3f(
          0,
          abs(fract((hit.pos.z + timeBuffer.$.elapsed * 0.005) * 50) - 0.5) +
            0.5,
          0,
        ),
        1,
      ),
      depth: hitClipPos.z / hitClipPos.w,
    }
  })
}

function scene(p: v3f, player: PlayerStruct, time: number): number {
  'use gpu'
  const centeredP = player.inverseTransform.mul(vec4f(p, 1)).xyz

  let dist = f32(1e9)
  // frame
  dist = opUnion(dist, sdBoxFrame3d(centeredP, vec3f(SIZE / 2), SIZE * 0.005))

  // Head
  dist = opUnion(
    dist,
    sdSphere(centeredP.sub(vec3f(SIZE * 0.2, 0, 0)), SIZE * 0.2),
  )

  // Torso
  dist = opSmoothUnion(
    dist,
    sdSphere(
      centeredP.sub(
        vec3f(
          -SIZE * 0.05, //
          sin(time * 3) * SIZE * 0.07,
          0,
        ),
      ),
      SIZE * 0.125,
    ),
    SIZE * 0.05,
  )

  // Tail
  dist = opSmoothUnion(
    dist,
    sdCapsule(
      centeredP,
      vec3f(-SIZE * 0.4, 0, 0),
      vec3f(SIZE * 0.4, 0, 0),
      SIZE * 0.01,
    ),
    SIZE * 0.2,
  )

  // wings
  dist = opSmoothUnion(
    dist,
    sdCapsule(
      centeredP,
      vec3f(SIZE * 0.2, -SIZE * 0.4, 0),
      vec3f(SIZE * 0.2, SIZE * 0.4, 0),
      SIZE * 0.05,
    ),
    SIZE * 0.2,
  )
  return dist
}

const Hit = struct({ hit: bool, pos: vec3f })
type Hit = Infer<typeof Hit>

function raymarch(
  cameraPos: v3f,
  worldPos: v3f,
  player: PlayerStruct,
  time: number,
): Hit {
  'use gpu'

  const MAX_DISTANCE = f32(100)
  const MAX_STEPS = 100
  const EPSILON = 0.0001

  const triDiff = worldPos.sub(cameraPos)
  let totalDistance = length(triDiff)
  const rayDirection = normalize(triDiff)

  for (let i = 0; i < MAX_STEPS; i++) {
    const point = cameraPos.add(rayDirection.mul(totalDistance))
    const distance = scene(point, player, time)

    if (distance < EPSILON) return Hit({ hit: true, pos: point })
    if (distance > MAX_DISTANCE) break

    totalDistance += distance
  }

  return Hit({ hit: false, pos: vec3f() })
}
