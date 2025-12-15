import {
  opSmoothUnion,
  opUnion,
  sdBoxFrame3d,
  sdPie,
  sdSphere,
} from '@typegpu/sdf'
import { query } from 'bitecs'
import tgpu, { type TgpuBufferUniform } from 'typegpu'
import {
  type Infer,
  bool,
  builtin,
  f32,
  struct,
  type v3f,
  vec2f,
  vec3f,
  vec4f,
} from 'typegpu/data'
import { discard, length, normalize, smoothstep } from 'typegpu/std'

import { cubeVertices } from '../lib/geometry'
import { sdTorus } from '../lib/sdf'
import { blending } from '../lib/web-gpu'
import type { World } from '../main'
import { depthFormat, presentationFormat, sampleCount } from '../setup-webgpu'

import type { CameraStruct } from './camera'
import { Player, Position } from './components'
import { SIZE } from './player'

const PlayerStruct = struct({
  position: vec2f,
})

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
        world.camera.buffer.as('uniform'),
        playerBuffer.as('uniform'),
      ),
      {
        color: {
          format: presentationFormat,
          blend: blending.normal,
        },
      },
    )
    .withDepthStencil({
      format: depthFormat,
      depthWriteEnabled: true,
      depthCompare: 'less',
    })
    .withPrimitive({
      topology: 'triangle-list',
      cullMode: 'back',
    })
    .withMultisample({ count: sampleCount })
    .createPipeline()

  function render(world: World) {
    const player = query(world, [Player, Position])[0]

    playerBuffer.write({
      position: Position[player],
    })

    renderPipeline
      .withColorAttachment({
        color: {
          view: world.colorTexture.createView(),
          resolveTarget: world.ctx.getCurrentTexture().createView(),
          loadOp: 'load',
          storeOp: 'store',
        },
      })
      .withDepthStencilAttachment({
        view: world.depthTexture.createView(),
        depthLoadOp: 'load',
        depthStoreOp: 'store',
      })
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
      localPos: vec3f,
      worldPos: vec3f,
      clipPos: builtin.position,
    },
  })(({ idx }) => {
    const localPos = cubeVertices.$[idx]
    const worldPos = localPos
      .mul(SIZE / 2)
      .add(vec3f(playerBuffer.$.position, SIZE / 2))
    const clipPos = cameraBuffer.$.viewMatrix.mul(vec4f(worldPos, 1))
    return {
      localPos,
      worldPos,
      clipPos,
    }
  })
}

function createFragmentProgram(
  cameraBuffer: TgpuBufferUniform<typeof CameraStruct>,
  playerBuffer: TgpuBufferUniform<typeof PlayerStruct>,
) {
  return tgpu['~unstable'].fragmentFn({
    in: { localPos: vec3f, worldPos: vec3f },
    out: { color: vec4f, depth: builtin.fragDepth },
  })(({ localPos, worldPos }) => {
    const hit = raymarch(
      cameraBuffer.$.pos,
      worldPos,
      vec3f(playerBuffer.$.position, SIZE / 2),
    )
    // if (!hit.hit) return { color: vec4f(0.1), depth: 0 } // debug
    if (!hit.hit) return { color: vec4f(0), depth: 1 }

    const hitClipPos = cameraBuffer.$.viewMatrix.mul(vec4f(hit.pos, 1))

    return {
      color: vec4f(vec3f(0, smoothstep(-0.1, SIZE + 0.1, hit.pos.z), 0), 1),
      depth: hitClipPos.z / hitClipPos.w,
    }
  })
}

function scene(p: v3f, playerPos: v3f): number {
  'use gpu'
  const centeredP = playerPos.sub(p)

  let dist = f32(1e9)
  dist = opUnion(
    dist,
    sdBoxFrame3d(centeredP, vec3f((SIZE / 2) * 0.9), SIZE * 0.015),
  )
  dist = opUnion(dist, sdTorus(centeredP, SIZE * 0.3, SIZE * 0.05))
  return dist
}

const Hit = struct({ hit: bool, pos: vec3f, totalDistance: f32 })
function raymarch(
  cameraPos: v3f,
  worldPos: v3f,
  playerPos: v3f,
): Infer<typeof Hit> {
  'use gpu'

  const MAX_DISTANCE = f32(100)
  const MAX_STEPS = 100
  const EPSILON = 0.0001

  const triDiff = worldPos.sub(cameraPos)
  let totalDistance = length(triDiff)
  const rayDirection = normalize(triDiff)

  for (let i = 0; i < MAX_STEPS; i++) {
    const point = cameraPos.add(rayDirection.mul(totalDistance))
    const distance = scene(point, playerPos)

    if (distance < EPSILON) return Hit({ hit: true, pos: point, totalDistance })
    if (distance > MAX_DISTANCE) break

    totalDistance += distance
  }

  return Hit({ hit: false, pos: vec3f(), totalDistance })
}
