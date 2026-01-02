import { opSmoothDifference, sdBox2d, sdBox3d, sdSphere } from '@typegpu/sdf'
import { type TgpuBufferUniform, tgpu } from 'typegpu'
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
import {
  clamp,
  dot,
  length,
  max,
  mix,
  normalize,
  pow,
  reflect,
  round,
} from 'typegpu/std'

import { dither } from '../../lib/dither'
import { quadVertices } from '../../lib/geometry'
import { createPipelinePerformanceCallback } from '../../lib/pipeline-perf'
import { remap } from '../../lib/remap'
import {
  createColorAttachment,
  createDepthAttachment,
  depthStencil,
} from '../../lib/web-gpu'
import type { World } from '../../main'
import { presentationFormat, sampleCount } from '../../setup-webgpu'

import { CameraStruct } from './camera'

export function createRenderBackgroundSystem(world: World) {
  const renderPipeline = world.root['~unstable']
    .withVertex(createVertexProgram(world.camera.buffer.as('uniform')), {})
    .withFragment(createFragmentProgram(world.camera.buffer.as('uniform')), {
      format: presentationFormat,
    })
    .withDepthStencil(depthStencil)
    .withMultisample({ count: sampleCount })
    .createPipeline()
    .withPerformanceCallback(createPipelinePerformanceCallback('background'))

  function render(world: World) {
    renderPipeline
      .withColorAttachment({
        ...createColorAttachment(world),
        loadOp: 'clear',
      })
      .withDepthStencilAttachment({
        ...createDepthAttachment(world),
        depthLoadOp: 'clear',
        depthClearValue: 1,
      })
      .draw(quadVertices.$.length)
  }

  return render
}

function createVertexProgram(
  cameraBuffer: TgpuBufferUniform<typeof CameraStruct>,
) {
  return tgpu['~unstable'].vertexFn({
    in: { idx: builtin.vertexIndex },
    out: {
      worldPos: vec3f,
      clipPos: builtin.position,
      uv: vec2f,
    },
  })(({ idx }) => {
    const uv = quadVertices.$[idx]
    const worldPos = vec3f(uv.mul(10.5), 0)
    const clipPos = cameraBuffer.$.viewMatrix.mul(vec4f(worldPos, 1))
    return {
      worldPos,
      clipPos,
      uv,
    }
  })
}

function createFragmentProgram(
  cameraBuffer: TgpuBufferUniform<typeof CameraStruct>,
) {
  const MAX_DISTANCE = f32(20)
  const MAX_STEPS = 50
  const EPSILON = 0.01

  const Hit = struct({ hit: bool, pos: vec3f, totalDistance: f32 })
  type Hit = Infer<typeof Hit>

  const COLOR = vec3f(0.2, 0.3, 0.1)

  const main = tgpu['~unstable'].fragmentFn({
    in: {
      worldPos: vec3f,
      clipPos: builtin.position,
    },
    out: vec4f,
  })(({ worldPos, clipPos }) => {
    const hit = raymarch(worldPos)
    if (hit.hit) {
      const lighting = calcLighting(calcNormal(hit.pos), hit.pos)
      const color = getColor(hit.pos).mul(lighting)
      return vec4f(dither(color, clipPos.xy), 1)
    }
    return vec4f(vec3f(0.2), 1)
  })

  function scene(p: v3f): number {
    'use gpu'

    const repeatedP = vec3f(p.xy.sub(round(p.xy.div(1))), p.z)

    return opSmoothDifference(
      sdSphere(repeatedP.sub(vec3f(0, 0, -0.5)), 0.8),
      sdBox3d(repeatedP.sub(vec3f(0, 0, 0.5)), vec3f(10, 10, 0.5)),
      0.1,
    )
  }

  function getColor(hitPos: v3f): v3f {
    'use gpu'
    const repeatedP = vec2f(hitPos.xy.sub(round(hitPos.xy.div(1))))
    const d = clamp(sdBox2d(repeatedP, vec2f(0.35)) * 6, 0, 1)
    return mix(COLOR.mul(0.5), COLOR, d)
  }

  function calcLighting(normal: v3f, hitPos: v3f): number {
    'use gpu'
    const lightPos = cameraBuffer.$.playerPos
    const lightDistance = lightPos.sub(hitPos)
    const lightDir = normalize(lightDistance)
    const diffuse =
      max(dot(normal, lightDir), 0) *
      remap(length(lightDistance), f32(2), f32(8), f32(1), f32(0))
    const viewDir = normalize(cameraBuffer.$.pos.sub(hitPos))
    const specular = pow(
      max(dot(reflect(lightDir.mul(-1), normal), viewDir), 0),
      32,
    )
    return diffuse * 0.8 + specular * 0.2
  }

  function raymarch(worldPos: v3f): Hit {
    'use gpu'

    const triDiff = worldPos.sub(cameraBuffer.$.pos)
    let totalDistance = length(triDiff)
    const rayDirection = normalize(triDiff)

    for (let i = 0; i < MAX_STEPS; i++) {
      const point = cameraBuffer.$.pos.add(rayDirection.mul(totalDistance))
      const distance = scene(point)

      if (distance < EPSILON)
        return Hit({ hit: true, pos: point, totalDistance })
      if (distance > MAX_DISTANCE) break

      totalDistance += distance
    }

    return Hit({ hit: false, pos: vec3f(), totalDistance })
  }

  function calcNormal(p: v3f): v3f {
    'use gpu'
    const h = EPSILON
    const k = vec2f(1, -1)
    return normalize(
      k.xyy
        .mul(scene(p.add(k.xyy.mul(h))))
        .add(
          k.yyx
            .mul(scene(p.add(k.yyx.mul(h))))
            .add(
              k.yxy
                .mul(scene(p.add(k.yxy.mul(h))))
                .add(k.xxx.mul(scene(p.add(k.xxx.mul(h))))),
            ),
        ),
    )
  }

  return main
}
