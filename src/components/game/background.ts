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
import { clamp, length, mix, normalize, round } from 'typegpu/std'

import { dither } from '../../lib/dither'
import { quadVertices } from '../../lib/geometry'
import { Lighting, Surface, calcSurfaceLighting } from '../../lib/lighting'
import { createPipelinePerformanceCallback } from '../../lib/pipeline-perf'
import { createCalcNormal, createRaymarch } from '../../lib/raymarching'
import {
  createColorAttachment,
  createDepthAttachment,
  depthStencil,
} from '../../lib/web-gpu'
import type { World } from '../../main'
import { presentationFormat, sampleCount } from '../../setup-webgpu'

import { CameraStruct, worldToClipSpace } from './camera'

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
    const clipPos = worldToClipSpace(cameraBuffer.$, worldPos)
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

  const COLOR = vec3f(0.2, 0.3, 0.1)

  function scene(p: v3f, _dummyArg: number): number {
    'use gpu'

    const repeatedP = vec3f(p.xy.sub(round(p.xy.div(1))), p.z)

    return opSmoothDifference(
      sdSphere(repeatedP.sub(vec3f(0, 0, -0.5)), 0.8),
      sdBox3d(repeatedP.sub(vec3f(0, 0, 0.5)), vec3f(10, 10, 0.5)),
      0.1,
    )
  }
  const calcNormal = createCalcNormal(scene, EPSILON)
  const raymarch = createRaymarch(scene, {
    maxSteps: MAX_STEPS,
    maxDistance: MAX_DISTANCE,
    epsilon: EPSILON,
  })

  const main = tgpu['~unstable'].fragmentFn({
    in: {
      worldPos: vec3f,
      clipPos: builtin.position,
    },
    out: vec4f,
  })(({ worldPos, clipPos }) => {
    const hit = raymarch(cameraBuffer.$.pos, worldPos, 0)
    if (hit.isHit) {
      const color = calcSurfaceLighting(
        Lighting({
          cameraPos: cameraBuffer.$.pos,
          lightPos: cameraBuffer.$.playerPos,
          surfacePos: hit.pos,
          normal: calcNormal(hit.pos, 0),
          surface: Surface({
            diffuse: getColor(hit.pos).mul(0.8),
            specular: vec3f(0.1),
            emissive: vec3f(0),
            shininess: f32(32),
          }),
        }),
      )
      return vec4f(dither(color, clipPos.xy), 1)
    }
    return vec4f(vec3f(0.2), 1)
  })

  function getColor(hitPos: v3f): v3f {
    'use gpu'
    const repeatedP = vec2f(hitPos.xy.sub(round(hitPos.xy.div(1))))
    const d = clamp(sdBox2d(repeatedP, vec2f(0.35)) * 6, 0, 1)
    return mix(COLOR.mul(0.5), COLOR, d)
  }

  return main
}
