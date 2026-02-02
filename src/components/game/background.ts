import { opSmoothDifference, sdBox2d, sdBox3d, sdSphere } from '@typegpu/sdf'
import { addEntity, query } from 'bitecs'
import { type Infer, f32, struct, vec2f, vec3f } from 'typegpu/data'
import { clamp, mix, round } from 'typegpu/std'

import { SurfaceColors } from '../../lib/lighting'
import {
  AABB,
  createSDFInstancesRenderer,
} from '../../lib/sdf-instance-pipeline'
import type { World } from '../../main'

type Background = {}
const Background = [] as Background[]

const BackgroundStruct = struct({
  pos: vec3f,
})
type BackgroundStruct = Infer<typeof BackgroundStruct>

export function createBackgroundEntity(world: World): void {
  addEntity(world, Background)
}

export function createRenderBackgroundSystem(world: World) {
  const baseRender = createSDFInstancesRenderer({
    name: 'Background',
    world,

    instanceStruct: BackgroundStruct,
    instanceCapacity: 1,

    writeBuffers: (buffer) => {
      const backgrounds = query(world, [Background])
      if (backgrounds.length === 0) return 0

      buffer.write([{ pos: vec3f(0) }])

      return backgrounds.length
    },

    calcAABB: () => {
      'use gpu'
      return AABB({
        min: vec3f(-10.5, -10.5, -1),
        max: vec3f(10.5, 10.5, 0),
      })
    },

    sdSurface: (p, _bg, _elapsed) => {
      'use gpu'

      const repeatedP = vec3f(p.xy.sub(round(p.xy.div(1))), p.z)

      return opSmoothDifference(
        sdSphere(repeatedP.sub(vec3f(0, 0, -0.5)), 0.8),
        sdBox3d(repeatedP.sub(vec3f(0, 0, 0.5)), vec3f(10, 10, 0.5)),
        0.1,
      )
    },

    calcSurfaceColors: (hitPos, _bg, _elapsed) => {
      'use gpu'

      const COLOR = vec3f(0.2, 0.3, 0.1)

      const repeatedP = vec2f(hitPos.xy.sub(round(hitPos.xy.div(1))))
      const d = clamp(sdBox2d(repeatedP, vec2f(0.35)) * 6, 0, 1)
      const diffuseColor = mix(COLOR.mul(0.5), COLOR, d)

      return SurfaceColors({
        diffuse: diffuseColor.mul(0.8),
        specular: vec3f(0.25),
        emissive: vec3f(0),
        shininess: f32(500),
      })
    },

    maxSteps: 20,
    maxDistance: 50,
    epsilon: 0.01,
    epsilonNormal: 0.01,

    debug: false,
  })

  return baseRender
}
