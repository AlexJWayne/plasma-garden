import { perlin3d } from '@typegpu/noise'
import { opExtrudeY, opSmoothUnion, sdLine } from '@typegpu/sdf'
import { addEntity, query, set } from 'bitecs'
import {
  type Infer,
  arrayOf,
  f32,
  struct,
  type v3f,
  vec2f,
  vec3f,
} from 'typegpu/data'
import { abs, saturate, sin, smoothstep } from 'typegpu/std'

import { hsl2rgb } from '../../lib/hsl'
import { SurfaceColors } from '../../lib/lighting'
import { randomRange } from '../../lib/random'
import { sdLink } from '../../lib/sdf'
import {
  AABB,
  createSDFInstancesRenderer,
} from '../../lib/sdf-instance-pipeline'
import { rotate2d } from '../../lib/transform'
import type { World } from '../../main'
import { GridPosition, getRandomEmptyGridPosition } from '../general/grid'
import { Lifetime, getLifetimeCompletion } from '../general/lifetime'

type Kelp = {
  height: number
  twist: number
  seed: number
}
const Kelp = [] as Kelp[]

const KelpStruct = struct({
  entityPos: vec3f,
  height: f32,
  twist: f32,
  growth: f32,
  seed: f32,
})
type KelpStruct = Infer<typeof KelpStruct>

const SPAWN_RATE = 0.05

export function createKelp(world: World): void {
  const gridPosition = getRandomEmptyGridPosition(world)
  if (!gridPosition) return

  const eid = addEntity(
    world,
    set(GridPosition, gridPosition),
    set(Lifetime, randomRange(10, 30)),
    Kelp,
  )
  Kelp[eid] = {
    height: randomRange(3, 6),
    twist: randomRange(0.75, 2) * (Math.random() > 0.5 ? 1 : -1),
    seed: Math.random(),
  }
}

export function spawnKelpSystem(world: World): void {
  if (Math.random() < SPAWN_RATE) createKelp(world)
}

export function createRenderKelpSystem(world: World) {
  const kelpsBuffer = world.root
    .createBuffer(arrayOf(KelpStruct, 400))
    .$usage('storage')

  const renderKelpSystem = createSDFInstancesRenderer({
    name: 'Kelp',
    world,

    instanceBuffer: kelpsBuffer.as('readonly'),

    writeBuffers: () => {
      const kelps = query(world, [Kelp, GridPosition, Lifetime])
      if (kelps.length === 0) return 0

      kelpsBuffer.writePartial(
        [...kelps].map((eid, idx) => ({
          idx,
          value: {
            entityPos: vec3f(GridPosition[eid], 0),
            height: Kelp[eid].height,
            growth: getLifetimeCompletion(world, eid),
            twist: Kelp[eid].twist,
            seed: Kelp[eid].seed,
          },
        })),
      )

      return kelps.length
    },

    calcAABB: ({ entityPos, height, growth }) => {
      'use gpu'
      const horizontalExtent = (0.9 * (1 - growth)) / 2
      return AABB({
        min: entityPos.add(vec3f(vec2f(-horizontalExtent), 0)),
        max: entityPos.add(vec3f(vec2f(horizontalExtent), height * growth)),
      })
    },

    sdSurface: (p, kelp, _elapsed): number => {
      'use gpu'

      const twistedP = calcTwistedP(p, kelp)
      const narrowness = 1 - kelp.growth
      const height = kelp.height * kelp.growth

      const linkThickness = 0.03
      const radius = narrowness * 0.4 - linkThickness

      const membrane2d =
        sdLine(
          twistedP.xz,
          vec2f(0, -radius),
          vec2f(0, height - radius - linkThickness),
        ) - radius
      const membrane = opExtrudeY(twistedP, membrane2d, 0.01)

      const membraneCenter = height * 0.5 - radius - linkThickness
      const centeredP = twistedP.sub(vec3f(0, 0, membraneCenter))
      const offsetP = vec3f(centeredP.xzy)
      const border = sdLink(offsetP, height * 0.5, radius, linkThickness)

      return opSmoothUnion(membrane, border, 0.07)
    },

    calcSurfaceColors: (worldPos, kelp, _elapsed) => {
      'use gpu'

      const twistedP = calcTwistedP(worldPos, kelp)
      const p = twistedP.xz

      const amplitude = sin((p.y - kelp.growth * 2) * 20) * 0.025
      const waveD = abs(p.x - amplitude)
      const waveGlow = smoothstep(0.1, 0, waveD) ** 8

      const noise = perlin3d.sample(
        vec3f(
          p.x, //
          p.y - kelp.growth * 5,
          kelp.growth * 0.25,
        ).mul(12),
      )

      return SurfaceColors({
        diffuse: hsl2rgb(
          vec3f(
            0.33, //
            0.4 + waveD * 0.5,
            0.08 + waveD,
          ),
        ),
        specular: hsl2rgb(vec3f(0.5, 0.5, saturate(1 - noise))),
        emissive: hsl2rgb(
          vec3f(
            0.6,
            0.7,
            saturate(waveGlow + smoothstep(0.0, 0.7, noise)) ** 4,
          ).mul(0.8),
        ),
        shininess: f32(32),
      })
    },

    maxSteps: 100,
    maxDistance: 100,
    epsilon: 0.003,
    epsilonNormal: 0.01,

    debug: false,
  })

  return renderKelpSystem
}

const calcTwistedP = (p: v3f, kelp: KelpStruct): v3f => {
  'use gpu'

  const localP = p.sub(kelp.entityPos)
  const twistAngle = localP.z * kelp.twist + (Math.PI / 2) * kelp.seed

  return vec3f(rotate2d(localP.xy, twistAngle), localP.z)
}
