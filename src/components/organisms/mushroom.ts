import {
  opSmoothDifference,
  opSmoothUnion,
  opUnion,
  sdBox3d,
  sdCapsule,
  sdSphere,
} from '@typegpu/sdf'
import { addEntity, query, set } from 'bitecs'
import { type Infer, arrayOf, f32, struct, vec3f } from 'typegpu/data'
import { abs, atan2, clamp, fract, sin, smoothstep } from 'typegpu/std'

import { easeInCubic, easeInSine } from '../../lib/ease'
import { hsl2rgb } from '../../lib/hsl'
import { SurfaceColors } from '../../lib/lighting'
import { remap } from '../../lib/remap'
import { sdCone } from '../../lib/sdf'
import {
  AABB,
  createSDFInstancesRenderer,
} from '../../lib/sdf-instance-pipeline'
import { rotate2d, rotateX } from '../../lib/transform'
import type { World } from '../../main'
import { GridPosition, getRandomEmptyGridPosition } from '../general/grid'
import { Lifetime, getLifetimeCompletion } from '../general/lifetime'

type Mushroom = {
  height: number
  lobes: number
  stemRadius: number
  capRadius: number
}
const Mushroom = [] as Mushroom[]

const MushroomStruct = struct({
  pos: vec3f,
  height: f32,
  lobes: f32,
  stemRadius: f32,
  capRadius: f32,
  completion: f32,
})
type MushroomStruct = Infer<typeof MushroomStruct>

const SPAWN_RATE = 0.05

export function createMushroom(world: World) {
  const gridPosition = getRandomEmptyGridPosition(world)
  if (!gridPosition) return

  const eid = addEntity(
    world,
    set(GridPosition, gridPosition),
    set(Lifetime, Math.random() * 30 + 15),
    Mushroom,
  )
  Mushroom[eid] = {
    height: Math.random() * 1.5 + 0.5,
    lobes: Math.floor(Math.random() * 8) + 3,
    stemRadius: Math.random() * 0.04 + 0.04,
    capRadius: Math.random() * 0.7 + 0.3,
  }
}

export function spawnMushroomsSystem(world: World) {
  if (Math.random() < SPAWN_RATE) createMushroom(world)
}

export function createRenderMushroomSystem(world: World) {
  const mushroomsBuffer = world.root
    .createBuffer(arrayOf(MushroomStruct, 1000))
    .$usage('storage')

  const renderMushroomSystem = createSDFInstancesRenderer({
    name: 'Mushroom',
    world,

    timeBuffer: world.time.buffer.as('uniform'),
    cameraBuffer: world.camera.buffer.as('uniform'),
    instanceBuffer: mushroomsBuffer.as('readonly'),

    writeBuffers: () => {
      const mushrooms = query(world, [Mushroom, GridPosition])
      if (mushrooms.length === 0) return 0

      mushroomsBuffer.writePartial(
        [...mushrooms].map((eid, idx) => {
          const completion = getLifetimeCompletion(world, eid)
          return {
            idx,
            value: {
              pos: vec3f(GridPosition[eid], 0),
              height: Mushroom[eid].height,
              lobes: Mushroom[eid].lobes,
              stemRadius: Mushroom[eid].stemRadius,
              capRadius: Mushroom[eid].capRadius,
              completion,
            },
          }
        }),
      )

      return mushrooms.length
    },

    calcAABB: (mushroom) => {
      'use gpu'
      const growth = easeInSine(mushroom.completion)
      const capRadius = mushroom.capRadius * growth
      const height = mushroom.height * growth

      return AABB({
        min: mushroom.pos.sub(vec3f(capRadius, capRadius, 0)),
        max: mushroom.pos.add(vec3f(capRadius, capRadius, height)),
      })
    },

    sdSurface: (p, mushroom, _elapsed) => {
      'use gpu'
      const localP = p.sub(mushroom.pos)

      const growth = easeInSine(mushroom.completion)
      const stemRadius = mushroom.stemRadius * growth
      const capRadius = mushroom.capRadius * growth
      const height = mushroom.height * growth

      const stem = opSmoothUnion(
        sdCapsule(
          localP,
          vec3f(0, 0, stemRadius),
          vec3f(0, 0, height - stemRadius),
          stemRadius,
        ),
        sdCone(
          rotateX(localP.sub(vec3f(0, 0, height - 0.2)), -Math.PI / 2),
          0.15,
          height - 0.2,
        ),
        0.15,
      )

      const capCenter = localP.sub(vec3f(0, 0, height - capRadius))
      let cap = sdSphere(capCenter, capRadius)
      cap = opSmoothDifference(
        cap,
        sdBox3d(localP, vec3f(1, 1, height - capRadius * 0.4)),
        0.05,
      )

      const expiryGrowth = clamp(
        remap(mushroom.completion, 0.65, f32(1), f32(0), f32(1)),
        f32(0),
        f32(1),
      )
      const expiry = sdCapsule(
        p,
        mushroom.pos.sub(vec3f(0, 0, 100)),
        mushroom.pos.add(
          vec3f(0, 0, -capRadius + (capRadius + height * 1.25) * expiryGrowth),
        ),
        capRadius * expiryGrowth,
      )

      return opSmoothDifference(opUnion(stem, cap), expiry, 0.1)
    },

    postProcessNormal: (normal, p, mushroom) => {
      'use gpu'
      const angle = atan2(p.y - mushroom.pos.y, p.x - mushroom.pos.x)
      return vec3f(
        rotate2d(normal.xy, sin(angle * mushroom.lobes) * 0.3),
        normal.z,
      )
    },

    calcSurfaceColors: (hitPos, mushroom, elapsed) => {
      'use gpu'

      const baseHueShift =
        (fract(mushroom.pos.x * 1.1577 + mushroom.pos.y * 3.7193) * 2 - 1) * 0.1
      const baseColor = hsl2rgb(vec3f(0.66 + baseHueShift, 0.4, 0.4))

      const angle = atan2(hitPos.y - mushroom.pos.y, hitPos.x - mushroom.pos.x)

      const glowZFactor = easeInCubic(
        hitPos.z / (mushroom.height * mushroom.completion),
      )
      const glowZ = glowZFactor * (mushroom.height * mushroom.completion)

      let glowValue =
        (glowZ + sin(angle * mushroom.lobes) * 0.01) * 2 +
        elapsed * 0.1 +
        mushroom.pos.x +
        mushroom.pos.y
      glowValue = abs(fract(glowValue) - 0.5) * 2

      let glowWhiteValue = smoothstep(0.95, 1, glowValue)
      glowValue = smoothstep(0.8, 1, glowValue)

      const glowHue = fract(
        hitPos.z +
          mushroom.pos.x * 3.77 +
          mushroom.pos.y * 5.37 +
          elapsed * 0.1,
      )
      const glowColor = hsl2rgb(vec3f(glowHue, 1, 0.6))
        .mul(glowValue)
        .add(vec3f(glowWhiteValue))
        .mul(easeInCubic(hitPos.z / mushroom.height))

      return SurfaceColors({
        diffuse: baseColor,
        specular: vec3f(1, 0.5, 0),
        emissive: glowColor,
        shininess: f32(64),
      })
    },

    maxSteps: 50,
    maxDistance: 100,
    epsilon: 0.001,
    epsilonNormal: 0.01,

    debug: false,
  })

  return renderMushroomSystem
}
