import { type Infer, f32, struct, type v3f, vec3f } from 'typegpu/data'
import { clamp, dot, length, max, mul, normalize, saturate } from 'typegpu/std'

import { remap } from './remap'

const FALLOFF_START = f32(2)
const FALLOFF_END = f32(8)

export const SurfaceColors = struct({
  diffuse: vec3f,
  specular: vec3f,
  emissive: vec3f,
  shininess: f32,
})
export type SurfaceColors = Infer<typeof SurfaceColors>

export const Lighting = struct({
  cameraPos: vec3f,
  lightPos: vec3f,
  surfacePos: vec3f,
  normal: vec3f,
  surface: SurfaceColors,
})
export type Lighting = Infer<typeof Lighting>

export const DiffuseLighting = struct({
  lightPos: vec3f,
  surfacePos: vec3f,
  normal: vec3f,
})

export const SpecularLighting = struct({
  lightPos: vec3f,
  surfacePos: vec3f,
  normal: vec3f,
  cameraPos: vec3f,
  shininess: f32,
})

function calcDiffuseLighting({
  lightPos,
  surfacePos,
  normal,
}: Infer<typeof DiffuseLighting>): number {
  'use gpu'

  // Calculate light direction from surface to light
  const lightDir = lightPos.sub(surfacePos)
  const lightDirNormalized = normalize(lightDir)

  // Calculate diffuse intensity using Lambertian reflectance
  let diffuse = max(dot(lightDirNormalized, normal), 0)

  // Apply lighting fade over distance
  diffuse *= clamp(
    remap(length(lightDir), FALLOFF_START, FALLOFF_END, f32(1), f32(0)),
    0,
    1,
  )

  return diffuse
}

function calcSpecularLighting({
  lightPos,
  surfacePos,
  normal,
  cameraPos,
  shininess,
}: Infer<typeof SpecularLighting>): number {
  'use gpu'

  // Calculate light direction from surface to light
  const lightDir = lightPos.sub(surfacePos)
  const lightDirNormalized = normalize(lightDir)

  // Calculate view direction from surface to camera
  const viewDir = normalize(cameraPos.sub(surfacePos))

  // Calculate halfway vector (Blinn-Phong)
  const halfwayDir = normalize(lightDirNormalized.add(viewDir))

  // Calculate specular intensity
  const spec = max(dot(normal, halfwayDir), 0)
  let specular = spec ** shininess

  // Apply lighting fade over distance
  specular *= clamp(
    remap(length(lightDir), FALLOFF_START, FALLOFF_END, f32(1), f32(0)),
    0,
    1,
  )

  return specular
}

export function calcSurfaceLighting({
  cameraPos,
  lightPos,
  surfacePos,
  normal,
  surface,
}: Lighting): v3f {
  'use gpu'

  const diffuse = calcDiffuseLighting(
    DiffuseLighting({
      lightPos: lightPos,
      surfacePos: surfacePos,
      normal: normal,
    }),
  )
  const specular = calcSpecularLighting(
    SpecularLighting({
      cameraPos: cameraPos,
      lightPos: lightPos,
      surfacePos: surfacePos,
      normal: normal,
      shininess: surface.shininess,
    }),
  )

  return saturate(
    mul(surface.diffuse, diffuse)
      .add(mul(surface.specular, specular))
      .add(surface.emissive),
  )
}
