import { type Infer, f32, struct, type v3f, vec3f } from 'typegpu/data'
import { add, clamp, dot, length, max, normalize, saturate } from 'typegpu/std'

import { remap } from './remap'

const FALLOFF_START = f32(2)
const FALLOFF_END = f32(8)

export const Surface = struct({
  diffuse: vec3f,
  specular: vec3f,
  emissive: vec3f,
  shininess: f32,
})
export type Surface = Infer<typeof Surface>

export const LightingPositions = struct({
  cameraPos: vec3f,
  lightPos: vec3f,
  surfacePos: vec3f,
  normal: vec3f,
})
export type LightingPositions = Infer<typeof LightingPositions>

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

export function calcDiffuseLighting({
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

export function calcSpecularLighting({
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

export function calcLighting(
  config: Infer<typeof SpecularLighting>,
  diffuseColor: v3f,
  specularColor: v3f,
): v3f {
  'use gpu'

  const diffuse = calcDiffuseLighting(
    DiffuseLighting({
      lightPos: config.lightPos,
      surfacePos: config.surfacePos,
      normal: config.normal,
    }),
  )
  const specular = calcSpecularLighting(config)

  return clamp(
    add(
      diffuseColor.mul(diffuse), //
      specularColor.mul(specular),
    ),
    vec3f(0),
    vec3f(1),
  )
}

export function calcSurfaceLighting(
  surface: Surface,
  lighting: LightingPositions,
): v3f {
  'use gpu'
  const color = calcLighting(
    SpecularLighting({
      cameraPos: lighting.cameraPos,
      lightPos: lighting.lightPos,
      surfacePos: lighting.surfacePos,
      normal: lighting.normal,
      shininess: surface.shininess,
    }),
    surface.diffuse,
    surface.specular,
  )

  return saturate(color.add(surface.emissive))
}
