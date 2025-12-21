import tgpu, { type ValidateBufferSchema } from 'typegpu'
import { type AnyWgslData, type WgslArray, arrayOf } from 'typegpu/data'

import type { World } from '../main'

export function createInstanceBuffer<StructData extends AnyWgslData>(
  world: World,
  struct: StructData,
  length: number,
) {
  const instanceBuffer = world.root
    .createBuffer(
      arrayOf(struct, length) as ValidateBufferSchema<WgslArray<StructData>>,
    )
    // @ts-expect-error ???
    .$usage('vertex', 'storage')

  const instanceLayout = tgpu.vertexLayout(
    (n) => arrayOf(struct, n),
    'instance',
  )
  return [instanceBuffer, instanceLayout] as const
}
