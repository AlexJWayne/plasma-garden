import type { TgpuRoot } from 'typegpu'
import { f32, struct } from 'typegpu/data'

import type { World } from '../main'

export const TimeStruct = struct({
  elapsed: f32,
})

export function setupTime(root: TgpuRoot) {
  const buffer = root.createBuffer(TimeStruct).$usage('uniform')
  return {
    elapsed: 0,
    delta: 0,
    buffer,
  }
}

let lastTimeMs = 0
export function updateWorldTimeSystem({ time }: World, timeMs: number) {
  time.delta = (timeMs - lastTimeMs) / 1000
  time.elapsed = timeMs / 1000
  lastTimeMs = timeMs

  time.buffer.write({ elapsed: time.elapsed })
}
