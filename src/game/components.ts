import { type v2f, vec2f, vec3f } from 'typegpu/data'

const MAX_ENT = 10_000

function createVec2Component<T>(extend: T = {} as T) {
  const Component = {
    x: new Float32Array(MAX_ENT),
    y: new Float32Array(MAX_ENT),

    set(eid: number, x: number, y: number) {
      Component.x[eid] = x
      Component.y[eid] = y
    },

    add(eid: number, x: number, y: number) {
      Component.x[eid] += x
      Component.y[eid] += y
    },

    asVec2f(eid: number) {
      return vec2f(Component.x[eid], Component.y[eid])
    },

    asVec3f(eid: number) {
      return vec3f(Component.x[eid], Component.y[eid], 0)
    },

    ...extend,
  }

  return Component
}

export const Position = [] as v2f[]
export const Velocity = createVec2Component({
  maxSpeed: new Float32Array(MAX_ENT),
})
export const Drag = new Float32Array(MAX_ENT)

export const Player = {}

export const Camera = {}
