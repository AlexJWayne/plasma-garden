import { type v2f } from 'typegpu/data'

// Physics
export const Position = [] as v2f[]
export const Velocity = [] as v2f[]
export const Acceleration = [] as v2f[]
export const Drag = [] as number[]
export const MaxSpeed = [] as number[]

// Game Objects
export const Player = {}

export const Camera = {}

export const Bullet = [] as Bullet[]
export type Bullet = {
  damage: number
}
