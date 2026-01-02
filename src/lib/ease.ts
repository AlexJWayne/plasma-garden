// Easing functions for GPU shader use with typegpu
// Based on standard easing equations from easings.net
import { abs, cos, exp, pow, sin, sqrt } from 'typegpu/std'

// ============================================================================
// Linear
// ============================================================================

export function linear(t: number): number {
  'use gpu'
  return t
}

// ============================================================================
// Sine Easing
// ============================================================================

export function easeInSine(t: number): number {
  'use gpu'
  return 1 - cos((t * Math.PI) / 2)
}

export function easeOutSine(t: number): number {
  'use gpu'
  return sin((t * Math.PI) / 2)
}

export function easeInOutSine(t: number): number {
  'use gpu'
  return -(cos(Math.PI * t) - 1) / 2
}

// ============================================================================
// Quadratic Easing
// ============================================================================

export function easeInQuad(t: number): number {
  'use gpu'
  return t * t
}

export function easeOutQuad(t: number): number {
  'use gpu'
  return 1 - (1 - t) * (1 - t)
}

export function easeInOutQuad(t: number): number {
  'use gpu'
  if (t < 0.5) {
    return 2 * t * t
  }
  return 1 - pow(-2 * t + 2, 2) / 2
}

// ============================================================================
// Cubic Easing
// ============================================================================

export function easeInCubic(t: number): number {
  'use gpu'
  return t * t * t
}

export function easeOutCubic(t: number): number {
  'use gpu'
  return 1 - pow(1 - t, 3)
}

export function easeInOutCubic(t: number): number {
  'use gpu'
  if (t < 0.5) {
    return 4 * t * t * t
  }
  return 1 - pow(-2 * t + 2, 3) / 2
}

// ============================================================================
// Quartic Easing
// ============================================================================

export function easeInQuart(t: number): number {
  'use gpu'
  return t * t * t * t
}

export function easeOutQuart(t: number): number {
  'use gpu'
  return 1 - pow(1 - t, 4)
}

export function easeInOutQuart(t: number): number {
  'use gpu'
  if (t < 0.5) {
    return 8 * t * t * t * t
  }
  return 1 - pow(-2 * t + 2, 4) / 2
}

// ============================================================================
// Quintic Easing
// ============================================================================

export function easeInQuint(t: number): number {
  'use gpu'
  return t * t * t * t * t
}

export function easeOutQuint(t: number): number {
  'use gpu'
  return 1 - pow(1 - t, 5)
}

export function easeInOutQuint(t: number): number {
  'use gpu'
  if (t < 0.5) {
    return 16 * t * t * t * t * t
  }
  return 1 - pow(-2 * t + 2, 5) / 2
}

// ============================================================================
// Exponential Easing
// ============================================================================

export function easeInExpo(t: number): number {
  'use gpu'
  if (t === 0) return 0
  return pow(2, 10 * t - 10)
}

export function easeOutExpo(t: number): number {
  'use gpu'
  if (t === 1) return 1
  return 1 - pow(2, -10 * t)
}

export function easeInOutExpo(t: number): number {
  'use gpu'
  if (t === 0) return 0
  if (t === 1) return 1
  if (t < 0.5) {
    return pow(2, 20 * t - 10) / 2
  }
  return (2 - pow(2, -20 * t + 10)) / 2
}

// ============================================================================
// Circular Easing
// ============================================================================

export function easeInCirc(t: number): number {
  'use gpu'
  return 1 - sqrt(1 - pow(t, 2))
}

export function easeOutCirc(t: number): number {
  'use gpu'
  return sqrt(1 - pow(t - 1, 2))
}

export function easeInOutCirc(t: number): number {
  'use gpu'
  if (t < 0.5) {
    return (1 - sqrt(1 - pow(2 * t, 2))) / 2
  }
  return (sqrt(1 - pow(-2 * t + 2, 2)) + 1) / 2
}

// ============================================================================
// Back Easing (overshoots)
// ============================================================================

export function easeInBack(t: number): number {
  'use gpu'
  const c1 = 1.70158
  const c3 = c1 + 1
  return c3 * t * t * t - c1 * t * t
}

export function easeOutBack(t: number): number {
  'use gpu'
  const c1 = 1.70158
  const c3 = c1 + 1
  return 1 + c3 * pow(t - 1, 3) + c1 * pow(t - 1, 2)
}

export function easeInOutBack(t: number): number {
  'use gpu'
  const c1 = 1.70158
  const c2 = c1 * 1.525
  if (t < 0.5) return (pow(2 * t, 2) * ((c2 + 1) * 2 * t - c2)) / 2
  return (pow(2 * t - 2, 2) * ((c2 + 1) * (t * 2 - 2) + c2) + 2) / 2
}

// ============================================================================
// Elastic Easing (spring-like)
// ============================================================================

export function easeInElastic(t: number): number {
  'use gpu'
  const c4 = (2 * Math.PI) / 3
  if (t === 0) return 0
  if (t === 1) return 1
  return -pow(2, 10 * t - 10) * sin((t * 10 - 10.75) * c4)
}

export function easeOutElastic(t: number): number {
  'use gpu'
  const c4 = (2 * Math.PI) / 3
  if (t === 0) return 0
  if (t === 1) return 1
  return pow(2, -10 * t) * sin((t * 10 - 0.75) * c4) + 1
}

export function easeInOutElastic(t: number): number {
  'use gpu'
  const c5 = (2 * Math.PI) / 4.5
  if (t === 0) return 0
  if (t === 1) return 1
  if (t < 0.5) {
    return -(pow(2, 20 * t - 10) * sin((20 * t - 11.125) * c5)) / 2
  }
  return (pow(2, -20 * t + 10) * sin((20 * t - 11.125) * c5)) / 2 + 1
}

// ============================================================================
// Bounce Easing
// ============================================================================

export function easeOutBounce(t: number): number {
  'use gpu'
  const n1 = 7.5625
  const d1 = 2.75

  if (t < 1 / d1) {
    return n1 * t * t
  } else if (t < 2 / d1) {
    const t2 = t - 1.5 / d1
    return n1 * t2 * t2 + 0.75
  } else if (t < 2.5 / d1) {
    const t2 = t - 2.25 / d1
    return n1 * t2 * t2 + 0.9375
  } else {
    const t2 = t - 2.625 / d1
    return n1 * t2 * t2 + 0.984375
  }
}

export function easeInBounce(t: number): number {
  'use gpu'
  return 1 - easeOutBounce(1 - t)
}

export function easeInOutBounce(t: number): number {
  'use gpu'
  if (t < 0.5) return (1 - easeOutBounce(1 - 2 * t)) / 2
  return (1 + easeOutBounce(2 * t - 1)) / 2
}

// ============================================================================
// Utility Functions
// ============================================================================

export function smoothstep(t: number): number {
  'use gpu'
  return t * t * (3 - 2 * t)
}

export function smootherstep(t: number): number {
  'use gpu'
  return t * t * t * (t * (t * 6 - 15) + 10)
}

export function parabola(t: number, k: number): number {
  'use gpu'
  return pow(4 * t * (1 - t), k)
}

export function gain(t: number, k: number): number {
  'use gpu'
  const a = 0.5 * pow(2 * t, k)
  const b = 1 - 0.5 * pow(2 - 2 * t, k)
  if (t < 0.5) {
    return a
  }
  return b
}

export function impulse(t: number, k: number): number {
  'use gpu'
  const h = k * t
  return h * exp(1 - h)
}

export function expImpulse(t: number, k: number): number {
  'use gpu'
  const h = k * t
  return h * exp(1 - h)
}

export function cubicPulse(center: number, width: number, t: number): number {
  'use gpu'
  const x = abs(t - center)
  if (x > width) return 0
  const x2 = x / width
  return 1 - x2 * x2 * (3 - 2 * x2)
}

export function expStep(t: number, k: number, n: number): number {
  'use gpu'
  return exp(-k * pow(t, n))
}

export function bounce(t: number): number {
  'use gpu'
  return abs(sin(2 * Math.PI * t * (1 + t)))
}
