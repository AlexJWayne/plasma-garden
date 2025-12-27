export function remap(
  value: number,
  fromMin: number,
  fromMax: number,
  toMin: number,
  toMax: number,
): number {
  'use gpu'
  return ((value - fromMin) / (fromMax - fromMin)) * (toMax - toMin) + toMin
}
