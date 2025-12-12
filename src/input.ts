type KeyState = 'up' | 'pressed' | 'down' | 'released'

const keysState: Record<
  'arrowup' | 'arrowdown' | 'arrowleft' | 'arrowright' | string,
  KeyState
> = {
  arrowup: 'up',
  arrowdown: 'up',
  arrowleft: 'up',
  arrowright: 'up',
  w: 'up',
  a: 'up',
  s: 'up',
  d: 'up',
}

const directions = {
  up: ['arrowup', 'w'],
  down: ['arrowdown', 's'],
  left: ['arrowleft', 'a'],
  right: ['arrowright', 'd'],
}

export function setupInput() {
  document.addEventListener('keydown', onKeyDown)
  document.addEventListener('keyup', onKeyUp)

  function isDown(key: string) {
    return keysState[key] === 'down' || keysState[key] === 'pressed'
  }

  function isDirectionDown(direction: keyof typeof directions) {
    return directions[direction].some(isDown)
  }

  return {
    state: keysState,
    isDown,
    isDirectionDown,
  }
}

function onKeyDown(event: KeyboardEvent) {
  const key = event.key.toLowerCase()
  if (!(key in keysState)) return
  const state = keysState[key]
  if (state === 'up') keysState[key] = 'pressed'
}

function onKeyUp(event: KeyboardEvent) {
  const key = event.key.toLowerCase()
  if (!(key in keysState)) return
  const state = keysState[key]
  if (state === 'down') keysState[key] = 'released'
}

export function keyboardInputSystem() {
  for (const key in keysState) {
    const state = keysState[key]
    if (state === 'pressed') keysState[key] = 'down'
    if (state === 'released') keysState[key] = 'up'
  }
}
