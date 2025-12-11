type KeyState = 'up' | 'pressed' | 'down' | 'released'

const keysState: Record<
  'arrowup' | 'arrowdown' | 'arrowleft' | 'arrowright' | string,
  KeyState
> = {
  arrowup: 'up',
  arrowdown: 'up',
  arrowleft: 'up',
  arrowright: 'up',
}

export function setupInput() {
  document.addEventListener('keydown', onKeyDown)
  document.addEventListener('keyup', onKeyUp)

  return {
    state: keysState,
    isDown: (key: string) =>
      keysState[key] === 'down' || keysState[key] === 'pressed',
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
