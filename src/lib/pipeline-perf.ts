export const pipelinePerfMs: Record<string, number> = {}
const perfEl = createPerfEl()

function createPerfEl() {
  const perfEl = document.body.appendChild(document.createElement('div'))
  perfEl.style.position = 'absolute'
  perfEl.style.top = '0'
  perfEl.style.left = '0'
  perfEl.style.zIndex = '9999'
  perfEl.style.color = '#ffffff88'
  perfEl.style.fontSize = '14px'
  perfEl.style.fontFamily = 'JetBrainsMono Nerd Font'
  perfEl.style.margin = '8px'
  return perfEl
}

export function createPipelinePerformanceCallback(name: string) {
  const times = Array(10).fill(0)
  let timesIndex = 0

  return (start: bigint, end: bigint) => {
    const time = Number(end - start) / 1_000_000
    times[timesIndex] = time
    timesIndex = (timesIndex + 1) % times.length
    pipelinePerfMs[name] = times.reduce((a, b) => a + b, 0) / times.length
  }
}

setInterval(() => {
  perfEl.innerHTML = Object.entries(pipelinePerfMs)
    .map(([name, timeMs]) => `${timeMs.toFixed(1)}ms ${name}`)
    .join('<br>')
}, 200)
