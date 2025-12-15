import tgpu from 'typegpu'

export const presentationFormat = navigator.gpu.getPreferredCanvasFormat()
export const depthFormat: GPUTextureFormat = 'depth24plus'

const supersampling = 0.5
const canvasSize = 1000

// MSAA sample count: 1 (disabled), 4 (recommended), or 8 (high quality)
export const sampleCount = 4

export async function setupWebgpu() {
  const canvas = document.getElementById('canvas') as HTMLCanvasElement
  canvas.width = canvasSize * supersampling
  canvas.height = canvasSize * supersampling
  canvas.style.width = `${canvasSize}px`
  canvas.style.height = `${canvasSize}px`

  const ctx = canvas.getContext('webgpu')
  if (!ctx) {
    const message = 'WebGPU is not supported in this browser.'
    alert(message)
    throw new Error(message)
  }

  const root = await tgpu.init()

  ctx.configure({
    device: root.device,
    format: presentationFormat,
    alphaMode: 'premultiplied',
  })

  const depthTexture = root.device.createTexture({
    size: [canvas.width, canvas.height],
    format: depthFormat,
    usage: GPUTextureUsage.RENDER_ATTACHMENT,
    sampleCount,
  })

  // Create multisample texture for MSAA
  const colorTexture = root.device.createTexture({
    size: [canvas.width, canvas.height],
    format: presentationFormat,
    usage: GPUTextureUsage.RENDER_ATTACHMENT,
    sampleCount,
  })

  return { canvas, ctx, root, depthTexture, colorTexture }
}
