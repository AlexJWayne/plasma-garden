import tgpu from 'typegpu'

export const presentationFormat = navigator.gpu.getPreferredCanvasFormat()
export const depthFormat: GPUTextureFormat = 'depth24plus'

export async function setupWebgpu() {
  const canvas = document.getElementById('canvas') as HTMLCanvasElement
  canvas.width = 1000
  canvas.height = 1000

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
  })

  return { canvas, ctx, root, depthTexture }
}
