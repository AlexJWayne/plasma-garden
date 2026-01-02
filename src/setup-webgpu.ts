import tgpu, { type TgpuRoot } from 'typegpu'

import type { World } from './main'

export const presentationFormat = navigator.gpu.getPreferredCanvasFormat()
export const depthFormat: GPUTextureFormat = 'depth24plus'

const supersampling = 1

// MSAA sample count: 1 (disabled), 4 (recommended), or 8 (high quality)
export const sampleCount = 4

export async function setupWebgpu() {
  const canvas = document.getElementById('canvas') as HTMLCanvasElement
  setCanvasSize(canvas)

  const ctx = canvas.getContext('webgpu')
  if (!ctx) {
    const message = 'WebGPU is not supported in this browser.'
    alert(message)
    throw new Error(message)
  }

  const root = await tgpu.init({
    device: {
      requiredFeatures: ['timestamp-query'],
    },
  })

  ctx.configure({
    device: root.device,
    format: presentationFormat,
    alphaMode: 'premultiplied',
  })

  const { depthTexture, colorTexture } = createRenderTextures(root, {
    width: canvas.width,
    height: canvas.height,
  })

  return { canvas, ctx, root, depthTexture, colorTexture }
}

export function listenForResize(world: World) {
  window.addEventListener('resize', () => {
    const { width, height } = setCanvasSize(world.canvas)

    const { depthTexture, colorTexture } = createRenderTextures(world.root, {
      width: width,
      height: height,
    })

    world.colorTexture = colorTexture
    world.depthTexture = depthTexture
  })
}

function setCanvasSize(canvas: HTMLCanvasElement): {
  width: number
  height: number
} {
  canvas.width = window.innerWidth * supersampling
  canvas.height = window.innerHeight * supersampling
  canvas.style.width = `${window.innerWidth}px`
  canvas.style.height = `${window.innerHeight}px`
  return { width: canvas.width, height: canvas.height }
}

function createRenderTextures(
  root: TgpuRoot,
  { width, height }: { width: number; height: number },
) {
  return {
    depthTexture: root.device.createTexture({
      size: [width, height],
      format: depthFormat,
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
      sampleCount,
    }),
    colorTexture: root.device.createTexture({
      size: [width, height],
      format: presentationFormat,
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
      sampleCount,
    }),
  }
}
