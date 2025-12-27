import type { World } from '../main'
import { depthFormat } from '../setup-webgpu'

export const blending = {
  // premultiplied alpha
  normal: {
    color: {
      srcFactor: 'one',
      dstFactor: 'one-minus-src-alpha',
    },
    alpha: {
      srcFactor: 'one',
      dstFactor: 'one-minus-src-alpha',
    },
  },
} satisfies Record<string, GPUBlendState>

export const depthStencil = {
  format: depthFormat,
  depthWriteEnabled: true,
  depthCompare: 'less',
} satisfies GPUDepthStencilState

export const depthStencilTransparent = {
  format: depthFormat,
  depthWriteEnabled: false,
  depthCompare: 'less',
} satisfies GPUDepthStencilState

export function createColorAttachment(world: World) {
  return {
    view: world.colorTexture.createView(),
    resolveTarget: world.ctx.getCurrentTexture().createView(),
    loadOp: 'load',
    storeOp: 'store',
  } as const
}

export function createDepthAttachment(world: World) {
  return {
    view: world.depthTexture.createView(),
    depthLoadOp: 'load',
    depthStoreOp: 'store',
  } as const
}
