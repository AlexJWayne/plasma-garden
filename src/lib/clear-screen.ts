import type { World } from '../main'

export function clearScreenSystem(world: World) {
  const commandEncoder = world.root.device.createCommandEncoder()
  const passEncoder = commandEncoder.beginRenderPass({
    colorAttachments: [
      {
        view: world.colorTexture.createView(),
        resolveTarget: world.ctx.getCurrentTexture().createView(),
        loadOp: 'clear',
        clearValue: { r: 0, g: 0, b: 0, a: 1 },
        storeOp: 'store',
      },
    ],
    depthStencilAttachment: {
      view: world.depthTexture.createView(),
      depthLoadOp: 'clear',
      depthClearValue: 1,
      depthStoreOp: 'store',
    },
  })
  passEncoder.end()
  world.root.device.queue.submit([commandEncoder.finish()])
}
