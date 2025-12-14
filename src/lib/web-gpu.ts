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
