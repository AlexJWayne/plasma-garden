import typegpuPlugin from 'unplugin-typegpu/vite'
import { defineConfig } from 'vite'

export default defineConfig({
  plugins: [typegpuPlugin({})],
  base: '/plasma-garden/',
})
