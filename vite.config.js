import { defineConfig } from 'vite';
import typegpuPlugin from 'unplugin-typegpu/vite';
import basicSsl from '@vitejs/plugin-basic-ssl'

export default defineConfig({
  base: '/raymond',
  plugins: [
    typegpuPlugin(),
    // Required for WebGPU to work on Mobile Safari in local dev
    basicSsl(),
  ],
  server: {
    https: true,
  },
});
