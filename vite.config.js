import { defineConfig } from 'vite';
import typegpuPlugin from 'unplugin-typegpu/vite';

export default defineConfig({
  base: '/raymond',
  plugins: [typegpuPlugin()],
});
