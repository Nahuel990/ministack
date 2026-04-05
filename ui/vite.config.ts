import path from "path"
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  base: '/_ministack/ui/',
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  build: {
    outDir: '../ministack/ui/dist',
    emptyOutDir: true,
  },
  server: {
    proxy: {
      '/_ministack/api': 'http://localhost:4566',
    },
  },
})
