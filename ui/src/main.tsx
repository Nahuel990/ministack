import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { BrowserRouter } from 'react-router-dom'
import { TooltipProvider } from '@/components/ui/tooltip'
import { Toaster } from '@/components/ui/sonner'
import App from './App'
import './index.css'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <BrowserRouter basename="/_ministack/ui">
      <TooltipProvider>
        <App />
        <Toaster richColors />
      </TooltipProvider>
    </BrowserRouter>
  </StrictMode>,
)
