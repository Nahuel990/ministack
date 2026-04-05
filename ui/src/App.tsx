import { Routes, Route, Navigate } from 'react-router-dom'
import Layout from './components/Layout'
import Dashboard from './pages/Dashboard'
import ResourceBrowser from './pages/ResourceBrowser'
import LogViewer from './pages/LogViewer'
import RequestInspector from './pages/RequestInspector'

export default function App() {
  return (
    <Layout>
      <Routes>
        <Route path="/" element={<Dashboard />} />
        <Route path="/resources" element={<ResourceBrowser />} />
        <Route path="/resources/:service" element={<ResourceBrowser />} />
        <Route path="/logs" element={<LogViewer />} />
        <Route path="/requests" element={<RequestInspector />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </Layout>
  )
}
