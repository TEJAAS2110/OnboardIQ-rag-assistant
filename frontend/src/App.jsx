import { useState, useEffect } from 'react'
import './App.css'
import ChatInterface from './components/chat/ChatInterface.jsx';
import { listDocuments, uploadDocument, checkHealth } from './services/api'

function App() {
  const [stats, setStats] = useState({
    totalDocs: 0,
    totalChunks: 0,
  })
  const [uploadKey, setUploadKey] = useState(0)
  const [isOnline, setIsOnline] = useState(true)

  useEffect(() => {
    loadStats()
    checkHealthStatus()
  }, [uploadKey])

  const loadStats = async () => {
    try {
      const data = await listDocuments()
      setStats({
        totalDocs: data.total_documents,
        totalChunks: data.total_chunks,
      })
      setIsOnline(true)
    } catch (error) {
      console.error('Error loading stats:', error)
      setIsOnline(false)
    }
  }

  const checkHealthStatus = async () => {
    try {
      await checkHealth()
      setIsOnline(true)
    } catch {
      setIsOnline(false)
    }
  }

  const handleFileUpload = async (event) => {
    const file = event.target.files[0]
    if (!file) return

    const maxSize = 10 * 1024 * 1024 // 10MB
    if (file.size > maxSize) {
      alert('❌ File too large! Maximum size is 10MB')
      return
    }

    const uploadBtn = document.querySelector('.upload-btn')
    const originalText = uploadBtn.textContent
    uploadBtn.textContent = '⏳ Uploading...'
    uploadBtn.disabled = true

    try {
      const result = await uploadDocument(file)
      
      if (result.success) {
        alert(`✅ Document uploaded successfully!\n\n📄 ${result.file_name}\n📊 Created ${result.chunks_created} chunks\n💾 ${(file.size / 1024).toFixed(1)} KB`)
        setUploadKey(prev => prev + 1)
      } else {
        alert('❌ Upload failed: ' + (result.error || 'Unknown error'))
      }
    } catch (error) {
      console.error('Upload error:', error)
      alert('❌ Connection error. Make sure backend is running and CORS is enabled')
      setIsOnline(false)
    } finally {
      uploadBtn.textContent = originalText
      uploadBtn.disabled = false
      event.target.value = ''
    }
  }

  const handleRefresh = () => {
    setUploadKey(prev => prev + 1)
    loadStats()
  }

  const API_DOCS_URL = import.meta.env.VITE_API_URL 
    ? `${import.meta.env.VITE_API_URL}/docs`
    : 'https://onboardiiq-api.onrender.com/docs'

  return (
    <div className="app-container">
      <div className="sidebar">
        <div>
          <h1 className="app-title">🎯 OnboardIQ</h1>
          <p className="app-subtitle">Intelligent Knowledge Assistant</p>
        </div>

        <div className="stats-card neon-glow">
          <h3>📊 Knowledge Base</h3>
          <div className="stat-item">
            <span className="stat-label">Documents</span>
            <span className="stat-value">{stats.totalDocs}</span>
          </div>
          <div className="stat-item">
            <span className="stat-label">Knowledge Chunks</span>
            <span className="stat-value">{stats.totalChunks}</span>
          </div>
          <div className="stat-item">
            <span className="stat-label">System Status</span>
            <span className="stat-value">{isOnline ? '🟢 Online' : '🔴 Offline'}</span>
          </div>
        </div>

        <div className="quick-actions">
          <button 
            className="action-btn"
            onClick={() => window.open(API_DOCS_URL, '_blank')}
            title="View API Documentation"
          >
            📊 API Documentation
          </button>
          <button 
            className="action-btn"
            onClick={handleRefresh}
            title="Refresh statistics"
          >
            🔄 Refresh Statistics
          </button>
          <button 
            className="action-btn"
            onClick={() => {
              if (confirm('Clear all chat history? This cannot be undone.')) {
                window.location.reload()
              }
            }}
            title="Clear chat history"
          >
            🗑️ Clear Chat
          </button>
        </div>

        <div className="upload-section">
          <input
            type="file"
            id="file-upload"
            style={{ display: 'none' }}
            onChange={handleFileUpload}
            accept=".pdf,.docx,.txt,.md,.html"
          />
          <button
            className="upload-btn"
            onClick={() => document.getElementById('file-upload').click()}
          >
            📤 Upload Document
          </button>
          <p style={{ fontSize: '10px', color: 'rgba(255,255,255,0.4)', marginTop: '8px', textAlign: 'center' }}>
            Supports: PDF, DOCX, TXT, MD, HTML (Max 10MB)
          </p>
        </div>

        <div className="footer-section">
          <p style={{ fontSize: '11px', color: 'rgba(255,255,255,0.4)', textAlign: 'center' }}>
            <strong>OnboardIQ v1.0</strong><br/>
            Powered by Advanced RAG<br/>
            Hybrid Search • Multi-Language • Citations
          </p>
        </div>
      </div>

      <div className="main-content">
        <ChatInterface key={uploadKey} isOnline={isOnline} />
      </div>
    </div>
  )
}

export default App
