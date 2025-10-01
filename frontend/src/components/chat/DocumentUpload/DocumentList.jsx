import { useState, useEffect } from 'react'
import { listDocuments } from '../../services/api'
import './DocumentList.css'

export default function DocumentList({ onRefresh }) {
  const [documents, setDocuments] = useState([])
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    loadDocuments()
  }, [onRefresh])

  const loadDocuments = async () => {
    setLoading(true)
    try {
      const data = await listDocuments()
      setDocuments(data.documents || [])
    } catch (error) {
      console.error('Error loading documents:', error)
    } finally {
      setLoading(false)
    }
  }

  const handleDelete = async (fileName) => {
    if (!window.confirm(`Delete ${fileName}?`)) return

    try {
      const response = await fetch(`http://localhost:8000/documents/${fileName}`, {
        method: 'DELETE',
      })
      
      if (response.ok) {
        alert('Document deleted!')
        loadDocuments()
        if (onRefresh) onRefresh()
      }
    } catch (error) {
      console.error('Delete error:', error)
    }
  }

  if (loading) {
    return (
      <div style={{padding: '20px', color: 'rgba(255,255,255,0.5)', fontSize: '13px', textAlign: 'center'}}>
        Loading documents...
      </div>
    )
  }

  if (documents.length === 0) {
    return (
      <div style={{padding: '20px', color: 'rgba(255,255,255,0.5)', fontSize: '13px', textAlign: 'center'}}>
        No documents yet
      </div>
    )
  }

  return (
    <div className="document-list">
      <h3>Uploaded Documents</h3>
      <div className="documents-grid">
        {documents.map((doc) => (
          <div key={doc.file_name} className="document-card">
            <div className="doc-icon">📄</div>
            <div className="doc-info">
              <div className="doc-name" title={doc.file_name}>
                {doc.file_name}
              </div>
              <div className="doc-meta">
                {doc.chunk_count} chunks • {doc.file_type}
              </div>
            </div>
            <button
              onClick={() => handleDelete(doc.file_name)}
              className="delete-btn"
              title="Delete document"
            >
              🗑️
            </button>
          </div>
        ))}
      </div>
    </div>
  )
}