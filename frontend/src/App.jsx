import { useState, useRef, useEffect } from 'react'
import { sendQuery, submitFeedback } from '../../services/api'
import './ChatInterface.css'

export default function ChatInterface({ isOnline }) {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [language, setLanguage] = useState('English')
  const messagesEndRef = useRef(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  const suggestedQuestions = [
    "How many days of leave do I get?",
    "What is the dress code policy?",
    "Who do I contact for IT support?",
    "How do I apply for remote work?",
    "What are the working hours?"
  ]

  const handleSend = async (text = input) => {
    if (!text.trim() || loading) return

    if (!isOnline) {
      alert('Backend is offline. Please start the backend server.')
      return
    }

    const userMessage = {
      role: 'user',
      content: text,
      timestamp: new Date().toISOString(),
    }

    setMessages((prev) => [...prev, userMessage])
    setInput('')
    setLoading(true)

    try {
      const languagePrompt = language !== 'English' 
        ? `Please answer in ${language}. ` 
        : ''
      
      const response = await sendQuery(languagePrompt + text, messages)
      
      const assistantMessage = {
        role: 'assistant',
        content: response.answer,
        citations: response.citations,
        confidence: response.confidence,
        timestamp: new Date().toISOString(),
      }

      setMessages((prev) => [...prev, assistantMessage])
    } catch (error) {
      console.error('Error:', error)
      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          content: 'Sorry, I encountered an error. Please ensure documents are uploaded and backend server is running.',
          timestamp: new Date().toISOString(),
          isError: true
        },
      ])
    } finally {
      setLoading(false)
    }
  }

  const handleFeedback = async (message, rating) => {
    const userQuery = messages.filter(m => m.role === 'user').slice(-1)[0]?.content || ''
    try {
      await submitFeedback(userQuery, message.content, rating)
      alert('Thank you for your feedback!')
    } catch (error) {
      console.error('Feedback error:', error)
    }
  }

  const downloadChat = () => {
    if (messages.length === 0) {
      alert('No chat to download yet')
      return
    }

    const timestamp = new Date().toLocaleString()
    let content = `OnboardIQ - Chat Session Export\n`
    content += `Downloaded: ${timestamp}\n`
    content += `Language: ${language}\n`
    content += '='.repeat(70) + '\n\n'

    messages.forEach((msg) => {
      content += `${msg.role.toUpperCase()}:\n${msg.content}\n`
      
      if (msg.citations && msg.citations.length > 0) {
        content += '\nSources Referenced:\n'
        msg.citations.forEach(cite => {
          content += `  - ${cite.file_name} - Page ${cite.page_number}\n`
          content += `    Relevance: ${(cite.relevance_score * 100).toFixed(1)}%\n`
        })
      }
      
      if (msg.confidence) {
        content += `\nConfidence Score: ${(msg.confidence * 100).toFixed(0)}%\n`
      }
      
      content += '\n' + '-'.repeat(70) + '\n\n'
    })

    content += `\nSession Statistics:\n`
    content += `Total Messages: ${messages.length}\n`
    content += `Questions Asked: ${messages.filter(m => m.role === 'user').length}\n`
    content += `Answers Provided: ${messages.filter(m => m.role === 'assistant').length}\n`

    const blob = new Blob([content], { type: 'text/plain;charset=utf-8' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `onboardiq-chat-${Date.now()}.txt`
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    URL.revokeObjectURL(url)

    alert('Chat session downloaded successfully!')
  }

  const handleSuggestion = (question) => {
    setInput(question)
  }

  return (
    <div className="chat-container">
      <div className="chat-header">
        <div>
          <h2>Ask anything about your documents</h2>
          <p style={{ fontSize: '12px', color: 'rgba(255,255,255,0.5)', margin: '5px 0 0 0' }}>
            Powered by Hybrid RAG - {messages.length} messages
          </p>
        </div>
        <button onClick={downloadChat} className="download-btn-header">
          Export Chat
        </button>
      </div>

      <div className="messages-area">
        {messages.length === 0 && (
          <div className="empty-state">
            <div className="empty-state-icon">💬</div>
            <h2>Welcome to OnboardIQ</h2>
            <p>Upload documents and start asking questions in any language</p>
            <div className="suggestions-container">
              <p style={{ fontSize: '13px', color: 'rgba(255,255,255,0.6)', marginBottom: '10px' }}>
                Try asking:
              </p>
              <div className="suggestions-grid">
                {suggestedQuestions.map((q, i) => (
                  <button
                    key={i}
                    onClick={() => handleSuggestion(q)}
                    className="suggestion-btn"
                  >
                    {q}
                  </button>
                ))}
              </div>
            </div>
          </div>
        )}

        {messages.map((message, index) => (
          <div key={index} className={`message-wrapper ${message.role}`}>
            <div className={`message-bubble ${message.role} ${message.isError ? 'error' : ''}`}>
              <div className="message-content">{message.content}</div>

              {message.citations && message.citations.length > 0 && (
                <div className="citations">
                  <div className="citations-title">Sources:</div>
                  {message.citations.map((cite, i) => (
                    <div key={i} className="citation-item">
                      <span className="citation-source">[{cite.source_id}]</span>
                      <span className="citation-text">
                        {cite.file_name} - Page {cite.page_number}
                      </span>
                    </div>
                  ))}
                </div>
              )}

              {message.role === 'assistant' && !message.isError && (
                <div className="message-feedback">
                  <button
                    onClick={() => handleFeedback(message, 'positive')}
                    className="feedback-btn"
                    title="Good response"
                  >
                    👍
                  </button>
                  <button
                    onClick={() => handleFeedback(message, 'negative')}
                    className="feedback-btn"
                    title="Bad response"
                  >
                    👎
                  </button>
                  {message.confidence && (
                    <span className="confidence-badge">
                      {(message.confidence * 100).toFixed(0)}% confidence
                    </span>
                  )}
                </div>
              )}
            </div>
          </div>
        ))}

        {loading && (
          <div className="message-wrapper assistant">
            <div className="loading-message">
              <div className="typing-dots">
                <div className="typing-dot"></div>
                <div className="typing-dot"></div>
                <div className="typing-dot"></div>
              </div>
              <span style={{ fontSize: '13px', color: 'rgba(255,255,255,0.6)', marginLeft: '10px' }}>
                Thinking...
              </span>
            </div>
          </div>
        )}

        {messages.length > 0 && !loading && (
          <div className="suggestions-footer">
            <p style={{ fontSize: '12px', color: 'rgba(255,255,255,0.5)', marginBottom: '8px' }}>
              Related questions:
            </p>
            <div className="suggestions-row">
              {suggestedQuestions.slice(0, 3).map((q, i) => (
                <button
                  key={i}
                  onClick={() => handleSend(q)}
                  className="suggestion-btn-small"
                >
                  {q}
                </button>
              ))}
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      <div className="input-area">
        <div className="language-selector">
          <label>Language:</label>
          <select value={language} onChange={(e) => setLanguage(e.target.value)}>
            <option>English</option>
            <option>Hindi</option>
            <option>Spanish</option>
            <option>French</option>
            <option>German</option>
            <option>Chinese</option>
            <option>Japanese</option>
            <option>Arabic</option>
          </select>
          <span className="language-hint">Ask in any language</span>
        </div>
        
        <div className="input-container">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={(e) => e.key === 'Enter' && !e.shiftKey && handleSend()}
            placeholder={`Ask a question in ${language}...`}
            className="input-field"
            disabled={loading || !isOnline}
          />
          <button
            onClick={() => handleSend()}
            disabled={loading || !input.trim() || !isOnline}
            className="send-btn"
          >
            {loading ? '⏳' : '➤'} Send
          </button>
        </div>
      </div>
    </div>
  )
}
