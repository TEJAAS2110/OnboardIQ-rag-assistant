import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const uploadDocument = async (file) => {
  const formData = new FormData();
  formData.append('file', file);
  
  const response = await axios.post(`${API_BASE_URL}/documents/upload`, formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });
  
  return response.data;
};

export const listDocuments = async () => {
  const response = await api.get('/documents/list');
  return response.data;
};

export const sendQuery = async (query, conversationHistory = []) => {
  const response = await api.post('/chat/query', {
    query,
    conversation_history: conversationHistory,
    top_k: 5,
  });
  return response.data;
};

export const submitFeedback = async (query, answer, rating) => {
  const response = await api.post('/chat/feedback', {
    query,
    answer,
    rating,
  });
  return response.data;
};

export default api;