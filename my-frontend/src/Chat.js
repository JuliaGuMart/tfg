import React, { useState } from 'react';
import './index.css';
import axios from 'axios';
import ReactMarkdown from 'react-markdown';

function Chat() {
  const [message, setMessage] = useState('');
  const [messages, setMessages] = useState([]);
  const [showInstructions, setShowInstructions] = useState(true);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (showInstructions) {
      setShowInstructions(false);
    }
    const userMessage = { text: message, type: 'user' };
    setMessages([...messages, userMessage]);
    setMessage('')

    
    const loadingMessage = { text: 'Procesando...', type: 'bot' };
    setMessages((prevMessages) => [...prevMessages, loadingMessage]);

    try {
      const res = await axios.post('http://127.0.0.1:8181/chat', { message });
      const botResponse = { text: res.data.response, type: 'bot' };
     
      setMessages((prevMessages) => {
        const updatedMessages = [...prevMessages];
        updatedMessages[updatedMessages.length - 1] = botResponse;
        return updatedMessages;
      });

    } catch (error) {
      console.error('Error fetching response:', error);
    }
  };

  return (
    <div className="container">
      <div className="header">
        <h1>Taquilla billetes</h1>
      </div>
      <div className="messages">
        {showInstructions && (
          <div className="instructions">
            Bienvenido!! Soy un asistente virtual con la misión de ayudarte a reservar tus billetes de tren.
            Antes de comenzar quiero que sepas que todos los datos personales que nos aportes serán cifrados para garantizar tu seguridad.
            Disfruta de la experiencia!!
          </div>
        )}
       
        {messages.map((msg, index) => (
          <div key={index} className={msg.type === 'user' ? 'message' : 'response'}>
            {msg.type === 'user' ? 'Tú: ' : 'Bot: '}
            <ReactMarkdown>{msg.text}</ReactMarkdown>
          </div>
        ))}

      </div>
      <form onSubmit={handleSubmit} className="form">
        <input
          type="text"
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          placeholder="Type your message"
        />
        <button type="submit">Send</button>
      </form>
    </div>
  );
}

export default Chat;
