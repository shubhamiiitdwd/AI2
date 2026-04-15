// import React, { useState, useRef, useEffect } from 'react';
// import { X, Send, Bot } from 'lucide-react';
// import './ChatButton.css';

// // ─── API Config ─────────────────────────────────────────────────────────────
// const API_BASE_URL = 'http://localhost:8766/api/chat';

// // ─── Types ──────────────────────────────────────────────────────────────────
// type Role = 'bot' | 'user';

// interface ChatMessage {
//   id: string;
//   role: Role;
//   content: string;
// }

// // ─── ID helper ───────────────────────────────────────────────────────────────
// const nextId = () =>
//   `msg-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`;

// // ─── Markdown renderer ────────────────────────────────────────────────────────
// // Strips code fences / headings, renders **bold** and bullet lists as JSX.
// function renderMarkdown(text: string): React.ReactNode {
//   // Remove fenced code blocks and inline code ticks
//   const cleaned = text
//     .replace(/```[\s\S]*?```/g, '')
//     .replace(/`([^`]+)`/g, '$1')
//     // Remove heading markers (# ## ###…)
//     .replace(/^#{1,6}\s+/gm, '');

//   return cleaned.split('\n').map((line, i) => {
//     if (!line.trim()) return <div key={i} style={{ height: '6px' }} />;

//     const isBullet = /^\s*[-*+]\s/.test(line);
//     const content  = isBullet ? line.replace(/^\s*[-*+]\s/, '') : line;

//     // Bold (**text**) and italic (*text*) segments
//     const parts = content.split(/(\*\*[^*]+\*\*|\*[^*]+\*)/g).map((part, j) => {
//       if (part.startsWith('**') && part.endsWith('**')) {
//         return (
//           <strong key={j} style={{ color: '#fb8c36', fontWeight: 600, letterSpacing: '0.01em' }}>
//             {part.slice(2, -2)}
//           </strong>
//         );
//       }
//       if (part.startsWith('*') && part.endsWith('*')) {
//         return <em key={j} style={{ color: '#9fb0ca' }}>{part.slice(1, -1)}</em>;
//       }
//       return <span key={j}>{part}</span>;
//     });

//     return (
//       <div key={i} style={{ marginBottom: '6px', paddingLeft: isBullet ? '14px' : '0', lineHeight: '1.55', display: 'flex', gap: isBullet ? '6px' : '0' }}>
//         {isBullet && <span style={{ color: '#fb8c36', flexShrink: 0 }}>•</span>}
//         <span>{parts}</span>
//       </div>
//     );
//   });
// }

// // ─── Typewriter ───────────────────────────────────────────────────────────────
// interface TypewriterProps {
//   text: string;
//   speed?: number;
//   onComplete: () => void;
// }

// const Typewriter = ({ text, speed = 18, onComplete }: TypewriterProps) => {
//   const [displayed, setDisplayed] = useState('');
//   // Keep onComplete stable across renders without adding it to the effect deps
//   const onCompleteRef = useRef(onComplete);
//   onCompleteRef.current = onComplete;

//   useEffect(() => {
//     let index = 0;
//     setDisplayed('');
//     const id = setInterval(() => {
//       index += 1;
//       setDisplayed(text.slice(0, index));
//       if (index >= text.length) {
//         clearInterval(id);
//         onCompleteRef.current();
//       }
//     }, speed);
//     return () => clearInterval(id);
//   }, [text, speed]);

//   return <>{renderMarkdown(displayed)}</>;
// };

// // ─── Initial messages (module-level so IDs are stable) ───────────────────────
// const INIT_MSGS: ChatMessage[] = [
//   { id: nextId(), role: 'bot', content: 'Hi there! 👋 Welcome to AI Sphere.' },
//   { id: nextId(), role: 'bot', content: 'How can I help you navigate the data and model exchange today?' },
// ];

// // ─── ChatButton ───────────────────────────────────────────────────────────────
// const ChatButton = () => {
//   const [isOpen,      setIsOpen]      = useState(false);
//   const [isSpinning,  setIsSpinning]  = useState(false);
//   const [inputMessage, setInputMessage] = useState('');
//   const [isLoading,   setIsLoading]   = useState(false);
//   const [messages,    setMessages]    = useState<ChatMessage[]>(INIT_MSGS);
//   // IDs of bot messages that are still being typed
//   const [typingIds,   setTypingIds]   = useState<Set<string>>(
//     () => new Set(INIT_MSGS.map(m => m.id))
//   );

//   const messagesEndRef = useRef<HTMLDivElement>(null);

//   const markDone = (id: string) =>
//     setTypingIds(prev => { const s = new Set(prev); s.delete(id); return s; });

//   const toggleChat = () => {
//     setIsOpen(v => !v);
//     setIsSpinning(true);
//   };

//   useEffect(() => {
//     messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
//   }, [messages, isOpen]);

//   const handleSend = async (e: React.FormEvent<HTMLFormElement>): Promise<void> => {
//     e.preventDefault();
//     if (!inputMessage.trim() || isLoading) return;

//     const userMsg: ChatMessage = { id: nextId(), role: 'user', content: inputMessage.trim() };
//     setInputMessage('');
//     const history = [...messages, userMsg];
//     setMessages(history);
//     setIsLoading(true);

//     try {
//       const response = await fetch(API_BASE_URL, {
//         method: 'POST',
//         headers: { 'Content-Type': 'application/json' },
//         body: JSON.stringify({ messages: history.map(({ role, content }) => ({ role, content })) }),
//       });

//       if (!response.ok) throw new Error('Network response was not ok');
//       const data = await response.json();
//       const botMsg: ChatMessage = { id: nextId(), role: 'bot', content: data.reply };
//       setMessages(prev => [...prev, botMsg]);
//       setTypingIds(prev => new Set(prev).add(botMsg.id));
//     } catch (error) {
//       console.error('Chat Error:', error);
//       const errMsg: ChatMessage = {
//         id: nextId(),
//         role: 'bot',
//         content: "Sorry, I'm having trouble connecting to the server right now. 😔",
//       };
//       setMessages(prev => [...prev, errMsg]);
//       setTypingIds(prev => new Set(prev).add(errMsg.id));
//     } finally {
//       setIsLoading(false);
//     }
//   };

//   return (
//     <>
//       <button
//         className={`chat-fab-button ${isOpen ? 'active' : ''}`}
//         onClick={toggleChat}
//         aria-label={isOpen ? 'Close chat' : 'Open chat'}
//       >
//         {isOpen ? <X size={24} className="chat-icon" /> : (
//           <span
//             className={`chat-gears-icon${isSpinning ? ' is-spinning' : ''}`}
//             aria-hidden="true"
//             onAnimationEnd={() => setIsSpinning(false)}
//           >
//             <svg className="chat-star chat-star--cw"  width="22" height="22" viewBox="0 0 24 24" fill="currentColor"><path d="M12 2l2.4 7.4H22l-6.2 4.5 2.4 7.4L12 17l-6.2 4.3 2.4-7.4L2 9.4h7.6z"/></svg>
//             <svg className="chat-star chat-star--ccw" width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><path d="M12 2l2.4 7.4H22l-6.2 4.5 2.4 7.4L12 17l-6.2 4.3 2.4-7.4L2 9.4h7.6z"/></svg>
//           </span>
//         )}
//       </button>

//       {isOpen && (
//         <div className="chat-window view-animate-in">
//           <div className="chat-header">
//             <div className="chat-header-info">
//               <Bot size={20} />
//               <div className="chat-title">
//                 <h4>AI Sphere Assistant</h4>
//                 <span>Online</span>
//               </div>
//             </div>
//             <button className="chat-close" onClick={toggleChat}>
//               <X size={18} />
//             </button>
//           </div>

//           <div className="chat-messages">
//             {messages.map(msg => (
//               <div key={msg.id} className={`chat-message ${msg.role}`}>
//                 {msg.role === 'bot' ? (
//                   typingIds.has(msg.id)
//                     ? <Typewriter text={msg.content} speed={18} onComplete={() => markDone(msg.id)} />
//                     : renderMarkdown(msg.content)
//                 ) : (
//                   <p style={{ margin: 0 }}>{msg.content}</p>
//                 )}
//               </div>
//             ))}

//             {isLoading && (
//               <div className="chat-message bot" style={{ display: 'flex', alignItems: 'center', gap: '5px' }}>
//                 <span style={{ color: '#7e8ca8', fontStyle: 'italic', fontSize: '13px' }}>
//                   AI is thinking...
//                 </span>
//               </div>
//             )}

//             <div ref={messagesEndRef} />
//           </div>

//           <form className="chat-input-area" onSubmit={handleSend}>
//             <input
//               type="text"
//               placeholder="Type your question here..."
//               value={inputMessage}
//               onChange={(e: React.ChangeEvent<HTMLInputElement>) => setInputMessage(e.target.value)}
//               className="chat-input"
//               disabled={isLoading}
//             />
//             <button type="submit" className="chat-submit" disabled={!inputMessage.trim() || isLoading}>
//               <Send size={18} />
//             </button>
//           </form>
//         </div>
//       )}
//     </>
//   );
// };

// export default ChatButton;
import React, { useState, useRef, useEffect } from 'react';
import { X, Send, Bot } from 'lucide-react';
import './ChatButton.css';

// ─── API Config ─────────────────────────────────────────────────────────────
const API_BASE_URL = 'http://localhost:8766/api/chat';

// ─── Types ──────────────────────────────────────────────────────────────────
type Role = 'bot' | 'user';

interface ChatMessage {
  id: string;
  role: Role;
  content: string;
}

// ─── ID helper ───────────────────────────────────────────────────────────────
const nextId = () =>
  `msg-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`;

// ─── Markdown renderer ────────────────────────────────────────────────────────
// Strips code fences / headings, renders **bold** and bullet lists as JSX.
function renderMarkdown(text: string): React.ReactNode {
  // Remove fenced code blocks and inline code ticks
  const cleaned = text
    .replace(/```[\s\S]*?```/g, '')
    .replace(/`([^`]+)`/g, '$1')
    // Remove heading markers (# ## ###…)
    .replace(/^#{1,6}\s+/gm, '');

  return cleaned.split('\n').map((line, i) => {
    if (!line.trim()) return <div key={i} style={{ height: '6px' }} />;
    if (line.trim() === '---') return <hr key={i} style={{ border: 'none', borderTop: '1px solid #e2e8f0', margin: '12px 0' }} />;

    const isBullet = /^\s*[-*+]\s/.test(line);
    const content  = isBullet ? line.replace(/^\s*[-*+]\s/, '') : line;

    // Bold (**text**) and italic (*text*) segments
    const parts = content.split(/(\*\*[^*]+\*\*|\*[^*]+\*)/g).map((part, j) => {
      if (part.startsWith('**') && part.endsWith('**')) {
        return (
          <strong key={j} style={{ color: '#fb8c36', fontWeight: 600, letterSpacing: '0.01em' }}>
            {part.slice(2, -2)}
          </strong>
        );
      }
      if (part.startsWith('*') && part.endsWith('*')) {
        return <em key={j} style={{ color: '#9fb0ca' }}>{part.slice(1, -1)}</em>;
      }
      return <span key={j}>{part}</span>;
    });

    return (
      <div key={i} style={{ marginBottom: '6px', paddingLeft: isBullet ? '14px' : '0', lineHeight: '1.55', display: 'flex', gap: isBullet ? '6px' : '0' }}>
        {isBullet && <span style={{ color: '#fb8c36', flexShrink: 0 }}>•</span>}
        <span>{parts}</span>
      </div>
    );
  });
}

// ─── Typewriter (Only used for Initial Messages now) ──────────────────────────
interface TypewriterProps {
  text: string;
  speed?: number;
  onComplete: () => void;
}

const Typewriter = ({ text, speed = 18, onComplete }: TypewriterProps) => {
  const [displayed, setDisplayed] = useState('');
  const onCompleteRef = useRef(onComplete);
  onCompleteRef.current = onComplete;

  useEffect(() => {
    let index = 0;
    setDisplayed('');
    const id = setInterval(() => {
      index += 1;
      setDisplayed(text.slice(0, index));
      if (index >= text.length) {
        clearInterval(id);
        onCompleteRef.current();
      }
    }, speed);
    return () => clearInterval(id);
  }, [text, speed]);

  return <>{renderMarkdown(displayed)}</>;
};

// ─── Initial messages (module-level so IDs are stable) ───────────────────────
const INIT_MSGS: ChatMessage[] = [
  { id: nextId(), role: 'bot', content: 'Hi there! 👋 Welcome to AI Sphere.' },
  
];

// ─── ChatButton ───────────────────────────────────────────────────────────────
const ChatButton = () => {
  const [isOpen,      setIsOpen]      = useState(false);
  const [isSpinning,  setIsSpinning]  = useState(false);
  const [inputMessage, setInputMessage] = useState('');
  const [isLoading,   setIsLoading]   = useState(false);
  const [messages,    setMessages]    = useState<ChatMessage[]>(INIT_MSGS);
  
  // IDs of bot messages that are still being typed (only used for INIT_MSGS now)
  const [typingIds,   setTypingIds]   = useState<Set<string>>(
    () => new Set(INIT_MSGS.map(m => m.id))
  );

  const messagesEndRef = useRef<HTMLDivElement>(null);

  const markDone = (id: string) =>
    setTypingIds(prev => { const s = new Set(prev); s.delete(id); return s; });

  const toggleChat = () => {
    setIsOpen(v => !v);
    setIsSpinning(true);
  };

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isOpen]);

  const handleSend = async (e: React.FormEvent<HTMLFormElement>): Promise<void> => {
    e.preventDefault();
    if (!inputMessage.trim() || isLoading) return;

    // 1. Add user message
    const userMsg: ChatMessage = { id: nextId(), role: 'user', content: inputMessage.trim() };
    setInputMessage('');
    const history = [...messages, userMsg];
    setMessages(history);
    
    // 2. Add an empty bot message placeholder immediately
    const botMsgId = nextId();
    setMessages(prev => [...prev, { id: botMsgId, role: 'bot', content: '' }]);
    
    setIsLoading(true);

    try {
      const response = await fetch(API_BASE_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        // Send history up to the user's message (excluding the empty bot placeholder)
        body: JSON.stringify({ messages: history.map(({ role, content }) => ({ role, content })) }),
      });

      if (!response.ok) throw new Error('Network response was not ok');
      
      // 3. Turn off the "AI is thinking..." spinner the moment data arrives
      setIsLoading(false);

      // 4. Stream reading logic
      if (!response.body) throw new Error('No response body');
      const reader = response.body.getReader();
      const decoder = new TextDecoder('utf-8');
      let done = false;

      while (!done) {
        const { value, done: readerDone } = await reader.read();
        done = readerDone;
        if (value) {
          const chunkValue = decoder.decode(value, { stream: true });
          
          // Incrementally update the specific bot message content
          setMessages(prev => 
            prev.map(msg => 
              msg.id === botMsgId 
                ? { ...msg, content: msg.content + chunkValue } 
                : msg
            )
          );
        }
      }
    } catch (error) {
      console.error('Chat Error:', error);
      setIsLoading(false);
      const errMsgId = nextId();
      setMessages(prev => [...prev, {
        id: errMsgId,
        role: 'bot',
        content: "Sorry, I'm having trouble connecting to the server right now. 😔",
      }]);
      // Use typewriter for the error message
      setTypingIds(prev => new Set(prev).add(errMsgId));
    }
  };

  return (
    <>
      <button
        className={`chat-fab-button ${isOpen ? 'active' : ''}`}
        onClick={toggleChat}
        aria-label={isOpen ? 'Close chat' : 'Open chat'}
      >
        {isOpen ? <X size={24} className="chat-icon" /> : (
          <span
            className={`chat-gears-icon${isSpinning ? ' is-spinning' : ''}`}
            aria-hidden="true"
            onAnimationEnd={() => setIsSpinning(false)}
          >
            <svg className="chat-star chat-star--cw"  width="22" height="22" viewBox="0 0 24 24" fill="currentColor"><path d="M12 2l2.4 7.4H22l-6.2 4.5 2.4 7.4L12 17l-6.2 4.3 2.4-7.4L2 9.4h7.6z"/></svg>
            <svg className="chat-star chat-star--ccw" width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><path d="M12 2l2.4 7.4H22l-6.2 4.5 2.4 7.4L12 17l-6.2 4.3 2.4-7.4L2 9.4h7.6z"/></svg>
          </span>
        )}
      </button>

      {isOpen && (
        <div className="chat-window view-animate-in">
          <div className="chat-header">
            <div className="chat-header-info">
              <Bot size={20} />
              <div className="chat-title">
                <h4>AI Sphere Assistant</h4>
                <span>Online</span>
              </div>
            </div>
            <button className="chat-close" onClick={toggleChat}>
              <X size={18} />
            </button>
          </div>

          <div className="chat-messages">
            {messages.map(msg => (
              <div key={msg.id} className={`chat-message ${msg.role}`}>
                {msg.role === 'bot' ? (
                  // If the message is currently empty (waiting for first chunk) -> Show dots
                  msg.content === '' ? (
                    <div className="typing-indicator">
                      <span></span><span></span><span></span>
                    </div>
                  ) : typingIds.has(msg.id) ? (
                    <Typewriter text={msg.content} speed={18} onComplete={() => markDone(msg.id)} />
                  ) : (
                    renderMarkdown(msg.content)
                  )
                ) : (
                  <p style={{ margin: 0 }}>{msg.content}</p>
                )}
              </div>
            ))}

           

            <div ref={messagesEndRef} />
          </div>

          <form className="chat-input-area" onSubmit={handleSend}>
            <input
              type="text"
              placeholder="Type your question here..."
              value={inputMessage}
              onChange={(e: React.ChangeEvent<HTMLInputElement>) => setInputMessage(e.target.value)}
              className="chat-input"
              disabled={isLoading}
            />
            <button type="submit" className="chat-submit" disabled={!inputMessage.trim() || isLoading}>
              <Send size={18} />
            </button>
          </form>
        </div>
      )}
    </>
  );
};

export default ChatButton;