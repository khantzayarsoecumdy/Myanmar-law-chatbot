
import React from 'react';
import { useEffect, useRef, useState } from 'react';
import { ArrowUp, Plus, Scale, FileText, PanelLeftClose, Menu, Square, BookOpen } from 'lucide-react';

const suggestions = [
  { title: 'Condominium Law', prompt: 'According to the uploaded Condominium Law, what does “Condominium” mean?' },
  { title: 'လယ်ယာမြေ', prompt: 'လယ်ယာမြေလုပ်ပိုင်ခွင့်ဆိုတာ ဘာလဲ။ ထည့်ထားတဲ့ ဥပဒေစာရွက်စာတမ်းတွေကိုပဲ အခြေခံဖြေပါ။' },
  { title: 'Check document coverage', prompt: 'What topics are covered by the uploaded documents? Answer only from the indexed documents.' },
];

function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [language, setLanguage] = useState('auto');
  const [busy, setBusy] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [error, setError] = useState('');
  const [apiStatus, setApiStatus] = useState('checking');
  const [topK, setTopK] = useState(null);
  const bottomRef = useRef(null);
  const abortRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, busy]);

  useEffect(() => {
    let alive = true;
    const checkApi = async () => {
      try {
        const response = await fetch('http://127.0.0.1:8000/health');
        if (!response.ok) throw new Error('API unavailable');
        const data = await response.json();
        if (alive) {
          setApiStatus(data.indexed ? 'ready' : 'not-indexed');
          setTopK(Number.isInteger(data.top_k) ? data.top_k : null);
        }
      } catch {
        if (alive) setApiStatus('offline');
      }
    };
    checkApi();
    const timer = window.setInterval(checkApi, 10000);
    return () => { alive = false; window.clearInterval(timer); };
  }, []);

  async function send(text = input) {
    const question = text.trim();
    if (!question || busy) return;
    setInput('');
    setError('');
    setMessages(old => [...old, { role: 'user', content: question }]);
    setBusy(true);
    const controller = new AbortController();
    abortRef.current = controller;
    try {
      const res = await fetch('http://127.0.0.1:8000/api/ask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question, language }),
        signal: controller.signal,
      });
      let data;
      try {
        data = await res.json();
      } catch {
        throw new Error('The API returned an invalid response. Check that the backend is running correctly.');
      }
      if (!res.ok) throw new Error(data.detail || 'Something went wrong.');
      if (typeof data.answer !== 'string' || !data.answer.trim()) {
        throw new Error('The API response did not include an answer. Check the backend response in /docs.');
      }
      setMessages(old => [...old, {
        role: 'assistant',
        content: data.answer.trim(),
        sources: Array.isArray(data.sources) ? data.sources : [],
      }]);
    } catch (e) {
      if (e.name === 'AbortError') {
        setMessages(old => [...old, { role: 'assistant', content: 'Response cancelled.' }]);
      } else {
        setError(e.message || 'Cannot connect to the backend. Check that the API is running and a model provider is available.');
      }
    } finally {
      setBusy(false);
      abortRef.current = null;
    }
  }

  function cancel() { abortRef.current?.abort(); }

  function newChat() { cancel(); setMessages([]); setInput(''); setError(''); }

  return (
    <div className="app-shell">
      {sidebarOpen && <aside className="sidebar">
        <div className="side-top">
          <div className="brand"><div className="brand-icon"><Scale size={19}/></div><span>Law Assistant</span></div>
          <button className="icon-btn" title="Hide sidebar" onClick={() => setSidebarOpen(false)}><PanelLeftClose size={18}/></button>
        </div>
        <button className="new-chat" onClick={newChat}><Plus size={17}/> New chat <span>Ctrl K</span></button>
        <div className="side-label">YOUR KNOWLEDGE BASE</div>
        <div className="side-info"><BookOpen size={17}/><div><strong>Uploaded law documents</strong><small>{topK ? `Top K: ${topK} passages per search` : 'Local document search'}</small></div></div>
        <div className="sidebar-bottom"><div className="privacy-dot"/><span>OpenRouter + Ollama</span><small>Local fallback on limits or outages</small></div>
      </aside>}
      <main className="main">
        <header className="topbar">
          <div className="top-left">
            {!sidebarOpen && <button className="icon-btn" onClick={() => setSidebarOpen(true)}><Menu size={19}/></button>}
            <span className="mobile-brand"><Scale size={18}/> Law Assistant</span>
            <span className="model-label">Document Chat <span className="chevron">⌄</span></span>
          </div>
          <div className={`local-pill status-${apiStatus}`} title={apiStatus === 'ready' ? 'Backend and document index are ready' : apiStatus === 'not-indexed' ? 'Backend is running, but documents have not been indexed' : apiStatus === 'offline' ? 'Start the backend API in a separate terminal' : 'Checking backend status'}><span/> {apiStatus === 'ready' ? 'AI ready' : apiStatus === 'not-indexed' ? 'Index documents' : apiStatus === 'offline' ? 'API offline' : 'Checking…'}</div>
        </header>
        <section className={`conversation ${messages.length ? 'has-messages' : ''}`}>
          {messages.length === 0 ? <div className="welcome">
            <div className="welcome-mark"><Scale size={25}/></div>
            <h1>What would you like to know?</h1>
            <p>Ask questions about your uploaded property and land-law documents.</p>
            <div className="suggestions">
              {suggestions.map((s, i) => <button key={i} className="suggestion" onClick={() => send(s.prompt)}><div><strong>{s.title}</strong><span>{s.prompt}</span></div><ArrowUp size={16}/></button>)}
            </div>
          </div> : <div className="messages">
            {messages.map((m, i) => <article key={i} className={`message ${m.role}`}>
              {m.role === 'assistant' && <div className="avatar"><Scale size={16}/></div>}
              <div className="message-body">
                {m.role === 'user' ? <div className="user-bubble">{m.content}</div> : <div className="answer-text">{m.content}</div>}
                {m.sources?.length > 0 && <div className="sources"><div className="sources-heading"><FileText size={14}/> Sources from your documents</div><div className="source-list">{m.sources.map((s, j) => <div key={j} className="source-chip"><span>{s.document} · chunk {s.chunk}</span>{s.quote && <details><summary>See more</summary><blockquote>{s.quote}</blockquote></details>}</div>)}</div></div>}
              </div>
            </article>)}
            {busy && <div className="message assistant"><div className="avatar"><Scale size={16}/></div><div className="thinking"><span/><span/><span/> <em>Searching documents and preparing answer…</em></div></div>}
            <div ref={bottomRef}/>
          </div>}
        </section>
        <div className="composer-wrap">
          {error && <div className="error-banner">{error}</div>}
          <div className="composer">
            <textarea value={input} onChange={e => setInput(e.target.value)} onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); } }} placeholder="Message Law Assistant…" rows={1} aria-label="Ask a question" />
            <div className="composer-bottom">
              <div className="composer-hint"><Plus size={17}/><span>Answers quote the original document text</span></div>
              <label className="language-control">Fallback in
                <select aria-label="Answer language" value={language} onChange={e => setLanguage(e.target.value)}>
                  <option value="auto">Auto</option>
                  <option value="my">မြန်မာ</option>
                  <option value="en">English</option>
                </select>
              </label>
              {busy ? <button className="send-btn stop-btn" onClick={cancel} title="Cancel response"><Square size={14} fill="currentColor"/></button> : <button className="send-btn" onClick={() => send()} disabled={!input.trim()} title="Send message"><ArrowUp size={18}/></button>}
            </div>
          </div>
          <div className="disclaimer">AI can make mistakes. Verify important legal matters with a qualified professional.</div>
        </div>
      </main>
    </div>
  );
}

export default App;
