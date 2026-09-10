import { createRoot } from 'react-dom/client';
import { useEffect, useMemo, useRef, useState, type ReactNode } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { listen } from '@tauri-apps/api/event';
import { open } from '@tauri-apps/plugin-dialog';
import { readTextFile } from '@tauri-apps/plugin-fs';
import { register } from '@tauri-apps/plugin-global-shortcut';
import {
  Activity, Archive, Bot, Calculator, Check, ChevronDown, CircleHelp, Clipboard,
  Command, Cpu, Download, FileCode2, FileText, FolderOpen, Gauge, History, KeyRound,
  Library, MessageSquarePlus, Moon, PanelLeft, Paperclip, Pencil, Pin, Plus, Search,
  Send, Settings, Sparkles, Sun, Trash2, Upload, UserRound, UsersRound, WandSparkles,
  X, Zap
} from 'lucide-react';
import './styles.css';

type Page = 'Chat' | 'Models' | 'Personas' | 'Lab' | 'Playground' | 'Knowledge' | 'Memory' | 'Settings';
type Message = { id: string; role: 'user' | 'assistant'; content: string; time: string; stats?: string };
type Chat = { id: string; title: string; pinned: boolean; messages: Message[] };
type Model = { id: string; name: string; path: string; size: string; bytes: number; architecture: string; quantization: string; context: number; status: 'ready' | 'unsupported' | 'missing'; parameterCount?: string; tokenizer: 'embedded' | 'sidecar' | 'missing' };
type Persona = { id: string; name: string; description: string; prompt: string; temperature: number };
type Stats = { cpu_usage: number; memory_used_mb: number; memory_total_mb: number; process_memory_mb: number; gpu_available: boolean; gpu_note: string };
type Gen = { text: string; tokens: number; seconds: number; tokens_per_sec: number; cancelled: boolean };
type Doc = { id: string; name: string; content: string; size: string };
type Memory = { id: string; text: string; scope: 'global' | 'chat'; created: string };

const personas0: Persona[] = [
  { id: 'balanced', name: 'Balanced', description: 'Clear general-purpose assistant', prompt: 'Be clear, useful, concise, and honest.', temperature: .7 },
  { id: 'coder', name: 'Coder', description: 'Programming-focused', prompt: 'Act as a pragmatic senior programmer. Prefer working code and explain bugs clearly.', temperature: .35 },
  { id: 'tutor', name: 'Tutor', description: 'Step-by-step explanations', prompt: 'Teach patiently with simple examples before adding depth.', temperature: .55 },
  { id: 'creative', name: 'Creative', description: 'Writing and ideation', prompt: 'Be imaginative, specific, polished, and distinctive.', temperature: .9 },
  { id: 'researcher', name: 'Researcher', description: 'Evidence-focused analysis', prompt: 'Separate facts, assumptions, and uncertainty. Never invent sources.', temperature: .35 },
  { id: 'companion', name: 'Companion', description: 'Natural, empathetic conversation', prompt: 'Be warm and emotionally aware while staying grounded and honest.', temperature: .78 }
];
const welcome: Chat = { id: 'welcome', title: 'Welcome to Synth Studio', pinned: true, messages: [{ id: 'welcome', role: 'assistant', content: 'Welcome to **Synth Studio**.\n\nLoad a compatible GGUF model in **Models**. Your conversations and local knowledge stay on this device.', time: now() }] };
const load = <T,>(key: string, fallback: T): T => { try { return JSON.parse(localStorage.getItem(key) || '') as T; } catch { return fallback; } };
const save = (key: string, value: unknown) => localStorage.setItem(key, JSON.stringify(value));
const uid = () => `${Date.now()}-${Math.random().toString(16).slice(2)}`;
const now = () => new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
const size = (n: number) => n >= 1024 ** 3 ? `${(n / 1024 ** 3).toFixed(2)} GB` : n >= 1024 ** 2 ? `${(n / 1024 ** 2).toFixed(1)} MB` : `${Math.max(1, Math.round(n / 1024))} KB`;
const escape = (s: string) => s.replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;');
const md = (s: string) => escape(s).replace(/```([\s\S]*?)```/g, '<pre>$1</pre>').replace(/`([^`]+)`/g, '<code>$1</code>').replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>').replace(/^### (.*)$/gm, '<h4>$1</h4>').replace(/^## (.*)$/gm, '<h3>$1</h3>').replace(/^# (.*)$/gm, '<h2>$1</h2>').replace(/\n/g, '<br/>');
const score = (query: string, text: string) => { const q = new Set(query.toLowerCase().split(/\W+/).filter(Boolean)); return text.toLowerCase().split(/\W+/).filter(Boolean).reduce((n, w) => n + (q.has(w) ? 1 : 0), 0); };

function App() {
  const [page, setPage] = useState<Page>('Chat');
  const [dark, setDark] = useState(load('synth.dark', false));
  const [collapsed, setCollapsed] = useState(false);
  const [query, setQuery] = useState('');
  const [input, setInput] = useState('');
  const [models, setModels] = useState<Model[]>(load('synth.models', []));
  const [selected, setSelected] = useState(load('synth.selected', ''));
  const [people, setPeople] = useState<Persona[]>(load('synth.personas', personas0));
  const [personaId, setPersonaId] = useState(load('synth.persona', 'balanced'));
  const [chats, setChats] = useState<Chat[]>(load('synth.chats', [welcome]));
  const [activeId, setActiveId] = useState(load('synth.active', 'welcome'));
  const [busy, setBusy] = useState(false);
  const [notice, setNotice] = useState('');
  const [stats, setStats] = useState<Stats | null>(null);
  const [palette, setPalette] = useState(false);
  const [quick, setQuick] = useState(false);
  const [editId, setEditId] = useState<string | null>(null);
  const [editText, setEditText] = useState('');
  const [knowledge, setKnowledge] = useState<Doc[]>(load('synth.knowledge', []));
  const [memories, setMemories] = useState<Memory[]>(load('synth.memories', []));
  const [labA, setLabA] = useState('');
  const [labB, setLabB] = useState('');
  const [labPrompt, setLabPrompt] = useState('Explain local AI in simple terms.');
  const [labResults, setLabResults] = useState<{ name: string; result?: Gen; error?: string }[]>([]);
  const [playground, setPlayground] = useState({ prompt: 'Explain recursion with a simple example.', system: 'Be concise and clear.', temperature: .7, topP: .9, topK: 40, maxTokens: 256, seed: 42 });
  const [toolInput, setToolInput] = useState('');
  const [toolResult, setToolResult] = useState('');
  const req = useRef('');

  const active = chats.find(c => c.id === activeId) || chats[0];
  const current = models.find(m => m.id === selected);
  const person = people.find(p => p.id === personaId) || people[0];
  const visible = useMemo(() => chats.filter(c => c.title.toLowerCase().includes(query.toLowerCase())), [chats, query]);
  const recommended = useMemo(() => {
    if (!stats) return 'Load a model to see a recommendation.';
    if (stats.memory_total_mb >= 16000) return 'A 1.5–3B Q4 model is a sensible lightweight starting point.';
    if (stats.memory_total_mb >= 8000) return 'A 0.8–1.5B Q4 model is the safest lightweight range.';
    return 'Stay near 0.5–1B Q4 models to preserve memory headroom.';
  }, [stats]);

  useEffect(() => save('synth.dark', dark), [dark]);
  useEffect(() => save('synth.models', models), [models]);
  useEffect(() => save('synth.selected', selected), [selected]);
  useEffect(() => save('synth.personas', people), [people]);
  useEffect(() => save('synth.persona', personaId), [personaId]);
  useEffect(() => { save('synth.chats', chats); save('synth.active', activeId); }, [chats, activeId]);
  useEffect(() => save('synth.knowledge', knowledge), [knowledge]);
  useEffect(() => save('synth.memories', memories), [memories]);
  useEffect(() => { const refresh = async () => { try { setStats(await invoke<Stats>('system_stats')); } catch {} }; void refresh(); const timer = setInterval(refresh, 4000); return () => clearInterval(timer); }, []);
  useEffect(() => { let off: (() => void) | undefined; listen<{ requestId: string; text: string }>('generation-token', event => { if (event.payload.requestId !== req.current) return; setChats(cs => cs.map(c => c.id === activeId ? { ...c, messages: c.messages.map(m => m.id === req.current ? { ...m, content: event.payload.text } : m) } : c)); }).then(unlisten => off = unlisten); return () => off?.(); }, [activeId]);
  useEffect(() => { let off: (() => void) | undefined; listen<string>('synth-quick', () => setQuick(true)).then(unlisten => off = unlisten); return () => off?.(); }, []);
  useEffect(() => { void register('CommandOrControl+Shift+Space').catch(() => undefined); }, []);
  useEffect(() => { const f = (e: KeyboardEvent) => { if ((e.ctrlKey || e.metaKey) && e.code === 'KeyK') { e.preventDefault(); setPalette(true); } if ((e.ctrlKey || e.metaKey) && e.code === 'KeyN') { e.preventDefault(); newChat(); } if (e.key === 'Escape') { setPalette(false); setQuick(false); } }; addEventListener('keydown', f); return () => removeEventListener('keydown', f); });

  const toast = (text: string) => { setNotice(text); window.setTimeout(() => setNotice(''), 2400); };
  const patch = (fn: (chat: Chat) => Chat) => setChats(cs => cs.map(c => c.id === activeId ? fn(c) : c));
  const newChat = () => { const id = uid(); setChats(cs => [{ id, title: 'New conversation', pinned: false, messages: [] }, ...cs]); setActiveId(id); setPage('Chat'); };
  const selectPage = (next: Page) => { setPage(next); setQuick(false); };

  async function importModel() {
    try {
      const picked = await open({ multiple: false, filters: [{ name: 'GGUF model', extensions: ['gguf'] }] });
      if (!picked || Array.isArray(picked)) return;
      const model = await invoke<Model>('inspect_model', { path: String(picked) });
      setModels(list => [model, ...list.filter(m => m.path !== model.path)]);
      if (model.status === 'ready') setSelected(model.id);
      toast(model.status === 'ready' ? `${model.name} is ready` : `${model.name}: ${model.status}`);
    } catch (error) { toast(String(error)); }
  }

  async function send(textOverride?: string) {
    const text = (textOverride ?? input).trim();
    if (!text || busy) return;
    if (!current) { toast('Import a supported GGUF model first'); selectPage('Models'); return; }
    const requestId = uid(); req.current = requestId; setInput(''); setBusy(true);
    const history = active.messages.slice(-8).map(m => `${m.role === 'user' ? 'User' : 'Assistant'}: ${m.content}`).join('\n\n');
    const relevantDocs = knowledge.map(doc => ({ doc, score: score(text, doc.content) })).filter(x => x.score > 0).sort((a, b) => b.score - a.score).slice(0, 3);
    const relevantMemory = memories.slice(0, 8).map(m => `- ${m.text}`).join('\n');
    const sources = relevantDocs.length ? `\n\nLocal knowledge:\n${relevantDocs.map(x => `### ${x.doc.name}\n${x.doc.content.slice(0, 3500)}`).join('\n\n')}` : '';
    const prompt = `${person.prompt}\n\n${relevantMemory ? `Local memory:\n${relevantMemory}\n\n` : ''}${history ? history + '\n\n' : ''}User: ${text}\nAssistant:${sources}`;
    patch(c => ({ ...c, title: c.messages.length ? c.title : text.slice(0, 42), messages: [...c.messages, { id: uid(), role: 'user', content: text, time: now() }, { id: requestId, role: 'assistant', content: '', time: now() }] }));
    try {
      const result = await invoke<Gen>('start_generation', { request: { requestId, modelPath: current.path, prompt, maxTokens: 768, temperature: person.temperature, topP: .9, topK: 40, repeatPenalty: 1.1, contextLength: Math.min(current.context || 4096, 8192), seed: 42 } });
      patch(c => ({ ...c, messages: c.messages.map(m => m.id === requestId ? { ...m, content: result.text, stats: `${result.tokens} tokens · ${result.tokens_per_sec.toFixed(1)} tok/s · ${result.seconds.toFixed(1)}s` } : m) }));
      if (result.cancelled) toast('Generation stopped');
    } catch (error) { patch(c => ({ ...c, messages: c.messages.map(m => m.id === requestId ? { ...m, content: `**Generation failed**\n\n${String(error)}`, stats: 'Error' } : m) })); }
    finally { setBusy(false); }
  }

  async function attachFile() {
    try {
      const picked = await open({ multiple: false, filters: [{ name: 'Text/code', extensions: ['txt', 'md', 'json', 'py', 'js', 'ts', 'tsx', 'jsx', 'html', 'css', 'rs', 'toml'] }] });
      if (!picked || Array.isArray(picked)) return;
      const path = String(picked); const name = path.split(/[\\/]/).pop() || 'file';
      const content = await readTextFile(path);
      const byteSize = new Blob([content]).size;
      if (byteSize > 5 * 1024 * 1024) { toast('File is too large for lightweight local context (5 MB limit).'); return; }
      patch(c => ({ ...c, messages: [...c.messages, { id: uid(), role: 'user', content: `Attached **${name}** locally for analysis.\n\n\`\`\`text\n${content.slice(0, 12000)}\n\`\`\``, time: now() }] }));
      toast(`${name} attached locally`);
    } catch (error) { toast(String(error)); }
  }

  async function addKnowledge() {
    try {
      const picked = await open({ multiple: true, filters: [{ name: 'Text', extensions: ['txt', 'md', 'json', 'py', 'js', 'ts', 'html', 'css', 'rs'] }] });
      const files = Array.isArray(picked) ? picked : picked ? [picked] : [];
      for (const file of files) { const path = String(file); const name = path.split(/[\\/]/).pop() || 'Document'; const content = await readTextFile(path); if (!content.trim()) continue; setKnowledge(list => [...list, { id: uid(), name, content, size: size(new Blob([content]).size) }]); }
      if (files.length) toast('Knowledge added locally.');
    } catch (error) { toast(String(error)); }
  }

  async function runLab() {
    const ids = [labA, labB].filter((x, i, a) => x && a.indexOf(x) === i);
    const picks = ids.map(id => models.find(m => m.id === id)).filter((m): m is Model => !!m && m.status === 'ready');
    if (!picks.length) { toast('Choose at least one installed model.'); return; }
    setLabResults([]);
    for (const model of picks) {
      try { const result = await invoke<Gen>('start_generation', { request: { requestId: uid(), modelPath: model.path, prompt: labPrompt, maxTokens: 256, temperature: .4, topP: .9, topK: 40, repeatPenalty: 1.1, contextLength: Math.min(model.context || 4096, 4096), seed: 42 } }); setLabResults(list => [...list, { name: model.name, result }]); }
      catch (error) { setLabResults(list => [...list, { name: model.name, error: String(error) }]); }
    }
  }

  async function runPlayground() {
    if (!current) { toast('Load a model first.'); return; }
    try {
      const result = await invoke<Gen>('start_generation', { request: { requestId: uid(), modelPath: current.path, prompt: `${playground.system}\n\nUser: ${playground.prompt}\nAssistant:`, maxTokens: playground.maxTokens, temperature: playground.temperature, topP: playground.topP, topK: playground.topK, repeatPenalty: 1.1, contextLength: Math.min(current.context || 4096, 8192), seed: playground.seed } });
      setToolResult(result.text);
    } catch (error) { setToolResult(String(error)); }
  }

  function calculator() { try { const expression = toolInput.replace(/[^0-9+\-*/().% ]/g, ''); if (!expression.trim()) throw new Error('Enter a numeric expression.'); setToolResult(String(Function(`"use strict"; return (${expression})`)())); } catch (error) { setToolResult(String(error)); } }
  function addMemory() { const text = prompt('Save a local memory for Synth:'); if (!text?.trim()) return; setMemories(m => [{ id: uid(), text: text.trim(), scope: 'global', created: new Date().toISOString() }, ...m]); toast('Memory saved locally.'); }
  function clearAll() { localStorage.removeItem('synth.chats'); localStorage.removeItem('synth.knowledge'); localStorage.removeItem('synth.memories'); setChats([welcome]); setKnowledge([]); setMemories([]); setActiveId('welcome'); toast('Local data cleared.'); }

  return <div className={dark ? 'app dark' : 'app'}>
    <Sidebar collapsed={collapsed} setCollapsed={setCollapsed} page={page} setPage={selectPage} query={query} setQuery={setQuery} chats={visible} activeId={activeId} setActiveId={id => { setActiveId(id); selectPage('Chat'); }} newChat={newChat} stats={stats} />
    <main className="main">
      <header className="topbar"><div className="crumb"><strong>{page === 'Lab' ? 'Synth Lab' : page}</strong>{page === 'Chat' && <><span>/</span><span className="muted">{active.title}</span></>}</div><div className="top-actions"><button className="model-pill" onClick={() => selectPage('Models')}><Bot size={15}/><span>{current?.name || 'No model'}</span><ChevronDown size={13}/></button><button className="icon-btn" onClick={() => setDark(v => !v)} title="Toggle theme">{dark ? <Sun size={16}/> : <Moon size={16}/>}</button><button className="icon-btn" onClick={() => setPalette(true)} title="Command palette"><Command size={16}/></button></div></header>
      {page === 'Chat' && <ChatPage active={active} input={input} setInput={setInput} busy={busy} current={current} send={() => void send()} stop={() => void invoke('cancel_generation')} attach={() => void attachFile()} copy={text => { navigator.clipboard?.writeText(text); toast('Copied'); }} editId={editId} editText={editText} setEditText={setEditText} edit={m => { setEditId(m.id); setEditText(m.content); }} save={id => { patch(c => ({ ...c, messages: c.messages.map(m => m.id === id ? { ...m, content: editText } : m) })); setEditId(null); toast('Message updated'); }} cancel={() => setEditId(null)} addMemory={addMemory} />}
      {page === 'Models' && <ModelsPage models={models} selected={selected} importModel={importModel} select={setSelected} remove={id => { setModels(ms => ms.filter(m => m.id !== id)); if (selected === id) setSelected(''); }} />}
      {page === 'Personas' && <PersonasPage items={people} active={personaId} setActive={setPersonaId} save={setPeople} toast={toast} />}
      {page === 'Lab' && <LabPage models={models.filter(m => m.status === 'ready')} a={labA} b={labB} setA={setLabA} setB={setLabB} prompt={labPrompt} setPrompt={setLabPrompt} results={labResults} run={runLab} />}
      {page === 'Playground' && <PlaygroundPage model={current} playground={playground} setPlayground={setPlayground} result={toolResult} run={runPlayground} toolInput={toolInput} setToolInput={setToolInput} calculator={calculator} />}
      {page === 'Knowledge' && <KnowledgePage docs={knowledge} add={addKnowledge} remove={id => setKnowledge(d => d.filter(x => x.id !== id))} search={input} setSearch={setInput} />}
      {page === 'Memory' && <MemoryPage memories={memories} add={addMemory} remove={id => setMemories(m => m.filter(x => x.id !== id))} />}
      {page === 'Settings' && <SettingsPage dark={dark} setDark={setDark} stats={stats} models={models} selected={selected} setSelected={setSelected} recommendation={recommended} clearAll={clearAll} toast={toast} />}
    </main>
    {palette && <Palette close={() => setPalette(false)} go={selectPage} newChat={() => { newChat(); setPalette(false); }} />}
    {quick && <Quick close={() => setQuick(false)} model={current} send={text => void send(text)} />}
    {notice && <div className="toast"><Check size={14}/>{notice}</div>}
  </div>;
}

function Sidebar(p: { collapsed:boolean; setCollapsed:(b:boolean)=>void; page:Page; setPage:(p:Page)=>void; query:string; setQuery:(s:string)=>void; chats:Chat[]; activeId:string; setActiveId:(s:string)=>void; newChat:()=>void; stats:Stats|null }) {
  const nav: [Page,string,ReactNode][] = [['Chat','Chat',<MessageSquarePlus/>],['Models','Models',<Library/>],['Personas','Personas',<UsersRound/>],['Lab','Synth Lab',<Activity/>],['Playground','Playground',<Command/>],['Knowledge','Knowledge',<Archive/>],['Memory','Memory',<KeyRound/>],['Settings','Settings',<Settings/>]];
  return <aside className={`sidebar ${p.collapsed ? 'collapsed' : ''}`}><div className="brand"><div className="brand-mark"><span/><span/><span/><span/><span/></div>{!p.collapsed && <div className="brand-copy"><strong>Synth Studio</strong><small>by Fulltrack</small></div>}{!p.collapsed && <button className="icon-btn tiny" onClick={() => p.setCollapsed(true)} title="Collapse sidebar"><PanelLeft size={15}/></button>}</div>{p.collapsed && <button className="icon-btn expand" onClick={() => p.setCollapsed(false)} title="Expand sidebar"><PanelLeft size={16}/></button>}<button className="new-chat" onClick={p.newChat}><Plus size={16}/>{!p.collapsed && <><span>New chat</span><kbd>Ctrl N</kbd></>}</button>{!p.collapsed && <div className="search"><Search size={13}/><input value={p.query} onChange={e => p.setQuery(e.target.value)} placeholder="Search chats"/><kbd>Ctrl K</kbd></div>}<div className="section-label">{!p.collapsed && 'Workspace'}</div>{nav.map(([id,label,icon]) => <button key={id} className={`nav ${p.page === id ? 'active' : ''}`} onClick={() => p.setPage(id)} title={label}><span className="nav-icon">{icon}</span>{!p.collapsed && <span>{label}</span>}</button>)}{!p.collapsed && <div className="recent"><div className="section-label">Recent</div>{p.chats.slice(0,7).map(c => <button key={c.id} className={`chat-row ${c.id === p.activeId ? 'active' : ''}`} onClick={() => p.setActiveId(c.id)}><History size={12}/><span>{c.title}</span>{c.pinned && <Pin size={10}/>}</button>)}</div>}<div className="sidebar-bottom"><div className="engine"><span className="online"/><span>{p.collapsed ? '' : p.stats ? `${Math.round(p.stats.process_memory_mb)} MB app` : 'On-device'}</span></div></div></aside>;
}

function ChatPage(p: { active:Chat; input:string; setInput:(s:string)=>void; busy:boolean; current?:Model; send:()=>void; stop:()=>void; attach:()=>void; copy:(s:string)=>void; editId:string|null; editText:string; setEditText:(s:string)=>void; edit:(m:Message)=>void; save:(id:string)=>void; cancel:()=>void; addMemory:()=>void }) {
  const suggestions = [['Explain something','Explain a difficult topic using a simple example.'],['Help me code','Find the bug and show me a clean fix.'],['Analyze a file','Summarize code, notes, or documentation locally.']];
  return <div className="chat-page"><div className="chat-scroll">{p.active.messages.length === 1 && p.active.messages[0].id === 'welcome' ? <div className="welcome"><div className="big-mark"><span/><span/><span/><span/><span/></div><div className="eyebrow">Local intelligence</div><h1>Welcome to Synth Studio.</h1><p>Private by default. Fast where your hardware allows. Powered locally with Candle.</p><div className="suggestions">{suggestions.map(([title,body],i) => <button key={title} onClick={() => { document.querySelector<HTMLTextAreaElement>('#composer')?.focus(); p.setInput(body); }}><Sparkles size={16}/><div><span>{title}</span><small>{body}</small></div></button>)}</div></div> : p.active.messages.map(m => <div className={`message ${m.role}`} key={m.id}><div className="avatar">{m.role === 'assistant' ? <Sparkles size={14}/> : <UserRound size={14}/>}</div><div className="message-content"><div className="message-head"><strong>{m.role === 'assistant' ? 'Synth' : 'You'}</strong><span>{m.time}</span></div>{p.editId === m.id ? <div className="edit-box"><textarea value={p.editText} onChange={e => p.setEditText(e.target.value)}/><div><button onClick={p.cancel}>Cancel</button><button className="primary-mini" onClick={() => p.save(m.id)}>Save</button></div></div> : <><div className="bubble" dangerouslySetInnerHTML={{ __html: m.content ? md(m.content) : '<div class="dots"><i></i><i></i><i></i></div>' }}/><div className="message-actions"><button onClick={() => p.copy(m.content)}><Clipboard size={11}/>Copy</button><button onClick={() => p.edit(m)}><Pencil size={11}/>Edit</button>{m.role === 'assistant' && <button onClick={p.addMemory}><KeyRound size={11}/>Remember</button>}{m.stats && <span>{m.stats}</span>}</div></>}</div></div>)}</div><div className="composer-area"><div className="composer"><textarea id="composer" value={p.input} onChange={e => p.setInput(e.target.value)} onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); p.send(); } }} placeholder={p.current ? 'Message Synth…' : 'Load a model to start…'} disabled={!p.current || p.busy}/><div className="composer-bar"><button className="ghost-btn" onClick={p.attach} title="Attach local text/code"><Paperclip size={16}/></button><span className="composer-meta">{p.current?.name || 'No model'} · local</span>{p.busy ? <button className="stop-btn" onClick={p.stop}><span/>Stop</button> : <button className="send-btn" onClick={p.send} disabled={!p.input.trim() || !p.current}><Send size={13}/>Send</button>}</div></div><div className="composer-bottom"><span>No cloud inference. Attached files are read locally.</span><span className="shortcut">Enter to send · Shift+Enter for newline</span></div></div></div>;
}

function PageHeader({eyebrow,title,description,action}:{eyebrow:string;title:string;description:string;action?:ReactNode}) { return <div className="page-title"><div><div className="eyebrow">{eyebrow}</div><h1>{title}</h1><p>{description}</p></div>{action}</div>; }
function ModelsPage(p:{models:Model[];selected:string;importModel:()=>void;select:(s:string)=>void;remove:(s:string)=>void}) { return <div className="page"><PageHeader eyebrow="Model library" title="Models" description="Small, local model files. Synth Studio rejects files above 2 GB." action={<button className="primary" onClick={p.importModel}><Upload size={15}/>Import GGUF</button>}/><div className="card-grid">{p.models.length ? p.models.map(m => <div className={`model-card ${m.id===p.selected?'selected':''}`} key={m.id} onClick={() => p.select(m.id)}><div className="card-top"><div className="round-icon"><Bot size={17}/></div><div><strong>{m.name}</strong><div className="model-status"><span className={`dot ${m.status==='ready'?'green':m.status==='missing'?'amber':''}`}/>{m.status}</div></div><button className="icon-btn" onClick={e => { e.stopPropagation(); p.remove(m.id); }} title="Remove"><Trash2 size={14}/></button></div><div className="tags"><span>{m.size}</span><span>{m.quantization}</span><span>{m.architecture}</span>{m.parameterCount && <span>{m.parameterCount}</span>}</div><div className="model-details"><span>Context {m.context.toLocaleString()}</span><span>Tokenizer {m.tokenizer}</span></div><p className="model-path">{m.path}</p><div className="card-footer"><button className="secondary-mini" onClick={() => p.select(m.id)}>{p.selected===m.id?'Selected':'Use model'}</button><span className="unsupported-copy">{m.status==='ready'?'Candle ready':'Compatibility check required'}</span></div></div>) : <div className="empty-card"><div className="round-icon"><Library size={19}/></div><h3>No local models yet</h3><p>Import a GGUF file. The model remains where you placed it; Synth Studio does not silently copy it.</p><button className="secondary" onClick={p.importModel}><Upload size={14}/>Choose GGUF</button></div>}</div></div>; }

function PersonasPage(p:{items:Persona[];active:string;setActive:(s:string)=>void;save:(x:Persona[])=>void;toast:(s:string)=>void}) { const [creating,setCreating]=useState(false);const [draft,setDraft]=useState<Persona>({id:uid(),name:'New Persona',description:'Custom behavior',prompt:'Be helpful and precise.',temperature:.7});return <div className="page"><PageHeader eyebrow="Behavior" title="Personas" description="Tune how Synth talks without changing the model." action={<button className="secondary" onClick={() => setCreating(v => !v)}><Plus size={14}/>Create persona</button>}/><div className="card-grid persona-grid">{p.items.map(x => <button key={x.id} className={`persona-card ${p.active===x.id?'selected':''}`} onClick={() => p.setActive(x.id)}><div className="round-icon"><WandSparkles size={16}/></div><div className="persona-copy"><h3>{x.name}</h3><p>{x.description}</p></div><span className="persona-temp">{x.temperature.toFixed(2)}</span></button>)}</div>{creating && <div className="editor-card"><div className="editor-form"><label>Name<input value={draft.name} onChange={e => setDraft({...draft,name:e.target.value})}/></label><label>Description<input value={draft.description} onChange={e => setDraft({...draft,description:e.target.value})}/></label><label>Instructions<textarea value={draft.prompt} onChange={e => setDraft({...draft,prompt:e.target.value})}/></label><label>Temperature<input type="range" min="0" max="1.2" step="0.05" value={draft.temperature} onChange={e => setDraft({...draft,temperature:Number(e.target.value)})}/></label><div><button className="secondary" onClick={() => setCreating(false)}>Cancel</button><button className="primary" onClick={() => {p.save([...p.items,draft]);p.setActive(draft.id);setCreating(false);p.toast('Persona created locally.');}}>Save persona</button></div></div></div>}</div>; }

function LabPage(p:{models:Model[];a:string;b:string;setA:(s:string)=>void;setB:(s:string)=>void;prompt:string;setPrompt:(s:string)=>void;results:{name:string;result?:Gen;error?:string}[];run:()=>void}) { return <div className="page"><PageHeader eyebrow="Compare" title="Synth Lab" description="Run the same prompt locally against two models and compare speed and output."/><div className="lab-controls"><label>Model A<select value={p.a} onChange={e => p.setA(e.target.value)}><option value="">Choose…</option>{p.models.map(m => <option key={m.id} value={m.id}>{m.name}</option>)}</select></label><label>Model B<select value={p.b} onChange={e => p.setB(e.target.value)}><option value="">Choose…</option>{p.models.map(m => <option key={m.id} value={m.id}>{m.name}</option>)}</select></label><label>Prompt<textarea value={p.prompt} onChange={e => p.setPrompt(e.target.value)}/></label><button className="primary lab-run" onClick={p.run}><Activity size={15}/>Run experiment</button></div>{p.results.length>0 && <div className="lab-results">{p.results.map(r => <div className="result-card" key={r.name}><div className="result-head"><strong>{r.name}</strong>{r.result && <span>{r.result.tokens_per_sec.toFixed(1)} tok/s</span>}</div>{r.error ? <p>{r.error}</p> : <div className="result-text">{r.result?.text}</div>}{r.result && <div className="result-meta">{r.result.tokens} tokens · {r.result.seconds.toFixed(1)}s</div>}</div>)}</div>}</div>; }

function PlaygroundPage(p:{model?:Model;playground:typeof defaultPlayground;setPlayground:React.Dispatch<React.SetStateAction<typeof defaultPlayground>>;result:string;run:()=>void;toolInput:string;setToolInput:(s:string)=>void;calculator:()=>void}) { return <div className="page"><PageHeader eyebrow="Experiment" title="Playground" description="Test prompts and generation parameters. The selected model stays local." action={<button className="primary" onClick={p.run} disabled={!p.model}><Zap size={15}/>Run</button>}/><div className="playground-grid"><div className="prompt-card"><label>System prompt<textarea value={p.playground.system} onChange={e=>p.setPlayground(x=>({...x,system:e.target.value}))}/></label><label>User prompt<textarea value={p.playground.prompt} onChange={e=>p.setPlayground(x=>({...x,prompt:e.target.value}))}/></label><div className="slider-grid"><label>Temperature<span>{p.playground.temperature.toFixed(2)}</span><input type="range" min="0" max="1.2" step="0.05" value={p.playground.temperature} onChange={e=>p.setPlayground(x=>({...x,temperature:Number(e.target.value)}))}/></label><label>Top P<span>{p.playground.topP.toFixed(2)}</span><input type="range" min="0.1" max="1" step="0.05" value={p.playground.topP} onChange={e=>p.setPlayground(x=>({...x,topP:Number(e.target.value)}))}/></label><label>Top K<span>{p.playground.topK}</span><input type="range" min="0" max="100" step="1" value={p.playground.topK} onChange={e=>p.setPlayground(x=>({...x,topK:Number(e.target.value)}))}/></label></div><div className="toolbox"><div className="tool-title"><Calculator size={15}/>Local calculator</div><div className="tool-row"><input value={p.toolInput} onChange={e=>p.setToolInput(e.target.value)} placeholder="18*7+4"/><button className="secondary" onClick={p.calculator}>Calculate</button></div></div></div><div className="output-card"><div className="output-head"><span>{p.model?.name || 'No model'}</span><span>local</span></div><div className="output">{p.result || <div className="empty-output">Run an experiment to see output.</div>}</div></div></div></div>; }
const defaultPlayground={prompt:'Explain recursion with a simple example.',system:'Be concise and clear.',temperature:.7,topP:.9,topK:40,maxTokens:256,seed:42};

function KnowledgePage(p:{docs:Doc[];add:()=>void;remove:(s:string)=>void;search:string;setSearch:(s:string)=>void}) { const filtered=p.docs.filter(d => !p.search || d.name.toLowerCase().includes(p.search.toLowerCase()) || d.content.toLowerCase().includes(p.search.toLowerCase()));return <div className="page"><PageHeader eyebrow="Local RAG" title="Knowledge" description="Store text locally and let Synth retrieve the most relevant pieces during chat." action={<button className="primary" onClick={p.add}><FolderOpen size={15}/>Add files</button>}/><div className="knowledge-search"><Search size={14}/><input value={p.search} onChange={e=>p.setSearch(e.target.value)} placeholder="Search your local knowledge"/></div><div className="doc-list">{filtered.length ? filtered.map(d=><div className="doc-row" key={d.id}><div className="round-icon"><FileText size={16}/></div><div><strong>{d.name}</strong><p>{d.size} · available to local retrieval</p></div><button className="icon-btn" onClick={()=>p.remove(d.id)} title="Remove"><Trash2 size={14}/></button></div>) : <div className="empty-card"><Archive size={18}/><h3>No local knowledge</h3><p>Add notes, source files, or documentation. Retrieval is keyword-based and stays on-device.</p></div>}</div></div>; }
function MemoryPage(p:{memories:Memory[];add:()=>void;remove:(s:string)=>void}) { return <div className="page"><PageHeader eyebrow="Local memory" title="Memory" description="Explicit memories only. Nothing is saved here unless you add it." action={<button className="primary" onClick={p.add}><Plus size={14}/>Add memory</button>}/><div className="memory-list">{p.memories.length ? p.memories.map(m=><div className="memory-row" key={m.id}><div className="round-icon"><KeyRound size={15}/></div><div><strong>{m.text}</strong><p>{m.scope} · {new Date(m.created).toLocaleString()}</p></div><button className="icon-btn" onClick={()=>p.remove(m.id)}><Trash2 size={14}/></button></div>) : <div className="empty-card"><KeyRound size={18}/><h3>No saved memories</h3><p>Use Remember from a message or add one explicitly.</p></div>}</div></div>; }
function SettingsPage(p:{dark:boolean;setDark:(b:boolean)=>void;stats:Stats|null;models:Model[];selected:string;setSelected:(s:string)=>void;recommendation:string;clearAll:()=>void;toast:(s:string)=>void}) { return <div className="page"><PageHeader eyebrow="System" title="Settings" description="Keep Synth Studio small, local, and predictable."/><div className="settings-list"><div className="setting-group"><div className="setting-line"><div className="setting-copy"><strong>Appearance</strong><span>Light is the default. Dark is available for low-light work.</span></div><button className="secondary" onClick={()=>p.setDark(!p.dark)}>{p.dark?<Sun size={14}/>:<Moon size={14}/>} {p.dark?'Switch to light':'Switch to dark'}</button></div></div><div className="setting-group"><div className="setting-line"><div className="setting-copy"><strong>Active model</strong><span>Choose the model used by Chat and Playground.</span></div><select value={p.selected} onChange={e=>p.setSelected(e.target.value)}><option value="">No model</option>{p.models.map(m=><option key={m.id} value={m.id}>{m.name}</option>)}</select></div></div><div className="setting-group"><div className="setting-line"><div className="setting-copy"><strong>Hardware advisor</strong><span>{p.recommendation}</span></div><Gauge size={18}/></div>{p.stats && <div className="stats-grid"><div><small>CPU</small><strong>{Math.round(p.stats.cpu_usage)}%</strong></div><div><small>RAM</small><strong>{p.stats.memory_used_mb} / {p.stats.memory_total_mb} MB</strong></div><div><small>App</small><strong>{p.stats.process_memory_mb} MB</strong></div><div><small>GPU</small><strong>{p.stats.gpu_available?'Available':'CPU default'}</strong></div></div>}</div><div className="setting-group danger-group"><div className="setting-line"><div className="setting-copy"><strong>Clear local data</strong><span>Deletes chats, local knowledge, and Synth memory stored in this app.</span></div><button className="danger-mini" onClick={p.clearAll}><Trash2 size={14}/>Clear data</button></div></div><div className="setting-group"><div className="privacy"><span><Check size={12}/>No account</span><span><Check size={12}/>No cloud inference</span><span><Check size={12}/>Local files</span></div></div></div></div>; }
function Palette(p:{close:()=>void;go:(x:Page)=>void;newChat:()=>void}) { const actions:[string,Page|string,ReactNode][]=[['Chat','Chat',<MessageSquarePlus/>],['Models','Models',<Library/>],['Personas','Personas',<UsersRound/>],['Synth Lab','Lab',<Activity/>],['Playground','Playground',<Command/>],['Knowledge','Knowledge',<Archive/>],['Memory','Memory',<KeyRound/>],['Settings','Settings',<Settings/>]];return <div className="overlay" onMouseDown={p.close}><div className="palette" onMouseDown={e=>e.stopPropagation()}><div className="palette-head"><Command size={16}/><input autoFocus placeholder="Jump to…"/><kbd>ESC</kbd></div>{actions.map(([label,target,icon])=><button key={label} onClick={()=>{p.go(target as Page);p.close();}}><span>{icon}</span>{label}</button>)}<button onClick={()=>{p.newChat();p.close();}}><Plus size={15}/>New chat</button></div></div>; }
function Quick(p:{close:()=>void;model?:Model;send:(text:string)=>void}) { const [text,setText]=useState('');return <div className="quick-wrap" onMouseDown={p.close}><div className="quick" onMouseDown={e=>e.stopPropagation()}><div className="quick-head"><div className="big-mark small"><span/><span/><span/><span/><span/></div><div><strong>Synth Quick</strong><small>{p.model?.name||'No model loaded'}</small></div><button className="icon-btn" onClick={p.close}><X size={15}/></button></div><textarea autoFocus value={text} onChange={e=>setText(e.target.value)} onKeyDown={e=>{if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();if(text.trim())p.send(text);p.close();}}} placeholder="Ask Synth anything…"/><div className="quick-foot"><span>Ctrl+Shift+Space</span><span>Enter to send</span></div></div></div>; }

createRoot(document.getElementById('root')!).render(<App />);
