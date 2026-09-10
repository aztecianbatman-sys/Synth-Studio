import React, { useEffect, useMemo, useState } from 'react';
import { createRoot } from 'react-dom/client';
import { invoke } from '@tauri-apps/api/core';
import { open } from '@tauri-apps/plugin-dialog';
import { 
  Archive, Bot, Check, ChevronDown, CircleHelp, Command, Cpu, Download, FileText, FolderOpen,
  Gauge, Hash, History, Library, MessageSquarePlus, Moon, MoreHorizontal, Paperclip, PanelLeft,
  Pin, Plus, Search, Send, Settings, Sparkles, Sun, Trash2, Upload, UserRound, UsersRound, X, Zap
} from 'lucide-react';
import './styles.css';

type Page = 'Chat' | 'Models' | 'Personas' | 'Settings';
type Message = { role: 'user' | 'assistant'; content: string; time: string; stats?: string };
type Chat = { id: number; title: string; pinned: boolean; messages: Message[] };
type Model = { name: string; size: string; status: 'installed' | 'missing'; quant: string; architecture: string };

const now = () => new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
const starterChats: Chat[] = [{ id: 1, title: 'Welcome to Synth Studio', pinned: true, messages: [{ role: 'assistant', content: 'Welcome to **Synth Studio**. Your local AI workspace is ready.\n\nImport a GGUF model from **Models** to begin local inference.', time: now() }] }];
const starterModels: Model[] = [{ name: 'No model imported', size: '—', status: 'missing', quant: '—', architecture: '—' }];
const personas = ['Balanced', 'Coder', 'Tutor', 'Creative', 'Researcher', 'Companion'];

function App() {
  const [page, setPage] = useState<Page>('Chat');
  const [dark, setDark] = useState(false);
  const [collapsed, setCollapsed] = useState(false);
  const [query, setQuery] = useState('');
  const [input, setInput] = useState('');
  const [selectedModel, setSelectedModel] = useState('No model imported');
  const [persona, setPersona] = useState('Balanced');
  const [chats, setChats] = useState(starterChats);
  const [activeId, setActiveId] = useState(1);
  const [models, setModels] = useState(starterModels);
  const [busy, setBusy] = useState(false);
  const [notice, setNotice] = useState('');
  const [engine, setEngine] = useState('Native Rust bridge ready');

  const active = chats.find(c => c.id === activeId) ?? chats[0];
  const visibleChats = useMemo(() => chats.filter(c => c.title.toLowerCase().includes(query.toLowerCase())), [chats, query]);

  useEffect(() => {
    invoke<string>('engine_status').then(setEngine).catch(() => setEngine('Rust engine unavailable'));
  }, []);

  const toast = (text: string) => { setNotice(text); window.setTimeout(() => setNotice(''), 2400); };
  const newChat = () => { const id = Date.now(); setChats(c => [{ id, title: 'New conversation', pinned: false, messages: [] }, ...c]); setActiveId(id); setPage('Chat'); };
  const deleteChat = (id: number) => { setChats(c => c.filter(x => x.id !== id)); if (activeId === id) setActiveId(chats.find(x => x.id !== id)?.id ?? 0); };
  const togglePin = (id: number) => setChats(c => c.map(x => x.id === id ? { ...x, pinned: !x.pinned } : x));

  const send = async () => {
    const text = input.trim();
    if (!text || busy) return;
    const stamp = now();
    setInput(''); setBusy(true);
    setChats(c => c.map(x => x.id === activeId ? { ...x, title: x.messages.length ? x.title : text.slice(0, 34), messages: [...x.messages, { role: 'user', content: text, time: stamp }] } : x));
    try {
      const response = await invoke<string>('local_inference', { prompt: text, model: selectedModel });
      setChats(c => c.map(x => x.id === activeId ? { ...x, messages: [...x.messages, { role: 'assistant', content: response, time: now(), stats: 'Local engine' }] } : x));
    } catch (e) {
      setChats(c => c.map(x => x.id === activeId ? { ...x, messages: [...x.messages, { role: 'assistant', content: `**Local inference is not configured yet.**\n\n${String(e)}\n\nImport a GGUF model in Models to continue.`, time: now() }] } : x));
    } finally { setBusy(false); }
  };

  const importModel = async () => {
    try {
      const selected = await open({ multiple: false, filters: [{ name: 'GGUF model', extensions: ['gguf'] }] });
      if (!selected || Array.isArray(selected)) return;
      const path = String(selected);
      const name = path.split(/[\\/]/).pop() || 'Imported GGUF';
      const result = await invoke<Model>('register_model', { path });
      setModels([result, ...models.filter(m => m.name !== 'No model imported')]); setSelectedModel(result.name); toast(`${name} imported`);
    } catch (e) { toast(`Model import failed: ${String(e)}`); }
  };

  return <div className={dark ? 'app dark' : 'app'}>
    <aside className={collapsed ? 'sidebar collapsed' : 'sidebar'}>
      <div className="brand"><div className="synth-mark"><Sparkles size={17}/></div>{!collapsed && <><div><strong>Synth Studio</strong><span>by Fulltrack</span></div><button className="icon-btn tiny" onClick={() => setCollapsed(true)} title="Collapse sidebar"><PanelLeft size={16}/></button></>}</div>
      {collapsed && <button className="icon-btn expand" onClick={() => setCollapsed(false)} title="Expand sidebar"><PanelLeft size={17}/></button>}
      <button className="new-chat" onClick={newChat}><Plus size={17}/>{!collapsed && 'New chat'}{!collapsed && <kbd>⌘N</kbd>}</button>
      {!collapsed && <div className="search"><Search size={15}/><input placeholder="Search chats" value={query} onChange={e => setQuery(e.target.value)}/><kbd>⌘K</kbd></div>}
      <div className="nav-label">{!collapsed && 'Workspace'}</div>
      <Nav icon={<MessageSquarePlus/>} label="Chat" active={page === 'Chat'} collapsed={collapsed} onClick={() => setPage('Chat')}/>
      <Nav icon={<Library/>} label="Models" active={page === 'Models'} collapsed={collapsed} onClick={() => setPage('Models')}/>
      <Nav icon={<UsersRound/>} label="Personas" active={page === 'Personas'} collapsed={collapsed} onClick={() => setPage('Personas')}/>
      <div className="nav-label spacer">{!collapsed && 'System'}</div>
      <Nav icon={<Settings/>} label="Settings" active={page === 'Settings'} collapsed={collapsed} onClick={() => setPage('Settings')}/>
      <div className="sidebar-bottom">
        <div className="engine"><span className="status-dot"/><span>{collapsed ? '' : engine}</span></div>
        {!collapsed && <div className="privacy"><span><Check size={13}/> Local-first</span><span>No account</span></div>}
      </div>
    </aside>

    <main className="main">
      <header className="topbar">
        <div className="crumb"><span>{page}</span>{page === 'Chat' && <><span className="slash">/</span><span className="muted">{active?.title ?? 'New conversation'}</span></>}</div>
        <div className="top-actions">
          <button className="model-select" onClick={() => setPage('Models')}><Bot size={16}/><span>{selectedModel}</span><ChevronDown size={14}/></button>
          <button className="icon-btn" onClick={() => setDark(!dark)} title="Toggle appearance">{dark ? <Sun size={17}/> : <Moon size={17}/>}</button>
          <button className="icon-btn" onClick={() => toast('Command palette: coming next')} title="Command palette"><Command size={17}/></button>
        </div>
      </header>

      {page === 'Chat' && <>
        <section className="chat-wrap">
          <div className="chat-scroll">
            {active?.messages.length === 0 && <Welcome onNew={newChat}/>} 
            {active?.messages.map((m, i) => <MessageView key={i} message={m} onCopy={() => navigator.clipboard?.writeText(m.content).then(() => toast('Copied'))}/>) }
            {busy && <div className="message assistant"><div className="avatar"><Sparkles size={15}/></div><div className="bubble"><span className="typing"><i/><i/><i/></span></div></div>}
          </div>
          <div className="composer-area">
            <div className="composer">
              <textarea value={input} onChange={e => setInput(e.target.value)} onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); } }} placeholder={selectedModel === 'No model imported' ? 'Import a model to chat locally…' : 'Message Synth…'} rows={1}/>
              <div className="composer-bar"><button className="composer-icon" title="Attach file" onClick={() => toast('File attachment is wired for the next integration step')}><Paperclip size={17}/></button><span className="composer-hint">Shift + Enter for a new line</span><button className="send" onClick={send} disabled={!input.trim() || busy}><Send size={16}/></button></div>
            </div>
            <p className="disclaimer">Synth Studio runs inference locally when a compatible model is installed. Your chats stay on this device.</p>
          </div>
        </section>
      </>}

      {page === 'Models' && <Models models={models} onImport={importModel} onRemove={name => { setModels(m => m.filter(x => x.name !== name)); if (selectedModel === name) setSelectedModel('No model imported'); toast('Model removed'); }}/>} 
      {page === 'Personas' && <Personas active={persona} setActive={setPersona} toast={toast}/>} 
      {page === 'Settings' && <SettingsPage dark={dark} setDark={setDark} toast={toast}/>} 
    </main>
    {notice && <div className="toast"><Check size={15}/>{notice}</div>}
  </div>;
}

function Nav({icon,label,active,collapsed,onClick}:{icon:React.ReactNode;label:string;active:boolean;collapsed:boolean;onClick:()=>void}) { return <button className={active ? 'nav active' : 'nav'} onClick={onClick} title={collapsed ? label : undefined}>{React.cloneElement(icon as React.ReactElement, {size:17})}<span>{!collapsed && label}</span></button>; }
function Welcome({onNew}:{onNew:()=>void}) { return <div className="welcome"><div className="hero-mark"><Sparkles size={25}/></div><h1>What can Synth do for you?</h1><p>A clean, private workspace for local intelligence.</p><div className="quick-grid"><button onClick={onNew}><FileText size={17}/><span>Analyze a file<small>Understand local documents</small></span></button><button onClick={onNew}><Zap size={17}/><span>Brainstorm<small>Explore an idea</small></span></button><button onClick={onNew}><Hash size={17}/><span>Write code<small>Build and debug</small></span></button><button onClick={onNew}><CircleHelp size={17}/><span>Learn something<small>Get a clear explanation</small></span></button></div></div>; }
function MessageView({message,onCopy}:{message:Message;onCopy:()=>void}) { return <div className={`message ${message.role}`}><div className="avatar">{message.role === 'assistant' ? <Sparkles size={15}/> : <UserRound size={15}/>}</div><div className="message-body"><div className="message-meta"><strong>{message.role === 'assistant' ? 'Synth' : 'You'}</strong><span>{message.time}</span></div><div className="bubble" dangerouslySetInnerHTML={{__html:renderMarkdown(message.content)}}/><div className="message-tools"><button onClick={onCopy}><Archive size={13}/> Copy</button>{message.role === 'assistant' && message.stats && <span>{message.stats}</span>}</div></div></div>; }
function renderMarkdown(text:string) { return text.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/\*\*(.*?)\*\*/g,'<strong>$1</strong>').replace(/`([^`]+)`/g,'<code>$1</code>').replace(/\n/g,'<br/>'); }
function Models({models,onImport,onRemove}:{models:Model[];onImport:()=>void;onRemove:(name:string)=>void}) { return <div className="page"><div className="page-head"><div><p className="eyebrow">LOCAL LIBRARY</p><h1>Models</h1><p>Manage the models Synth can run on this machine.</p></div><button className="primary" onClick={onImport}><Upload size={16}/> Import GGUF</button></div><div className="model-grid">{models.map(m => <div className="model-card" key={m.name}><div className="card-icon"><Bot size={19}/></div><div className="card-title"><div><h3>{m.name}</h3><span className={m.status === 'installed' ? 'installed' : 'missing'}>{m.status === 'installed' ? '● Installed' : '○ Not installed'}</span></div><button className="icon-btn" title="Model actions"><MoreHorizontal size={17}/></button></div><div className="model-meta"><span>{m.size}</span><span>{m.quant}</span><span>{m.architecture}</span></div>{m.status === 'installed' ? <div className="card-actions"><button className="secondary"><Check size={14}/> Ready</button><button className="danger" onClick={() => onRemove(m.name)}><Trash2 size={14}/> Remove</button></div> : <div className="empty-model">Import a local <strong>.gguf</strong> file to use this model.</div>}</div>)}</div></div>; }
function Personas({active,setActive,toast}:{active:string;setActive:(x:string)=>void;toast:(x:string)=>void}) { return <div className="page"><div className="page-head"><div><p className="eyebrow">BEHAVIOR</p><h1>Personas</h1><p>Give your local model a consistent way of working.</p></div><button className="primary" onClick={() => toast('Custom persona editor coming next')}><Plus size={16}/> Create persona</button></div><div className="persona-grid">{personas.map((p,i)=><button key={p} className={active===p ? 'persona active' : 'persona'} onClick={()=>{setActive(p);toast(`${p} selected`)}}><div className="persona-icon"><Sparkles size={17}/></div><div><strong>{p}</strong><p>{['General purpose assistant','Programming focused','Step-by-step explanations','Writing and ideation','Evidence-focused analysis','Natural conversation'][i]}</p></div>{active===p&&<Check size={16}/>}</button>)}</div><div className="editor-card"><div><p className="eyebrow">ACTIVE PERSONA</p><h2>{active}</h2><p>Temperature presets and system instructions will be editable here. Your selection is stored locally.</p></div><button className="secondary" onClick={()=>toast('Persona editor coming next')}><Settings size={14}/> Configure</button></div></div>; }
function SettingsPage({dark,setDark,toast}:{dark:boolean;setDark:(x:boolean)=>void;toast:(x:string)=>void}) { return <div className="page settings-page"><div className="page-head"><div><p className="eyebrow">SYSTEM</p><h1>Settings</h1><p>Control how Synth Studio behaves on this device.</p></div></div><SettingGroup title="Appearance"><SettingRow icon={dark?<Moon/>:<Sun/>} title="Theme" desc="Choose the visual appearance of Synth Studio."><button className="segmented" onClick={()=>setDark(false)}><span className={!dark?'selected':''}>Light</span><span onClick={(e)=>{e.stopPropagation();setDark(true)}} className={dark?'selected':''}>Dark</span></button></SettingRow><SettingRow icon={<Gauge/>} title="Interface scale" desc="Adjust the density of the workspace."><span className="value">100%</span></SettingRow></SettingGroup><SettingGroup title="Local AI"><SettingRow icon={<Library/>} title="Model directory" desc="Where imported local models are stored or discovered."><button className="secondary" onClick={()=>toast('Directory picker coming with model storage integration')}><FolderOpen size={14}/> Choose</button></SettingRow><SettingRow icon={<Cpu/>} title="Runtime" desc="Native Rust bridge with local inference adapter."><span className="badge">Local</span></SettingRow></SettingGroup><SettingGroup title="Privacy & data"><SettingRow icon={<Archive/>} title="Local-first storage" desc="Conversations and settings remain on this device."><span className="badge"><Check size={12}/> Enabled</span></SettingRow><SettingRow icon={<Trash2/>} title="Clear conversations" desc="Permanently remove local conversation history."><button className="danger" onClick={()=>toast('Conversation clearing requires confirmation')}>Clear</button></SettingRow></SettingGroup></div>; }
function SettingGroup({title,children}:{title:string;children:React.ReactNode}) { return <section className="setting-group"><h2>{title}</h2>{children}</section>; }
function SettingRow({icon,title,desc,children}:{icon:React.ReactNode;title:string;desc:string;children:React.ReactNode}) { return <div className="setting-row"><div className="setting-icon">{React.cloneElement(icon as React.ReactElement,{size:17})}</div><div className="setting-copy"><strong>{title}</strong><span>{desc}</span></div><div>{children}</div></div>; }

createRoot(document.getElementById('root')!).render(<React.StrictMode><App/></React.StrictMode>);
