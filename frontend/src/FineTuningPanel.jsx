import React from 'react';
import { useCallback, useEffect, useMemo, useState } from 'react';
import {
  AlertCircle, BadgeCheck, BookOpenCheck, Check, ChevronRight, CircleHelp,
  ClipboardCheck, Cpu, Database, FileText, GraduationCap, LoaderCircle,
  Play, RefreshCw, Save, ShieldCheck, Sparkles,
} from 'lucide-react';

const API_URL = (import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000').replace(/\/+$/, '');

async function api(path, options) {
  const response = await fetch(`${API_URL}${path}`, options);
  let payload;
  try { payload = await response.json(); } catch { payload = {}; }
  if (!response.ok) throw new Error(payload.detail || `Request မအောင်မြင်ပါ (${response.status})`);
  return payload;
}

function formatDate(value) {
  if (!value) return '—';
  const date = new Date(value);
  return Number.isNaN(date.valueOf()) ? value : date.toLocaleString();
}

function FineTuningPanel() {
  const [status, setStatus] = useState(null);
  const [selectedId, setSelectedId] = useState('');
  const [answer, setAnswer] = useState('');
  const [sourceDocument, setSourceDocument] = useState('');
  const [sourceQuote, setSourceQuote] = useState('');
  const [verifiedBy, setVerifiedBy] = useState('');
  const [verified, setVerified] = useState(false);
  const [baseModel, setBaseModel] = useState('Qwen/Qwen2.5-3B-Instruct');
  const [epochs, setEpochs] = useState(3);
  const [busy, setBusy] = useState('');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [notice, setNotice] = useState('');

  const refresh = useCallback(async (quiet = false) => {
    if (!quiet) setLoading(true);
    try {
      const fresh = await api('/api/fine-tuning/status');
      setStatus(fresh);
      setError(fresh.error || '');
      setSelectedId(current => fresh.pending_questions?.some(row => row.fingerprint === current)
        ? current : (fresh.pending_questions?.[0]?.fingerprint || ''));
    } catch (e) {
      setError(e.message || 'Backend နှင့် ဆက်သွယ်၍မရပါ။');
    } finally {
      if (!quiet) setLoading(false);
    }
  }, []);

  useEffect(() => { refresh(); }, [refresh]);
  const taskState = status?.task?.state || 'idle';
  useEffect(() => {
    if (!['running', 'installing'].includes(taskState)) return undefined;
    const timer = window.setInterval(() => refresh(true), 2500);
    return () => window.clearInterval(timer);
  }, [refresh, taskState]);

  const selected = useMemo(
    () => status?.pending_questions?.find(row => row.fingerprint === selectedId),
    [status, selectedId],
  );

  function selectQuestion(row) {
    setSelectedId(row.fingerprint);
    setAnswer(''); setSourceDocument(''); setSourceQuote(''); setVerified(false);
    setNotice(''); setError('');
  }

  async function runAction(key, path, body, successText) {
    setBusy(key); setError(''); setNotice('');
    try {
      const result = await api(path, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        ...(body ? { body: JSON.stringify(body) } : {}),
      });
      setNotice(result.message || successText);
      await refresh(true);
      return result;
    } catch (e) {
      setError(e.message || 'လုပ်ဆောင်မှု မအောင်မြင်ပါ။');
      return null;
    } finally { setBusy(''); }
  }

  async function saveReview(event) {
    event.preventDefault();
    if (!selected) return;
    const saved = await runAction('review', '/api/fine-tuning/review', {
      fingerprint: selected.fingerprint,
      answer,
      source_document: sourceDocument,
      source_quote: sourceQuote,
      verified_by: verifiedBy,
      verified,
    }, 'အဖြေကို အတည်ပြုပြီး သိမ်းဆည်းလိုက်ပါပြီ။');
    if (saved) {
      setAnswer(''); setSourceDocument(''); setSourceQuote(''); setVerified(false);
    }
  }

  async function prepareDataset() {
    const result = await runAction('prepare', '/api/fine-tuning/prepare', {}, 'သင်ကြားရန် dataset ပြင်ဆင်ပြီးပါပြီ။');
    if (result) setNotice(`Dataset ပြင်ဆင်ပြီးပါပြီ။ သင်ကြားရန် ${result.exported} ခု ထည့်ပြီး၊ ${result.skipped} ခုကို ကျော်ထားပါသည်။`);
  }

  const runtime = status?.runtime || {};
  const minExamples = status?.minimum_examples || 10;
  const examplesReady = (status?.training_examples || 0) >= minExamples;
  const activeTask = ['running', 'installing'].includes(taskState);
  const canTrain = runtime.ready && examplesReady && !activeTask;

  return (
    <section className="fine-page">
      <div className="fine-heading">
        <div>
          <div className="fine-kicker"><Sparkles size={14}/> မေးဖြေစနစ်ကို တိုးတက်အောင်လုပ်ရန်</div>
          <h1>အဖြေများကို သုံးသပ်ပြီး သင်ကြားပါ</h1>
          <p>မဖြေနိုင်ခဲ့သောမေးခွန်းကို စာရွက်စာတမ်းနှင့်တိုက်စစ်၊ အတည်ပြုထားသောအဖြေကို သိမ်းဆည်းပြီး dataset ပြင်ဆင်ကာ fine-tuning လုပ်နိုင်ပါသည်။</p>
        </div>
        <button className="fine-refresh" onClick={() => refresh()} disabled={loading || activeTask}>
          {loading ? <LoaderCircle className="spin" size={16}/> : <RefreshCw size={16}/>} ပြန်စစ်ရန်
        </button>
      </div>

      {error && <div className="fine-alert error"><AlertCircle size={17}/><span>{error}</span></div>}
      {notice && <div className="fine-alert success"><BadgeCheck size={17}/><span>{notice}</span></div>}

      <div className="fine-stats">
        <div className="fine-stat"><span className="stat-icon amber"><CircleHelp size={18}/></span><div><strong>{status?.pending_count ?? '—'}</strong><small>စစ်ဆေးရန် မေးခွန်း</small></div></div>
        <div className="fine-stat"><span className="stat-icon green"><ShieldCheck size={18}/></span><div><strong>{status?.reviewed_count ?? '—'}</strong><small>အတည်ပြုပြီး</small></div></div>
        <div className="fine-stat"><span className="stat-icon blue"><Database size={18}/></span><div><strong>{status?.training_examples ?? '—'} <small>/ {minExamples}</small></strong><small>သင်ကြားရန်အချက်များ</small></div></div>
        <div className="fine-stat"><span className="stat-icon violet"><Cpu size={18}/></span><div><strong>{runtime.ready ? 'အသင့်' : 'မပြင်ဆင်ရသေး'}</strong><small>သင်ကြားရေးစက်</small></div></div>
      </div>

      {status?.provider_only_count > 0 && <div className="fine-info"><AlertCircle size={16}/><span>Provider error ကြောင့် မဖြေနိုင်ခဲ့သော မေးခွန်း {status.provider_only_count} ခုကို သုံးသပ်ရန်စာရင်းထဲ မထည့်ထားပါ။ Provider နှင့် document search ပြဿနာကို အရင်ဖြေရှင်းပါ။</span></div>}

      <div className="fine-steps">
        <section className="fine-card review-card">
          <div className="fine-card-head">
            <span className="step-no">၁</span>
            <div><h2>မေးခွန်းနှင့် အဖြေကို သုံးသပ်ပါ</h2><p>ရင်းမြစ်စာသားထဲက အဖြေကိုသာ အတည်ပြုသိမ်းဆည်းပါ။</p></div>
          </div>
          <div className="review-layout">
            <div className="question-list" aria-label="သုံးသပ်ရန် မေးခွန်းများ">
              {status?.pending_questions?.length ? status.pending_questions.map(row => (
                <button key={row.fingerprint} className={`question-row ${selectedId === row.fingerprint ? 'selected' : ''}`} onClick={() => selectQuestion(row)}>
                  <span className="question-row-title">{row.question}</span>
                  <span className="question-row-meta">{row.attempts} ကြိမ် · နောက်ဆုံး {formatDate(row.last_seen)}</span>
                </button>
              )) : <div className="empty-questions"><BookOpenCheck size={23}/><strong>သုံးသပ်ရန် မေးခွန်းမရှိသေးပါ</strong><span>Chat မှာ ဖြေရှင်းမရခဲ့သော မေးခွန်းများရှိလာလျှင် ဒီနေရာတွင် ပေါ်လာပါမည်။</span></div>}
            </div>
            <form className="review-form" onSubmit={saveReview}>
              {selected ? <>
                <div className="selected-question"><small>ရွေးထားသောမေးခွန်း</small><strong>{selected.question}</strong></div>
                <label>အတည်ပြုထားသော အဖြေ <span>*</span>
                  <textarea required maxLength={12000} rows={4} value={answer} onChange={e => setAnswer(e.target.value)} placeholder="စာရွက်စာတမ်းက ထောက်ခံထားတဲ့ အဖြေကို ရေးပါ" />
                </label>
                <label>ရင်းမြစ်စာရွက်စာတမ်း <span>*</span>
                  <input required maxLength={500} value={sourceDocument} onChange={e => setSourceDocument(e.target.value)} placeholder="ဥပမာ - preRelease2.docx၊ အခန်း ၉" />
                </label>
                <label>စာရွက်စာတမ်းမှ သက်ဆိုင်ရာစာပိုဒ် <span>*</span>
                  <textarea required maxLength={24000} rows={5} value={sourceQuote} onChange={e => setSourceQuote(e.target.value)} placeholder="အဖြေကို ထောက်ခံတဲ့ မူရင်းစာသားကို ကူးထည့်ပါ" />
                </label>
                <label>စစ်ဆေးသူ (ရွေးချယ်နိုင်သည်)
                  <input maxLength={200} value={verifiedBy} onChange={e => setVerifiedBy(e.target.value)} placeholder="အမည် သို့မဟုတ် အဖွဲ့" />
                </label>
                <label className="verify-check"><input type="checkbox" checked={verified} onChange={e => setVerified(e.target.checked)} /><span>အဖြေနှင့် ကိုးကားထားသော မူရင်းစာသားကို စစ်ဆေးပြီးဖြစ်သည်။</span></label>
                <button className="fine-primary" type="submit" disabled={busy === 'review' || !verified}>
                  {busy === 'review' ? <LoaderCircle className="spin" size={16}/> : <Save size={16}/>} အတည်ပြုပြီး သိမ်းမည်
                </button>
                <small className="form-hint">အဖြေစာသားကို မူရင်းစာပိုဒ်ထဲတွင် ပါဝင်ကြောင်း backend က ထပ်မံစစ်ဆေးပါမည်။</small>
              </> : <div className="choose-prompt"><ChevronRight size={18}/> သုံးသပ်ရန် ဘယ်ဘက်မှ မေးခွန်းတစ်ခု ရွေးပါ။</div>}
            </form>
          </div>
        </section>

        <section className="fine-card">
          <div className="fine-card-head">
            <span className="step-no">၂</span>
            <div><h2>သင်ကြားရန် dataset ပြင်ဆင်ပါ</h2><p>အတည်ပြုထားပြီး ရင်းမြစ်နှင့် ကိုက်ညီသော QA အချက်အလက်များကိုသာ ထုတ်ယူပါမည်။</p></div>
          </div>
          <div className="action-strip">
            <div className="action-explain"><Database size={19}/><div><strong>Dataset ဖိုင် ပြန်တည်ဆောက်မည်</strong><span>Training အတွက် လက်ရှိအတည်ပြုထားသော မေးခွန်း/အဖြေများကို ပြင်ဆင်ပါ။</span></div></div>
            <button className="fine-secondary" disabled={!!busy || activeTask} onClick={prepareDataset}>
              {busy === 'prepare' ? <LoaderCircle className="spin" size={16}/> : <ClipboardCheck size={16}/>} ပြင်ဆင်ရန်
            </button>
          </div>
          {status?.training_examples != null && <div className={`dataset-note ${examplesReady ? 'ready' : ''}`}><Check size={15}/>{examplesReady ? `အနည်းဆုံး ${minExamples} ခု ပြည့်ပါပြီ။ Training စတင်နိုင်သည်။` : `Training မစတင်မီ အတည်ပြုထားသော ${minExamples} ခု လိုအပ်ပါသည်။ လက်ရှိ ${status.training_examples} ခု ရှိပါသည်။`}</div>}
        </section>

        <section className="fine-card">
          <div className="fine-card-head">
            <span className="step-no">၃</span>
            <div><h2>Model ကို သင်ကြားပါ</h2><p>လုပ်ငန်းစဉ်သည် နောက်ခံတွင် လုပ်ဆောင်မည်ဖြစ်ပြီး အခြေအနေကို ဒီနေရာတွင် ပြပါမည်။</p></div>
          </div>
          <div className="runtime-box">
            <div className={`runtime-status ${runtime.ready ? 'ready' : 'blocked'}`}>
              {runtime.ready ? <BadgeCheck size={18}/> : <AlertCircle size={18}/>}
              <div><strong>{runtime.ready ? 'သင်ကြားရေး ပတ်ဝန်းကျင် အသင့်ရှိသည်' : 'သင်ကြားရေး ပတ်ဝန်းကျင် မပြည့်စုံသေးပါ'}</strong>
                <span>{runtime.ready ? `Python: ${runtime.python}` : (runtime.message || 'LoRA training အတွက် CUDA GPU နှင့် training packages လိုအပ်ပါသည်။')}</span>
              </div>
            </div>
            {!runtime.ready && <div className="runtime-details">
              {runtime.missing_dependencies?.length > 0 && <span>မရှိသော packages: {runtime.missing_dependencies.join(', ')}</span>}
              {!runtime.cuda_available && <span>NVIDIA GPU နှင့် CUDA သုံးနိုင်သော training ပတ်ဝန်းကျင်ကို ချိတ်ဆက်ပြီးနောက် ဒီနေရာမှ ပြန်စစ်ပါ။</span>}
            </div>}
          </div>
          <div className="train-controls">
            <label>Base model<select value={baseModel} onChange={e => setBaseModel(e.target.value)} disabled={activeTask}><option value="Qwen/Qwen2.5-3B-Instruct">Qwen 2.5 · 3B (မှတ်ဉာဏ်နည်းသုံး)</option><option value="Qwen/Qwen2.5-7B-Instruct">Qwen 2.5 · 7B</option></select></label>
            <label>သင်ကြားမည့် အကြိမ်<select value={epochs} onChange={e => setEpochs(Number(e.target.value))} disabled={activeTask}><option value={1}>၁ ကြိမ်</option><option value={2}>၂ ကြိမ်</option><option value={3}>၃ ကြိမ်</option><option value={4}>၄ ကြိမ်</option><option value={5}>၅ ကြိမ်</option></select></label>
            <button className="fine-primary train-button" disabled={!canTrain || !!busy} onClick={() => runAction('train', '/api/fine-tuning/train', { base_model: baseModel, epochs }, 'Model သင်ကြားမှု စတင်လိုက်ပါပြီ။')}>
              {busy === 'train' || activeTask ? <LoaderCircle className="spin" size={16}/> : <Play size={16} fill="currentColor"/>} {activeTask ? 'လုပ်ဆောင်နေသည်…' : 'သင်ကြားမှု စတင်မည်'}
            </button>
          </div>
          {!examplesReady && <p className="disabled-reason">အရင်အဆင့်တွင် အတည်ပြုထားသော အနည်းဆုံး {minExamples} ခု လိုအပ်ပါသည်။</p>}
        </section>

        <section className="fine-card">
          <div className="fine-card-head">
            <span className="step-no">၄</span>
            <div><h2>Ollama သို့ ထည့်ပြီး အသုံးပြုပါ</h2><p>Model တည်ဆောက်ပြီးမှ Ollama တွင် ထည့်သွင်းကာ fallback model အဖြစ် ဖွင့်နိုင်ပါသည်။</p></div>
          </div>
          <div className="action-strip install-strip">
            <div className="action-explain"><GraduationCap size={20}/><div><strong>{status?.trained_model_name || 'myanmar-law-finetuned'}</strong><span>{status?.ollama_available ? `Ollama တွင် လက်ရှိ model: ${status.ollama_model}` : 'Backend စက်တွင် Ollama မတွေ့ပါ။'}</span></div></div>
            <div className="button-group">
              <button className="fine-secondary" disabled={!!busy || activeTask || taskState !== 'trained' || !status?.ollama_available} onClick={() => runAction('install', '/api/fine-tuning/install', {}, 'Ollama တွင် model ထည့်သွင်းမှု စတင်ပါပြီ။')}>
                {busy === 'install' ? <LoaderCircle className="spin" size={16}/> : <FileText size={16}/>} Ollama ထဲထည့်မည်
              </button>
              <button className="fine-primary" disabled={!!busy || activeTask || taskState !== 'installed'} onClick={() => runAction('activate', '/api/fine-tuning/activate', {}, 'Fine-tuned model ကို fallback အဖြစ် ဖွင့်ထားပါပြီ။')}>
                {busy === 'activate' ? <LoaderCircle className="spin" size={16}/> : <Check size={16}/>} Fallback အဖြစ်သုံးမည်
              </button>
            </div>
          </div>
        </section>
      </div>

      <section className="fine-card task-card">
        <div className="fine-card-head task-head">
          <span className="step-no"><RefreshCw size={15}/></span>
          <div><h2>နောက်ခံလုပ်ငန်းအခြေအနေ</h2><p>{status?.task?.message || 'လက်ရှိလုပ်ဆောင်နေသော သင်ကြားမှုနှင့် model တည်ဆောက်မှုအခြေအနေ။'}</p></div>
          <span className={`task-badge task-${taskState}`}>{({idle:'မစရသေး',running:'လုပ်ဆောင်နေ',installing:'ထည့်သွင်းနေ',trained:'ပြီးပါပြီ',installed:'Ollama ထဲရောက်ပြီ',failed:'မအောင်မြင်',interrupted:'ရပ်တန့်သွား'})[taskState] || taskState}</span>
        </div>
        {status?.task?.started_at && <div className="task-meta">စတင်ချိန်: {formatDate(status.task.started_at)}{status.task.finished_at ? ` · ပြီးဆုံးချိန်: ${formatDate(status.task.finished_at)}` : ''}</div>}
        {status?.log && <details className="task-log"><summary>လုပ်ငန်းအသေးစိတ် မှတ်တမ်း</summary><pre>{status.log}</pre></details>}
      </section>

      <div className="fine-footer"><ShieldCheck size={15}/> အတည်ပြုပြီးသော QA များကိုသာ dataset ထဲ ထည့်ပါမည်။ မူရင်းစာသားမထောက်ခံသော အဖြေများကို သိမ်းဆည်းခွင့်မပြုပါ။</div>
    </section>
  );
}

export default FineTuningPanel;
