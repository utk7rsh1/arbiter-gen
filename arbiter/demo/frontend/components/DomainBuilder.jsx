// DomainBuilder.jsx — Step 1: Describe → Step 2: Edit nodes → Step 3: Launch
(function () {
  const { useState, useRef } = React;

  const EXAMPLE_PROMPTS = [
    "A hiring AI that screens software engineering resumes",
    "A healthcare triage system that prioritizes ER patients",
    "A parole board AI that predicts recidivism risk",
    "An insurance claims AI that approves or denies payouts",
    "A college admissions AI that evaluates applicants",
  ];

  const FEATURE_TYPES = ['explicit', 'proxy', 'hidden'];
  const TYPE_META = {
    explicit: { label: 'Explicit Feature', color: 'var(--green)', bg: 'rgba(16,185,129,0.10)' },
    proxy:    { label: 'Proxy Feature',    color: 'var(--amber)', bg: 'rgba(245,158,11,0.10)' },
    hidden:   { label: 'Hidden Feature',   color: 'var(--red)',   bg: 'rgba(239,68,68,0.10)'  },
  };

  // ─── Step 1: Description input ────────────────────────────────────────────
  function DescribeStep({ onGenerate, loading, error }) {
    const [desc, setDesc] = useState('');
    const [seed, setSeed] = useState(42);
    const taRef = useRef();

    const handleExample = (ex) => { setDesc(ex); taRef.current && taRef.current.focus(); };
    const canSubmit = desc.trim().length > 10 && !loading;

    return (
      <div style={{ maxWidth: 660, margin: '0 auto', width: '100%' }}>
        <div style={{ marginBottom: 32 }}>
          <h2 style={{ fontFamily: 'var(--font-display)', fontWeight: 800, fontSize: 26, color: 'var(--text-primary)', marginBottom: 8 }}>
            Describe the AI system
          </h2>
          <p style={{ fontFamily: 'var(--font-body)', fontSize: 14, color: 'var(--text-secondary)', lineHeight: 1.6 }}>
            Write one or two sentences about the AI decision system you want to audit. Be specific about what it decides, who it evaluates, and what data it uses.
          </p>
        </div>

        {/* Example chips */}
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, marginBottom: 16 }}>
          {EXAMPLE_PROMPTS.map(ex => (
            <button key={ex} onClick={() => handleExample(ex)} style={{
              padding: '5px 14px',
              background: 'var(--bg-card)',
              border: '1px solid var(--border-light)',
              borderRadius: 999,
              fontFamily: 'var(--font-body)',
              fontSize: 12,
              color: 'var(--text-secondary)',
              cursor: 'pointer',
              transition: 'all 0.15s',
              whiteSpace: 'nowrap',
            }}
            onMouseEnter={e => e.currentTarget.style.borderColor = 'var(--cyan)'}
            onMouseLeave={e => e.currentTarget.style.borderColor = 'var(--border-light)'}
            >
              {ex}
            </button>
          ))}
        </div>

        {/* Textarea */}
        <textarea
          ref={taRef}
          value={desc}
          onChange={e => setDesc(e.target.value)}
          placeholder="e.g. A hiring AI that screens software engineering candidates and decides whether to advance them to an interview..."
          rows={4}
          style={{
            width: '100%',
            padding: '16px 18px',
            background: 'var(--bg-card)',
            border: '1.5px solid var(--border-light)',
            borderRadius: 14,
            fontFamily: 'var(--font-body)',
            fontSize: 14,
            color: 'var(--text-primary)',
            resize: 'vertical',
            outline: 'none',
            boxShadow: 'var(--shadow-card)',
            lineHeight: 1.6,
            transition: 'border-color 0.15s',
          }}
          onFocus={e => e.currentTarget.style.borderColor = 'var(--cyan)'}
          onBlur={e => e.currentTarget.style.borderColor = 'var(--border-light)'}
        />

        {/* Seed + submit row */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginTop: 16 }}>
          <span style={{ fontFamily: 'var(--font-mono)', fontSize: 12, color: 'var(--text-muted)' }}>Seed:</span>
          <input type="number" value={seed} onChange={e => setSeed(parseInt(e.target.value) || 42)}
            style={{ width: 70, padding: '6px 10px', fontFamily: 'var(--font-mono)', fontSize: 12, background: 'var(--bg-card)', border: '1px solid var(--border-light)', borderRadius: 999, color: 'var(--text-primary)', outline: 'none', textAlign: 'center' }}
          />
          <div style={{ flex: 1 }} />
          {error && <span style={{ fontFamily: 'var(--font-body)', fontSize: 12, color: 'var(--red)' }}>{error}</span>}
          <button
            onClick={() => onGenerate(desc.trim(), seed)}
            disabled={!canSubmit}
            style={{
              padding: '10px 28px',
              background: canSubmit ? 'var(--cyan)' : 'var(--border-light)',
              color: canSubmit ? '#000' : 'var(--text-muted)',
              border: 'none',
              borderRadius: 999,
              fontFamily: 'var(--font-body)',
              fontWeight: 600,
              fontSize: 13,
              cursor: canSubmit ? 'pointer' : 'not-allowed',
              display: 'flex',
              alignItems: 'center',
              gap: 8,
              transition: 'all 0.2s',
            }}
          >
            {loading ? <><div className="spinner" />Generating...</> : '⚡ Generate with Groq →'}
          </button>
        </div>
      </div>
    );
  }

  // ─── Step 2: Node editor ──────────────────────────────────────────────────
  function NodeEditor({ domainJson, onBack, onLaunch, level, setLevel, seed, setSeed }) {
    const [doc, setDoc] = useState(domainJson);

    const allFeatures = [
      ...doc.explicit_features.map(f => ({ ...f, _type: 'explicit' })),
      ...doc.proxy_features.map(f => ({ ...f, _type: 'proxy' })),
      ...doc.hidden_features.map(f => ({ ...f, _type: 'hidden' })),
    ];

    const updateFeatureName = (type, idx, newName) => {
      const key = `${type}_features`;
      const arr = [...doc[key]];
      arr[idx] = { ...arr[idx], name: newName };
      setDoc({ ...doc, [key]: arr });
    };

    const updateFeatureDesc = (type, idx, newDesc) => {
      const key = `${type}_features`;
      const arr = [...doc[key]];
      arr[idx] = { ...arr[idx], description: newDesc };
      setDoc({ ...doc, [key]: arr });
    };

    return (
      <div style={{ maxWidth: 860, margin: '0 auto', width: '100%' }}>
        {/* Header */}
        <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', marginBottom: 28 }}>
          <div>
            <h2 style={{ fontFamily: 'var(--font-display)', fontWeight: 800, fontSize: 24, color: 'var(--text-primary)', marginBottom: 6 }}>
              Review &amp; Edit Nodes
            </h2>
            <p style={{ fontFamily: 'var(--font-body)', fontSize: 13, color: 'var(--text-secondary)' }}>
              Domain: <strong>{doc.domain_name}</strong> &nbsp;·&nbsp;
              Outcomes: <span style={{ color: 'var(--green)' }}>{doc.positive_outcome}</span> / <span style={{ color: 'var(--red)' }}>{doc.negative_outcome}</span>
            </p>
          </div>
          <button onClick={onBack} style={{ padding: '7px 16px', background: 'none', border: '1px solid var(--border-light)', borderRadius: 999, fontFamily: 'var(--font-body)', fontSize: 12, color: 'var(--text-secondary)', cursor: 'pointer' }}>
            ← Back
          </button>
        </div>

        {/* Feature groups */}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 16, marginBottom: 24 }}>
          {FEATURE_TYPES.map(ftype => {
            const key = `${ftype}_features`;
            const features = doc[key] || [];
            const meta = TYPE_META[ftype];
            return (
              <div key={ftype} style={{ background: 'var(--bg-card)', borderRadius: 14, padding: 16, border: '1px solid var(--border-light)', boxShadow: 'var(--shadow-card)' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 14 }}>
                  <span style={{ display: 'inline-block', width: 10, height: 10, borderRadius: '50%', background: meta.color }} />
                  <span style={{ fontFamily: 'var(--font-mono)', fontSize: 11, fontWeight: 600, color: meta.color, letterSpacing: '0.06em', textTransform: 'uppercase' }}>{meta.label}s</span>
                  <span style={{ marginLeft: 'auto', fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--text-muted)', background: meta.bg, padding: '2px 8px', borderRadius: 999 }}>{features.length}</span>
                </div>
                <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                  {features.map((f, i) => (
                    <div key={i} style={{ background: 'var(--bg-shell)', borderRadius: 10, padding: '10px 12px' }}>
                      <input
                        value={f.name}
                        onChange={e => updateFeatureName(ftype, i, e.target.value)}
                        style={{ width: '100%', background: 'transparent', border: 'none', outline: 'none', fontFamily: 'var(--font-mono)', fontSize: 12, color: 'var(--text-primary)', fontWeight: 500, marginBottom: 4 }}
                      />
                      <input
                        value={f.description || ''}
                        onChange={e => updateFeatureDesc(ftype, i, e.target.value)}
                        style={{ width: '100%', background: 'transparent', border: 'none', outline: 'none', fontFamily: 'var(--font-body)', fontSize: 11, color: 'var(--text-muted)' }}
                        placeholder="description..."
                      />
                    </div>
                  ))}
                </div>
              </div>
            );
          })}
        </div>

        {/* Anomaly context */}
        <div style={{ background: 'var(--bg-card)', borderRadius: 14, padding: 16, border: '1px solid var(--border-light)', marginBottom: 24, boxShadow: 'var(--shadow-card)' }}>
          <div style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--text-secondary)', letterSpacing: '0.08em', textTransform: 'uppercase', marginBottom: 10 }}>Anomaly Context</div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
            {[
              ['Discriminated Feature', doc.discriminated_group_feature],
              ['Discriminated Value', doc.discriminated_group_value],
              ['Threshold Feature', doc.approval_threshold_feature],
              ['Regulation (L6)', doc.drift_regulation_name],
            ].map(([label, val]) => (
              <div key={label} style={{ background: 'var(--bg-shell)', borderRadius: 8, padding: '8px 12px' }}>
                <div style={{ fontFamily: 'var(--font-mono)', fontSize: 10, color: 'var(--text-muted)', marginBottom: 3 }}>{label}</div>
                <div style={{ fontFamily: 'var(--font-mono)', fontSize: 12, color: 'var(--cyan)' }}>{val || '—'}</div>
              </div>
            ))}
          </div>
          {doc.anomaly_description && (
            <div style={{ marginTop: 12, fontFamily: 'var(--font-body)', fontSize: 12, color: 'var(--text-secondary)', lineHeight: 1.5, padding: '8px 12px', background: 'rgba(245,158,11,0.06)', borderRadius: 8, borderLeft: '3px solid var(--amber)' }}>
              {doc.anomaly_description}
            </div>
          )}
        </div>

        {/* Launch controls */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <span style={{ fontFamily: 'var(--font-mono)', fontSize: 12, color: 'var(--text-muted)' }}>Level:</span>
          <div style={{ display: 'flex', gap: 4 }}>
            {[1,2,3,4,5,6,7].map(l => (
              <button key={l} onClick={() => setLevel(l)} style={{
                width: 32, height: 32,
                borderRadius: 8,
                border: 'none',
                fontFamily: 'var(--font-mono)',
                fontSize: 12,
                cursor: 'pointer',
                background: level === l ? 'var(--cyan)' : 'var(--bg-card)',
                color: level === l ? '#000' : 'var(--text-secondary)',
                boxShadow: 'var(--shadow-card)',
                fontWeight: level === l ? 700 : 400,
              }}>{l}</button>
            ))}
          </div>
          <span style={{ fontFamily: 'var(--font-mono)', fontSize: 12, color: 'var(--text-muted)', marginLeft: 8 }}>Seed:</span>
          <input type="number" value={seed} onChange={e => setSeed(parseInt(e.target.value) || 42)}
            style={{ width: 70, padding: '6px 10px', fontFamily: 'var(--font-mono)', fontSize: 12, background: 'var(--bg-card)', border: '1px solid var(--border-light)', borderRadius: 999, color: 'var(--text-primary)', outline: 'none', textAlign: 'center' }}
          />
          <div style={{ flex: 1 }} />
          <button onClick={() => onLaunch(doc)} style={{
            padding: '11px 32px',
            background: 'var(--cyan)',
            color: '#000',
            border: 'none',
            borderRadius: 999,
            fontFamily: 'var(--font-body)',
            fontWeight: 700,
            fontSize: 14,
            cursor: 'pointer',
          }}>
            🚀 Launch Audit →
          </button>
        </div>
      </div>
    );
  }

  // ─── Main DomainBuilder orchestrator ─────────────────────────────────────
  window.DomainBuilder = function DomainBuilder({ onLaunch, onBack }) {
    const [step, setStep] = useState('describe'); // 'describe' | 'edit'
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState('');
    const [domainJson, setDomainJson] = useState(null);
    const [level, setLevel] = useState(3);
    const [seed, setSeed] = useState(42);

    const handleGenerate = async (desc, seedVal) => {
      setLoading(true);
      setError('');
      try {
        const res = await fetch('/generate-domain', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ description: desc, seed: seedVal }),
        });
        if (!res.ok) {
          const err = await res.json();
          throw new Error(err.detail || 'Generation failed');
        }
        const data = await res.json();
        setDomainJson(data);
        setSeed(seedVal);
        setStep('edit');
      } catch (e) {
        setError(e.message);
      } finally {
        setLoading(false);
      }
    };

    const handleLaunch = (editedDomain) => {
      onLaunch({ domainJson: editedDomain, level, seed });
    };

    return (
      <div style={{ height: '100vh', display: 'flex', flexDirection: 'column', background: 'var(--bg-shell)' }}>
        {/* Header bar */}
        <div style={{ height: 56, background: 'var(--bg-card)', borderBottom: '1px solid var(--border-light)', display: 'flex', alignItems: 'center', padding: '0 28px', gap: 16, boxShadow: '0 1px 3px rgba(0,0,0,0.06)', flexShrink: 0 }}>
          <button onClick={onBack} style={{ background: 'none', border: 'none', cursor: 'pointer', fontFamily: 'var(--font-mono)', fontSize: 12, color: 'var(--text-muted)', padding: '4px 0' }}>← Landing</button>
          <div style={{ width: 1, height: 18, background: 'var(--border-light)' }} />
          <span style={{ fontFamily: 'var(--font-display)', fontWeight: 800, fontSize: 16, letterSpacing: '0.06em', color: 'var(--text-primary)' }}>ARBITER</span>
          <span style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--cyan)' }}>DOMAIN BUILDER</span>
          <div style={{ flex: 1 }} />
          {/* Step indicator */}
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            {['DESCRIBE', 'EDIT NODES', 'LAUNCH'].map((s, i) => {
              const active = i === (step === 'describe' ? 0 : 1);
              const done = i < (step === 'describe' ? 0 : 1);
              return (
                <React.Fragment key={s}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                    <div style={{ width: 22, height: 22, borderRadius: '50%', background: done ? 'var(--green)' : active ? 'var(--cyan)' : 'var(--border-light)', display: 'flex', alignItems: 'center', justifyContent: 'center', fontFamily: 'var(--font-mono)', fontSize: 10, color: active || done ? '#000' : 'var(--text-muted)', fontWeight: 700 }}>
                      {done ? '✓' : i + 1}
                    </div>
                    <span style={{ fontFamily: 'var(--font-mono)', fontSize: 10, color: active ? 'var(--text-primary)' : 'var(--text-muted)', letterSpacing: '0.06em' }}>{s}</span>
                  </div>
                  {i < 2 && <div style={{ width: 20, height: 1, background: 'var(--border-light)' }} />}
                </React.Fragment>
              );
            })}
          </div>
        </div>

        {/* Content */}
        <div style={{ flex: 1, overflowY: 'auto', padding: '48px 24px' }}>
          {step === 'describe' && (
            <DescribeStep onGenerate={handleGenerate} loading={loading} error={error} />
          )}
          {step === 'edit' && domainJson && (
            <NodeEditor
              domainJson={domainJson}
              onBack={() => setStep('describe')}
              onLaunch={handleLaunch}
              level={level} setLevel={setLevel}
              seed={seed} setSeed={setSeed}
            />
          )}
        </div>
      </div>
    );
  };
})();
