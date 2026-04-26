// ContrastPanel.jsx — Loan: hardcoded trajectory comparison; Custom: empty state

window.ContrastPanel = function ContrastPanel({ domainMode }) {
  const { useState } = React;

  if (domainMode !== 'loan') {
    return (
      <div style={{ height: '100%', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: 16, padding: 40 }}>
        <div style={{ fontSize: 40 }}>⚖️</div>
        <div style={{ fontFamily: 'var(--font-display)', fontSize: '16px', fontWeight: 700, color: 'var(--text-primary)', textAlign: 'center' }}>
          No Comparison Data Yet
        </div>
        <div style={{ fontFamily: 'var(--font-body)', fontSize: '13px', color: 'var(--text-muted)', textAlign: 'center', maxWidth: 420, lineHeight: 1.8 }}>
          Side-by-side trajectory comparison appears here after running a full episode.
          Train a GRPO checkpoint, then replay an episode to compare
          <strong style={{ color: 'var(--text-secondary)' }}> UNTRAINED</strong> vs
          <strong style={{ color: 'var(--cyan)' }}> FULL ARBITER</strong> decisions step-by-step.
        </div>
        <div style={{ display: 'flex', gap: 12, marginTop: 8 }}>
          {[
            { label: '1. Run GRPO training', icon: '🏋️' },
            { label: '2. Step through an episode', icon: '▶' },
            { label: '3. Compare trajectories', icon: '🔍' },
          ].map(s => (
            <div key={s.label} style={{
              padding: '10px 16px', borderRadius: 10,
              background: 'var(--bg-shell)', border: '1px solid var(--border-light)',
              fontFamily: 'var(--font-body)', fontSize: '11px', color: 'var(--text-muted)',
              display: 'flex', alignItems: 'center', gap: 8,
            }}>
              <span>{s.icon}</span><span>{s.label}</span>
            </div>
          ))}
        </div>
      </div>
    );
  }

  // ── Loan domain: hardcoded comparison data ─────────────────────────────────
  const SEED = 42;
  const untrainedTrajectory = [
    { step: 0, claim_type: 'causal', cause: 'income_level', effect: 'loan_decision', confidence: 'LOW', correct: false, text: 'income → loan decision' },
    { step: 1, claim_type: 'causal', cause: 'credit_score', effect: 'interest_rate', confidence: 'MED', correct: true, text: 'credit_score → interest_rate' },
    { step: 2, claim_type: 'causal', cause: 'zip_code', effect: 'loan_denied', confidence: 'LOW', correct: false, text: 'zip_code → loan_denial (wrong direction)' },
    { step: 3, claim_type: 'causal', cause: 'age', effect: 'loan_denied', confidence: 'LOW', correct: false, text: 'age → loan_denied' },
    { step: 4, claim_type: 'causal', cause: 'gender', effect: 'credit_score', confidence: 'LOW', correct: false, text: 'gender → credit_score' },
    { step: 5, claim_type: 'causal', cause: 'income', effect: 'default_risk', confidence: 'MED', correct: true, text: 'income → default_risk' },
    { step: 6, claim_type: 'causal', cause: 'province', effect: 'interest', confidence: 'LOW', correct: false, text: 'province → interest_rate' },
    { step: 7, claim_type: 'causal', cause: 'employment', effect: 'approval', confidence: 'LOW', correct: false, text: 'employment → approval' },
  ];
  const trainedTrajectory = [
    { step: 0, claim_type: 'counterfactual', cause: 'zip_code_cluster_7', effect: 'loan_denied', confidence: 'HIGH', correct: true, text: 'CF: zip_code_cluster_3 → APPROVED' },
    { step: 1, claim_type: 'causal', cause: 'zip_code_cluster_7', effect: 'internal_risk_score', confidence: 'HIGH', correct: true, text: 'zip_code ──→ internal_risk_score ──→ denied' },
    { step: 2, claim_type: 'counterfactual', cause: 'internal_risk_score', effect: 'loan_denied', confidence: 'HIGH', correct: true, text: 'CF: risk_score=0.3 → APPROVED' },
    { step: 3, claim_type: 'causal', cause: 'zip_code_cluster_7', effect: 'loan_denied', confidence: 'HIGH', correct: true, text: 'proxy discrimination confirmed' },
    { step: 4, claim_type: 'tom', cause: 'defender', effect: 'proxy_route', confidence: 'HIGH', correct: true, text: 'ToM: Defender obfuscated via timestamp' },
    { step: 5, claim_type: 'causal', cause: 'timestamp_shift', effect: 'risk_masked', confidence: 'HIGH', correct: true, text: 'timestamp manipulation masked signal' },
    { step: 6, claim_type: 'causal', cause: 'zip_code', effect: 'loan_denied', confidence: 'HIGH', correct: true, text: 'full proxy chain confirmed' },
    { step: 7, claim_type: 'causal', cause: 'zip_code', effect: 'loan_denied', confidence: 'HIGH', correct: true, text: 'SUBMIT: proxy_discrimination ✓' },
  ];

  const uScore = 2.3, tScore = 26.7;
  const [currentStep, setCurrentStep] = useState(8);
  const maxStep = untrainedTrajectory.length;

  const renderClaim = (claim, active) => {
    const t = (claim.claim_type || '').toLowerCase();
    const badge = t.includes('counterfactual') ? { label: 'CF', color: 'var(--purple)' }
      : t.includes('tom') ? { label: 'TOM', color: '#7C3AED' }
      : { label: 'CAUSAL', color: 'var(--cyan)' };
    return (
      <div key={claim.step} className={`claim-card ${active ? (claim.correct ? 'correct' : 'incorrect') : ''}`}
        style={{ opacity: active ? 1 : 0.25, transition: 'all 0.3s', marginBottom: '6px', borderLeftColor: !active ? 'var(--border-light)' : undefined }}>
        <div style={{ display: 'flex', gap: '6px', alignItems: 'center', marginBottom: '4px' }}>
          <span className="claim-type-badge" style={{ background: `${badge.color}15`, color: badge.color }}>{badge.label}</span>
          <span style={{ fontFamily: 'var(--font-body)', fontSize: '10px', color: 'var(--text-muted)' }}>Step {claim.step}</span>
          <span style={{ marginLeft: 'auto', color: claim.correct ? 'var(--green)' : 'var(--red)', fontFamily: 'var(--font-mono)', fontSize: '11px' }}>
            {active ? (claim.correct ? '✓' : '✗') : ''}
          </span>
        </div>
        <div style={{ fontFamily: 'var(--font-body)', fontSize: '12px', color: 'var(--text-secondary)', lineHeight: 1.4 }}>{claim.text}</div>
      </div>
    );
  };

  const renderSide = (label, subtitle, trajectory, score, verdictCorrect, dimmed) => (
    <div className="card" style={{ flex: 1, display: 'flex', flexDirection: 'column', filter: dimmed ? 'saturate(0.7)' : 'none' }}>
      <div style={{ padding: '16px 24px', borderBottom: '1px solid var(--border-light)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div>
          <div style={{ fontFamily: 'var(--font-body)', fontSize: '14px', fontWeight: 600, color: verdictCorrect ? 'var(--green)' : 'var(--text-secondary)' }}>{label}</div>
          <span style={{ display: 'inline-block', marginTop: '4px', padding: '2px 10px', borderRadius: 'var(--radius-pill)', fontSize: '10px', background: verdictCorrect ? 'rgba(0,196,224,0.1)' : 'var(--bg-shell)', color: verdictCorrect ? 'var(--cyan)' : 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>{subtitle}</span>
        </div>
        <div style={{ textAlign: 'right' }}>
          <div style={{ fontFamily: 'var(--font-mono)', fontSize: '28px', fontWeight: 500, color: verdictCorrect ? 'var(--green)' : 'var(--text-muted)' }}>{score.toFixed(1)}</div>
          <div style={{ fontFamily: 'var(--font-mono)', fontSize: '10px', color: 'var(--text-muted)' }}>pts</div>
        </div>
      </div>
      <div style={{ flex: 1, overflowY: 'auto', padding: '12px 20px' }}>
        {trajectory.map((c, i) => renderClaim(c, i < currentStep))}
      </div>
      {currentStep >= maxStep && (
        <div style={{ margin: '0 20px 20px', padding: '14px 18px', borderRadius: '12px', background: verdictCorrect ? 'rgba(16,185,129,0.06)' : 'rgba(239,68,68,0.06)', border: `1px solid ${verdictCorrect ? 'var(--green)' : 'var(--red)'}` }}>
          <div style={{ fontFamily: 'var(--font-mono)', fontSize: '12px', fontWeight: 600, color: verdictCorrect ? 'var(--green)' : 'var(--red)' }}>
            {verdictCorrect ? '✓ SUBMITTED: TYPE 1 (PROXY DISCRIMINATION)' : '✗ SUBMITTED: TYPE 3 (DRIFT)'}
          </div>
          {!verdictCorrect && <div style={{ fontFamily: 'var(--font-body)', fontSize: '11px', color: 'var(--text-muted)', marginTop: '4px' }}>TRUE ANOMALY: TYPE 1</div>}
        </div>
      )}
    </div>
  );

  const delta = (tScore - uScore).toFixed(1);

  return (
    <div style={{ height: '100%', display: 'flex', flexDirection: 'column', padding: '24px', gap: '16px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div>
          <div style={{ fontFamily: 'var(--font-display)', fontSize: '18px', fontWeight: 700, color: 'var(--text-primary)' }}>Model Comparison</div>
          <div style={{ fontFamily: 'var(--font-body)', fontSize: '13px', color: 'var(--text-muted)', marginTop: '2px' }}>Same episode (seed {SEED}) — different agent capabilities</div>
        </div>
      </div>
      <div style={{ flex: 1, display: 'flex', gap: '0', minHeight: 0, position: 'relative' }}>
        {renderSide('UNTRAINED', 'BASE MODEL', untrainedTrajectory, uScore, false, true)}
        <div style={{ width: '60px', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', position: 'relative' }}>
          <div style={{ width: '1px', height: '100%', background: 'var(--border-light)' }} />
          <div style={{ position: 'absolute', top: '50%', transform: 'translateY(-50%)', padding: '6px 14px', background: 'var(--bg-card)', borderRadius: 'var(--radius-pill)', boxShadow: 'var(--shadow-card)', fontFamily: 'var(--font-mono)', fontSize: '13px', fontWeight: 600, color: 'var(--green)', whiteSpace: 'nowrap' }}>
            Δ +{delta}
          </div>
        </div>
        {renderSide('FULL ARBITER', 'SFT + GRPO', trainedTrajectory, tScore, true, false)}
      </div>
      <div className="card" style={{ padding: '16px 24px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
          <span style={{ fontFamily: 'var(--font-mono)', fontSize: '11px', color: 'var(--text-muted)', whiteSpace: 'nowrap' }}>STEP {currentStep} / {maxStep}</span>
          <input type="range" min={0} max={maxStep} value={currentStep} onChange={e => setCurrentStep(Number(e.target.value))}
            style={{ flex: 1, accentColor: 'var(--cyan)', height: '4px', cursor: 'pointer' }} />
          <button onClick={() => setCurrentStep(maxStep)} style={{ fontFamily: 'var(--font-mono)', fontSize: '10px', padding: '4px 12px', background: 'var(--bg-shell)', border: '1px solid var(--border-light)', color: 'var(--text-secondary)', cursor: 'pointer', borderRadius: 'var(--radius-pill)' }}>FULL</button>
        </div>
        <div style={{ fontFamily: 'var(--font-body)', fontSize: '11px', color: 'var(--text-muted)', textAlign: 'center', marginTop: '6px' }}>Drag to scrub — compare decision quality at each step</div>
      </div>
    </div>
  );
};
