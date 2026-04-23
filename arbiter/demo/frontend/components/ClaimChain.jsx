// ClaimChain.jsx — White card with scrollable claim feed, type filters, animated entry & verdict reveal

window.ClaimChain = function ClaimChain({ claims, overseers }) {
  const { useRef, useEffect, useState } = React;
  const bottomRef = useRef(null);
  const [filters, setFilters] = useState({ cf: true, causal: true, tom: true });

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [claims, overseers]);

  const toggleFilter = (key) => setFilters(prev => ({ ...prev, [key]: !prev[key] }));

  const getClaimType = (claim) => {
    const t = (claim.claim_type || '').toLowerCase();
    if (t.includes('counterfactual')) return 'counterfactual';
    if (t.includes('theory') || t.includes('tom')) return 'tom';
    return 'causal';
  };

  const isVisible = (claim) => {
    const t = getClaimType(claim);
    if (t === 'counterfactual') return filters.cf;
    if (t === 'tom') return filters.tom;
    return filters.causal;
  };

  const getBadge = (type) => {
    if (type === 'counterfactual') return <span className="claim-type-badge badge-counterfactual">COUNTERFACTUAL</span>;
    if (type === 'tom') return <span className="claim-type-badge badge-tom">THEORY-OF-MIND</span>;
    return <span className="claim-type-badge badge-causal">CAUSAL</span>;
  };

  const renderVerification = (v) => {
    if (!v || typeof v !== 'object') return null;
    return (
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px', marginTop: '8px', paddingTop: '8px', borderTop: '1px solid var(--border-light)' }}>
        {Object.entries(v).map(([k, val]) => {
          if (typeof val !== 'boolean') return null;
          return (
            <span key={k} style={{
              display: 'inline-flex', alignItems: 'center', gap: '3px',
              fontFamily: 'var(--font-mono)', fontSize: '10px',
              color: val ? 'var(--green)' : 'var(--red)',
            }}>
              {val ? '✓' : '✗'} {k.replace(/_/g, ' ')}
            </span>
          );
        })}
      </div>
    );
  };

  const renderClaimCard = (claim, idx) => {
    const type = getClaimType(claim);
    const statusClass = claim.correct === true ? 'correct' : claim.correct === false ? 'incorrect' : 'pending';

    return (
      <div key={`claim-${idx}`} className={`claim-card ${statusClass} slide-in`}
        style={{ animationDelay: `${idx * 0.06}s` }}
      >
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '8px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            {getBadge(type)}
            <span style={{ fontFamily: 'var(--font-body)', fontSize: '11px', color: 'var(--text-muted)' }}>Step {claim.step ?? idx + 1}</span>
          </div>
          <span style={{
            fontFamily: 'var(--font-mono)', fontSize: '10px',
            color: claim.confidence === 'HIGH' ? 'var(--cyan)' : 'var(--text-muted)',
            display: 'flex', alignItems: 'center', gap: '4px',
          }}>
            {claim.confidence || 'MED'}
            <span style={{
              width: 6, height: 6, borderRadius: '50%',
              background: claim.confidence === 'HIGH' ? 'var(--cyan)' : 'var(--text-muted)',
            }} />
          </span>
        </div>

        {/* Causal claim body */}
        {claim.cause_feature && type === 'causal' && (
          <div style={{ fontFamily: 'var(--font-body)', fontSize: '13px', marginBottom: '4px' }}>
            <span style={{ color: 'var(--cyan)', fontWeight: 500 }}>{claim.cause_feature}</span>
            <span style={{ color: 'var(--text-muted)', margin: '0 6px' }}>──→</span>
            <span style={{ color: 'var(--text-primary)', fontWeight: 500 }}>{claim.effect_feature || '?'}</span>
            {claim.mechanism && (
              <div style={{ color: 'var(--text-muted)', fontSize: '11px', marginTop: '2px' }}>
                via <span style={{ color: 'var(--text-secondary)' }}>{claim.mechanism}</span>
              </div>
            )}
            {claim.basis_records && claim.basis_records.length > 0 && (
              <div style={{ color: 'var(--text-muted)', fontSize: '11px', marginTop: '2px' }}>
                Evidence: {claim.basis_records.slice(0, 4).join(', ')}
                {claim.basis_records.length > 4 && ' …'}
              </div>
            )}
          </div>
        )}

        {/* Counterfactual layout */}
        {type === 'counterfactual' && (
          <div style={{ marginBottom: '4px' }}>
            <div style={{ fontFamily: 'var(--font-body)', fontSize: '13px', fontStyle: 'italic', color: 'var(--text-secondary)', marginBottom: '8px' }}>
              "What if {claim.record_id || 'record'} had {claim.feature_id} = {String(claim.predicted_outcome || '?')}?"
            </div>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '24px', padding: '8px 0' }}>
              <div style={{ textAlign: 'center' }}>
                <div style={{ fontFamily: 'var(--font-body)', fontSize: '10px', color: 'var(--text-muted)', marginBottom: '4px', letterSpacing: '0.08em' }}>PREDICTED</div>
                <div style={{ fontFamily: 'var(--font-mono)', fontSize: '14px', fontWeight: 500, color: 'var(--text-primary)' }}>{claim.predicted_outcome || '?'}</div>
              </div>
              <span style={{ color: 'var(--text-muted)', fontSize: '16px' }}>→</span>
              <div style={{ textAlign: 'center' }}>
                <div style={{ fontFamily: 'var(--font-body)', fontSize: '10px', color: 'var(--text-muted)', marginBottom: '4px', letterSpacing: '0.08em' }}>ACTUAL</div>
                <div style={{ fontFamily: 'var(--font-mono)', fontSize: '14px', fontWeight: 500, color: claim.correct ? 'var(--green)' : 'var(--text-primary)' }}>
                  {claim.actual_outcome || '?'}
                  {claim.correct === true && <span style={{ marginLeft: '4px', color: 'var(--green)' }}>✓</span>}
                </div>
              </div>
            </div>
            {typeof claim.reward_delta === 'number' && claim.reward_delta > 0 && (
              <div style={{ textAlign: 'center', fontFamily: 'var(--font-mono)', fontSize: '11px', color: 'var(--cyan)' }}>
                +{claim.reward_delta.toFixed(1)} pts
              </div>
            )}
          </div>
        )}

        {/* Theory of Mind */}
        {type === 'tom' && (
          <div style={{ fontFamily: 'var(--font-body)', fontSize: '13px', marginBottom: '4px' }}>
            <span style={{ color: 'var(--purple)', fontWeight: 500 }}>Defender {claim.defender_action || 'action'}</span>
            {claim.obfuscation_method && (
              <span style={{ color: 'var(--text-secondary)' }}> via {claim.obfuscation_method}</span>
            )}
            {claim.target_link && (
              <div style={{ color: 'var(--text-muted)', fontSize: '11px', marginTop: '2px' }}>
                Target: {claim.target_link}
              </div>
            )}
          </div>
        )}

        {renderVerification(claim.verification)}
      </div>
    );
  };

  const renderOverseerWarning = (warn, idx) => (
    <div key={`warn-${idx}`} className="overseer-warning slide-in">
      <div style={{ fontWeight: 600, marginBottom: '4px', display: 'flex', alignItems: 'center', gap: '6px' }}>
        ⚠ META-OVERSEER: Contradiction
        {warn.step !== undefined && <span style={{ fontWeight: 400, fontSize: '11px' }}>Step {warn.step}</span>}
      </div>
      <div style={{ color: 'var(--text-secondary)', fontSize: '12px' }}>{warn.message}</div>
      {warn.penalty !== undefined && (
        <div style={{ fontFamily: 'var(--font-mono)', fontSize: '11px', color: 'var(--red)', marginTop: '4px' }}>
          Penalty {warn.penalty > 0 ? '−' : ''}{Math.abs(warn.penalty).toFixed(1)}
        </div>
      )}
    </div>
  );

  // Interleave and sort
  const allItems = [];
  (claims || []).forEach((c, i) => {
    if (isVisible(c)) allItems.push({ type: 'claim', data: c, idx: i });
  });
  (overseers || []).forEach((w, i) => { allItems.push({ type: 'warning', data: w, idx: i }); });
  allItems.sort((a, b) => (a.data.step ?? a.idx) - (b.data.step ?? b.idx));

  const claimCount = (claims || []).length;

  return (
    <div className="card" style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
      {/* Header */}
      <div className="card-header">
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <span className="card-header-label">CLAIM CHAIN</span>
          <span style={{
            fontFamily: 'var(--font-mono)', fontSize: '11px',
            padding: '2px 10px', borderRadius: 'var(--radius-pill)',
            background: 'rgba(0,196,224,0.1)', color: 'var(--cyan)',
          }}>{claimCount} CLAIMS</span>
        </div>
        <div style={{ display: 'flex', gap: '4px' }}>
          {[
            { key: 'cf', label: 'CF', color: 'var(--purple)' },
            { key: 'causal', label: 'CAUSAL', color: 'var(--cyan)' },
            { key: 'tom', label: 'TOM', color: '#7C3AED' },
          ].map(f => (
            <button key={f.key} onClick={() => toggleFilter(f.key)} style={{
              fontFamily: 'var(--font-mono)', fontSize: '10px', padding: '2px 8px',
              borderRadius: 'var(--radius-pill)', cursor: 'pointer',
              background: filters[f.key] ? `${f.color}15` : 'transparent',
              border: `1px solid ${filters[f.key] ? f.color : 'var(--border-light)'}`,
              color: filters[f.key] ? f.color : 'var(--text-muted)',
            }}>{f.label}</button>
          ))}
        </div>
      </div>

      {/* Scrollable feed */}
      <div style={{ flex: 1, overflowY: 'auto', padding: '12px 24px 24px', display: 'flex', flexDirection: 'column', gap: '8px' }}>
        {allItems.length === 0 ? (
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '100%', gap: '12px' }}>
            <div style={{ fontSize: '32px', opacity: 0.15, color: 'var(--text-muted)' }}>⬡</div>
            <span style={{ fontFamily: 'var(--font-body)', fontSize: '13px', color: 'var(--text-muted)' }}>No claims yet — start stepping</span>
          </div>
        ) : (
          allItems.map(item =>
            item.type === 'claim' ? renderClaimCard(item.data, item.idx) : renderOverseerWarning(item.data, item.idx)
          )
        )}
        <div ref={bottomRef} />
      </div>
    </div>
  );
};
