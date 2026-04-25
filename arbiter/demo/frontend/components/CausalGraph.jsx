// CausalGraph.jsx — Dark panel with SVG-based causal graph, force-directed layout, node state animations

// ── Loan domain static fallback graph (mirrors graph.py on GitHub) ────────────
// Shown immediately; replaced by live backend data once session is ready.
const LOAN_GRAPH_NODES = [
  // Explicit features
  { id: 'credit_score',      name: 'Credit Score',           node_type: 'input',  proxy: false, hidden: false },
  { id: 'employment_status', name: 'Employment Status',      node_type: 'input',  proxy: false, hidden: false },
  { id: 'debt_ratio',        name: 'Debt-to-Income Ratio',   node_type: 'input',  proxy: false, hidden: false },
  { id: 'income',            name: 'Annual Income',          node_type: 'input',  proxy: false, hidden: false },
  { id: 'loan_amount',       name: 'Loan Amount Requested',  node_type: 'input',  proxy: false, hidden: false },
  { id: 'payment_history',   name: 'Payment History Score',  node_type: 'input',  proxy: false, hidden: false },
  // Proxy features (visible to auditor)
  { id: 'zip_code_cluster',       name: 'ZIP Code Cluster',                node_type: 'proxy',  proxy: true, hidden: false, protected: true },
  { id: 'surname_pattern',        name: 'Surname Pattern Group',           node_type: 'proxy',  proxy: true, hidden: false, protected: true },
  { id: 'neighborhood_density',   name: 'Neighborhood Commercial Density', node_type: 'proxy',  proxy: true, hidden: false, protected: true },
  // Hidden features (latent — shown dimmed)
  { id: 'internal_risk_score', name: 'Internal Risk Score',     node_type: 'hidden', proxy: false, hidden: true },
  { id: 'behavioral_score',    name: 'Behavioral Pattern Score', node_type: 'hidden', proxy: false, hidden: true },
  // Policy
  { id: 'policy_main', name: 'Main Decision Policy', node_type: 'policy' },
  // Outcomes
  { id: 'denial_rate_overall',    name: 'Overall Denial Rate',              node_type: 'outcome' },
  { id: 'approval_rate_overall',  name: 'Overall Approval Rate',            node_type: 'outcome' },
  { id: 'denial_rate_minority',   name: 'Denial Rate – Minority',           node_type: 'outcome' },
  { id: 'denial_rate_zip7',       name: 'Denial Rate – ZIP 7',              node_type: 'outcome' },
];

const LOAN_GRAPH_EDGES = [
  // Policy edges (stated)
  { source: 'credit_score',      target: 'policy_main',         edge_type: 'policy' },
  { source: 'employment_status', target: 'policy_main',         edge_type: 'policy' },
  { source: 'debt_ratio',        target: 'policy_main',         edge_type: 'policy' },
  { source: 'credit_score',      target: 'denial_rate_overall', edge_type: 'policy' },
  { source: 'employment_status', target: 'denial_rate_overall', edge_type: 'policy' },
  { source: 'debt_ratio',        target: 'denial_rate_overall', edge_type: 'policy' },
  // Benign causal
  { source: 'credit_score',      target: 'approval_rate_overall', edge_type: 'causal' },
  { source: 'debt_ratio',        target: 'denial_rate_overall',   edge_type: 'causal' },
  { source: 'employment_status', target: 'approval_rate_overall', edge_type: 'causal' },
  // Anomaly chain (Type 1: proxy discrimination — shown to auditor via judge view)
  { source: 'zip_code_cluster',  target: 'internal_risk_score',  edge_type: 'causal', anomalous: true },
  { source: 'internal_risk_score', target: 'denial_rate_overall', edge_type: 'causal', anomalous: true },
  { source: 'internal_risk_score', target: 'denial_rate_minority', edge_type: 'causal', anomalous: true },
  // Proxy → outcome (Type 2 fingerprint)
  { source: 'zip_code_cluster',  target: 'denial_rate_minority', edge_type: 'causal' },
  { source: 'zip_code_cluster',  target: 'denial_rate_zip7',     edge_type: 'causal' },
];

window.CausalGraph = function CausalGraph({ nodes, edges, nodeStates, anomalyNodes, anomalyEdges, judgeView, onJudgeToggle }) {
  const { useState, useRef, useCallback, useEffect, useMemo } = React;

  // Use backend data if available, otherwise show the loan domain static graph
  const effectiveNodes = (nodes && nodes.length > 0) ? nodes : LOAN_GRAPH_NODES;
  const effectiveEdges = (nodes && nodes.length > 0) ? edges : LOAN_GRAPH_EDGES;
  const isLive = nodes && nodes.length > 0;

  const containerRef = useRef(null);
  const svgRef = useRef(null);
  const [transform, setTransform] = useState({ x: 0, y: 0, scale: 1 });
  const dragRef = useRef(null);
  const [showLegend, setShowLegend] = useState(false);
  const [simPositions, setSimPositions] = useState({});
  const animFrameRef = useRef(null);

  // Force-directed simulation (simplified physics)
  useEffect(() => {
    if (!effectiveNodes || effectiveNodes.length === 0) { setSimPositions({}); return; }
    const W = 800, H = 500;
    const cx = W / 2, cy = H / 2;

    // Initialize all nodes at center with slight random offset
    let positions = {};
    effectiveNodes.forEach((n, i) => {
      const angle = (i / effectiveNodes.length) * Math.PI * 2;
      positions[n.id] = {
        x: cx + (Math.random() - 0.5) * 30,
        y: cy + (Math.random() - 0.5) * 30,
        vx: 0, vy: 0,
        // Target positions based on layer
        layer: _getLayer(n),
      };
    });

    // Assign target X by layer
    const layerOrder = ['input', 'proxy', 'hidden', 'decision', 'policy', 'outcome'];
    const xStep = W / (layerOrder.length + 1);
    const layerCounts = {};
    effectiveNodes.forEach(n => {
      const l = _getLayer(n);
      layerCounts[l] = (layerCounts[l] || 0) + 1;
    });
    const layerIdx = {};
    effectiveNodes.forEach(n => {
      const l = _getLayer(n);
      if (!layerIdx[l]) layerIdx[l] = 0;
      const li = layerOrder.indexOf(l);
      const count = layerCounts[l] || 1;
      const yStep = H / (count + 1);
      positions[n.id].targetX = xStep * (li + 1);
      positions[n.id].targetY = yStep * (layerIdx[l] + 1);
      layerIdx[l]++;
    });

    let alpha = 1.0;
    let frame = 0;
    const maxFrames = 90; // ~1.5s at 60fps

    const tick = () => {
      alpha *= 0.96;
      frame++;
      const newPos = { ...positions };

      Object.keys(newPos).forEach(id => {
        const p = newPos[id];
        // Spring force toward target
        const dx = p.targetX - p.x;
        const dy = p.targetY - p.y;
        p.vx += dx * 0.08 * alpha;
        p.vy += dy * 0.08 * alpha;
        p.vx *= 0.6;
        p.vy *= 0.6;
        p.x += p.vx;
        p.y += p.vy;
      });

      // Repulsion between nodes
      const ids = Object.keys(newPos);
      for (let i = 0; i < ids.length; i++) {
        for (let j = i + 1; j < ids.length; j++) {
          const a = newPos[ids[i]], b = newPos[ids[j]];
          const ddx = b.x - a.x, ddy = b.y - a.y;
          const dist = Math.sqrt(ddx * ddx + ddy * ddy) || 1;
          if (dist < 80) {
            const force = (80 - dist) * 0.3 * alpha;
            const fx = (ddx / dist) * force;
            const fy = (ddy / dist) * force;
            a.x -= fx; a.y -= fy;
            b.x += fx; b.y += fy;
          }
        }
      }

      positions = newPos;
      setSimPositions({ ...newPos });

      if (frame < maxFrames && alpha > 0.01) {
        animFrameRef.current = requestAnimationFrame(tick);
      }
    };

    // Start animation
    animFrameRef.current = requestAnimationFrame(tick);
    return () => { if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current); };
  }, [effectiveNodes]);

  function _getLayer(node) {
    const t = (node.node_type || node.type || 'input').toLowerCase();
    if (t.includes('proxy')) return 'proxy';
    if (t.includes('hidden') || t.includes('latent')) return 'hidden';
    if (t.includes('decision') || t.includes('record')) return 'decision';
    if (t.includes('outcome')) return 'outcome';
    if (t.includes('policy')) return 'policy';
    return 'input';
  }

  // Pan/zoom
  const onMouseDown = useCallback((e) => {
    if (e.button !== 0) return;
    dragRef.current = { startX: e.clientX - transform.x, startY: e.clientY - transform.y };
  }, [transform]);
  const onMouseMove = useCallback((e) => {
    if (!dragRef.current) return;
    setTransform(prev => ({ ...prev, x: e.clientX - dragRef.current.startX, y: e.clientY - dragRef.current.startY }));
  }, []);
  const onMouseUp = useCallback(() => { dragRef.current = null; }, []);
  const onWheel = useCallback((e) => {
    e.preventDefault();
    const delta = e.deltaY > 0 ? 0.9 : 1.1;
    setTransform(prev => ({ ...prev, scale: Math.min(2.5, Math.max(0.3, prev.scale * delta)) }));
  }, []);

  // Node visual spec per type
  const getNodeStyle = (node) => {
    const nid = node.id || '';
    const t = (node.node_type || node.type || 'input').toLowerCase();
    const state = nodeStates[nid] || 'default';
    const isAnomaly = judgeView && anomalyNodes.includes(nid);
    let fill = '#0D1420', stroke = '#1E3A5F', strokeDash = '', size = 36, shape = 'circle';

    if (t.includes('proxy')) { fill = '#130D1F'; stroke = '#4C1D95'; strokeDash = '4,3'; }
    else if (t.includes('hidden') || t.includes('latent')) { fill = '#0F0820'; stroke = '#8B5CF6'; size = 40; }
    else if (t.includes('decision') || t.includes('record')) { fill = '#0A1420'; stroke = '#1E3A5F'; size = 28; shape = 'rect'; }
    else if (t.includes('outcome')) { fill = '#0A1A0F'; stroke = '#065F46'; size = 48; shape = 'hex'; }
    else if (t.includes('policy')) { fill = '#1A1305'; stroke = '#92400E'; size = 40; shape = 'diamond'; strokeDash = '3,2'; }

    let glow = '';
    if (isAnomaly) { stroke = '#EF4444'; fill = 'rgba(239,68,68,0.06)'; glow = `drop-shadow(0 0 8px rgba(239,68,68,0.5))`; }
    if (state === 'queried') { stroke = '#F59E0B'; glow = `drop-shadow(0 0 6px rgba(245,158,11,0.6))`; }
    if (state === 'correct') { stroke = '#10B981'; glow = `drop-shadow(0 0 8px rgba(16,185,129,0.7))`; }
    if (state === 'incorrect') { stroke = '#F97316'; }

    return { fill, stroke, strokeDash, glow, size, shape, isHidden: t.includes('hidden') || t.includes('latent') };
  };

  const getEdgeStyle = (edge) => {
    const key = `${edge.source}->${edge.target}`;
    const isAnomaly = judgeView && anomalyEdges.includes(key);
    const isPolicy = (edge.edge_type || '').toLowerCase().includes('policy');
    if (isAnomaly) return { stroke: '#EF4444', strokeWidth: 2, strokeDash: '6,4', animated: true };
    if (isPolicy) return { stroke: '#92400E', strokeWidth: 1.5, strokeDash: '6,3', animated: false };
    return { stroke: '#1E3A5F', strokeWidth: 1.5, strokeDash: '', animated: false };
  };

  const getPos = (node) => {
    const sp = simPositions[node.id];
    return sp ? { x: sp.x, y: sp.y } : { x: node.x || 100, y: node.y || 100 };
  };

  const renderNode = (node, idx) => {
    const style = getNodeStyle(node);
    const pos = getPos(node);
    const r = style.size / 2;
    const label = (node.id || '').replace(/_/g, '\u200b_');
    const labelShort = label.length > 14 ? label.substring(0, 14) + '…' : label;

    const commonProps = {
      fill: style.fill, stroke: style.stroke, strokeWidth: 1.5,
      strokeDasharray: style.strokeDash || undefined,
      className: 'node-circle',
    };

    let shape;
    if (style.shape === 'rect') {
      shape = <rect x={-r} y={-r*0.7} width={r*2} height={r*1.4} rx={6} {...commonProps} />;
    } else if (style.shape === 'hex') {
      const s = r;
      const pts = [0,-s, s*0.866,-s*0.5, s*0.866,s*0.5, 0,s, -s*0.866,s*0.5, -s*0.866,-s*0.5].join(',');
      shape = <polygon points={pts} {...commonProps} />;
    } else if (style.shape === 'diamond') {
      shape = <polygon points={`0,${-r} ${r},0 0,${r} ${-r},0`} {...commonProps} />;
    } else {
      shape = <circle r={r} {...commonProps} />;
    }

    const filterId = `glow-${idx}`;
    return (
      <g key={node.id || idx} transform={`translate(${pos.x},${pos.y})`} style={{ transition: 'transform 0.15s ease-out' }}>
        {style.glow && (
          <defs>
            <filter id={filterId} x="-60%" y="-60%" width="220%" height="220%">
              <feGaussianBlur stdDeviation="4" result="coloredBlur" />
              <feMerge><feMergeNode in="coloredBlur" /><feMergeNode in="SourceGraphic" /></feMerge>
            </filter>
          </defs>
        )}
        {/* Hidden node halo ring */}
        {style.isHidden && (
          <circle r={r} fill="none" stroke="#8B5CF6" strokeWidth={1} opacity={0.4}>
            <animate attributeName="r" values={`${r};${r+12};${r}`} dur="2s" repeatCount="indefinite" />
            <animate attributeName="opacity" values="0.5;0;0.5" dur="2s" repeatCount="indefinite" />
          </circle>
        )}
        <g filter={style.glow ? `url(#${filterId})` : undefined}>
          {shape}
        </g>
        <text y={r + 14} textAnchor="middle" style={{
          fill: 'var(--text-on-dark)', fontSize: '9px', fontFamily: 'var(--font-mono)', pointerEvents: 'none', fontWeight: 400,
        }}>{labelShort}</text>
      </g>
    );
  };

  const renderEdge = (edge, idx) => {
    const srcNode = effectiveNodes.find(n => n.id === edge.source);
    const tgtNode = effectiveNodes.find(n => n.id === edge.target);
    if (!srcNode || !tgtNode) return null;
    const style = getEdgeStyle(edge);
    const s = getPos(srcNode), t = getPos(tgtNode);
    const dx = t.x - s.x, dy = t.y - s.y;
    const len = Math.sqrt(dx*dx + dy*dy) || 1;
    const r1 = 18, r2 = 18;
    const sx = s.x + (dx/len)*r1, sy = s.y + (dy/len)*r1;
    const ex = t.x - (dx/len)*r2, ey = t.y - (dy/len)*r2;
    // Curve control point
    const mx = (sx+ex)/2 + (sy-ey)*0.1, my = (sy+ey)/2 + (ex-sx)*0.1;

    return (
      <g key={`${edge.source}-${edge.target}-${idx}`}>
        <defs>
          <marker id={`arrow-${idx}`} markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
            <path d="M0,0 L0,6 L8,3 z" fill={style.stroke} opacity={0.7} />
          </marker>
        </defs>
        <path
          d={`M${sx},${sy} Q${mx},${my} ${ex},${ey}`}
          fill="none" stroke={style.stroke} strokeWidth={style.strokeWidth}
          strokeDasharray={style.strokeDash || undefined}
          markerEnd={`url(#arrow-${idx})`} opacity={0.6}
          style={style.animated ? { animation: 'dash-travel 0.8s linear infinite' } : undefined}
        />
      </g>
    );
  };

  const hasGraph = effectiveNodes && effectiveNodes.length > 0;

  return (
    <div className="dark-panel" style={{ display: 'flex', flexDirection: 'column', height: '100%', position: 'relative' }}>
      {/* Header */}
      <div className="dark-panel-header">
        <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
          <span className="dark-panel-label">CAUSAL GRAPH</span>
          <span style={{ fontFamily: 'var(--font-mono)', fontSize: '10px', color: 'var(--text-muted-dark)' }}>
            {effectiveNodes.length} NODES · {effectiveEdges.length} EDGES
          </span>
          {!isLive && (
            <span style={{ fontFamily: 'var(--font-mono)', fontSize: '9px', color: 'var(--amber)', background: 'rgba(245,158,11,0.1)', padding: '2px 8px', borderRadius: 999, border: '1px solid rgba(245,158,11,0.3)' }}>LOAN DOMAIN · STATIC</span>
          )}
        </div>
        <button className={`judge-toggle ${judgeView ? 'on' : 'off'}`} onClick={onJudgeToggle} id="btn-judge-view">
          JUDGE VIEW: {judgeView ? 'ON' : 'OFF'}
        </button>
      </div>

      {/* Graph area */}
      <div id="graph-canvas" ref={containerRef}
        onMouseDown={onMouseDown} onMouseMove={onMouseMove}
        onMouseUp={onMouseUp} onMouseLeave={onMouseUp} onWheel={onWheel}
        style={{ flex: 1, overflow: 'hidden', position: 'relative' }}
      >
        {!hasGraph ? (
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '100%', gap: '16px' }}>
            <div style={{ width: 48, height: 48, borderRadius: '50%', border: '2px solid var(--border-dark)', borderTop: '2px solid var(--cyan)', animation: 'spin 1s linear infinite' }} />
            <span style={{ fontFamily: 'var(--font-body)', fontSize: '13px', color: 'var(--text-muted-dark)', fontWeight: 400 }}>
              Loading graph...
            </span>
          </div>
        ) : (
          <svg ref={svgRef} style={{ width: '100%', height: '100%' }} onWheel={onWheel}>
            <g transform={`translate(${transform.x},${transform.y}) scale(${transform.scale})`}>
              {effectiveEdges.map((e, i) => renderEdge(e, i))}
              {effectiveNodes.map((n, i) => renderNode(n, i))}
            </g>
          </svg>
        )}

        {/* Zoom controls */}
        <div style={{ position: 'absolute', bottom: 16, right: 16, display: 'flex', flexDirection: 'column', gap: 4 }}>
          {[
            { label: '+', fn: () => setTransform(p => ({ ...p, scale: Math.min(2.5, p.scale * 1.2) })) },
            { label: '−', fn: () => setTransform(p => ({ ...p, scale: Math.max(0.3, p.scale * 0.8) })) },
            { label: '⊙', fn: () => setTransform({ x: 0, y: 0, scale: 1 }) },
          ].map(b => (
            <button key={b.label} onClick={b.fn} style={{
              width: 32, height: 32, display: 'flex', alignItems: 'center', justifyContent: 'center',
              background: 'rgba(13,17,23,0.8)', border: '1px solid var(--border-dark)',
              borderRadius: 8, color: 'var(--text-on-dark)', fontSize: 14, cursor: 'pointer',
            }}>{b.label}</button>
          ))}
        </div>

        {/* Legend toggle */}
        <div style={{ position: 'absolute', bottom: 16, left: 16 }}>
          <button onClick={() => setShowLegend(v => !v)} style={{
            fontFamily: 'var(--font-mono)', fontSize: '10px', padding: '4px 12px',
            background: 'rgba(9,12,20,0.85)', border: '1px solid var(--border-dark)',
            borderRadius: 'var(--radius-pill)', color: 'var(--text-muted-dark)', cursor: 'pointer',
          }}>LEGEND</button>
          {showLegend && (
            <div className="graph-legend" style={{ bottom: '36px' }}>
              {[
                { color: '#F59E0B', label: 'Queried' },
                { color: '#10B981', label: 'Correct' },
                { color: '#F97316', label: 'Incorrect' },
                ...(judgeView ? [{ color: '#EF4444', label: 'Anomaly' }] : []),
              ].map(({ color, label }) => (
                <div key={label} className="legend-item">
                  <div className="legend-dot" style={{ background: color, boxShadow: `0 0 6px ${color}` }} />
                  <span>{label}</span>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
