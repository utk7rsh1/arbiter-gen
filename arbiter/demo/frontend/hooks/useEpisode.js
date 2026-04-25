// useEpisode.js — episode state machine and step logic

window.useEpisode = function useEpisode(backend) {
  const { useState, useRef, useCallback, useEffect } = React;

  const [sessionId, setSessionId] = useState(null);
  const [graphState, setGraphState] = useState({ nodes: [], edges: [] });
  const [claims, setClaims] = useState([]);
  const [reward, setReward] = useState({
    claim: 0, counterfactual: 0, tom: 0,
    chain: 0, consistency: 0, budget: 0, verdict: 0, total: 0,
  });
  const [hypotheses, setHypotheses] = useState({
    h1: { label: 'PROXY DISCRIM.', type: 'proxy_discrimination', status: 'ACTIVE' },
    h2: { label: 'ADV. INJECTION', type: 'adversarial_injection', status: 'ACTIVE' },
    h3: { label: 'MODEL DRIFT',    type: 'model_drift',          status: 'ACTIVE' },
  });
  const [step, setStep] = useState(0);
  const [maxSteps]      = useState(20);
  const [isRunning, setIsRunning] = useState(false);
  const [isDone, setIsDone]       = useState(false);
  const [nodeStates, setNodeStates] = useState({});
  const [anomalyNodes, setAnomalyNodes] = useState([]);
  const [anomalyEdges, setAnomalyEdges] = useState([]);
  const [overseers, setOverseers]       = useState([]);
  const [rewardDeltas, setRewardDeltas] = useState([]);
  const [episodeReward, setEpisodeReward] = useState(null);
  const [serverStatus, setServerStatus]   = useState('unknown');
  const [budget, setBudget]               = useState(20);
  const [episodeHistory, setEpisodeHistory] = useState([]);

  const runIntervalRef = useRef(null);
  const deltaIdRef     = useRef(0);
  const stepRef        = useRef(0); // always-current ref for use inside intervals

  // Keep stepRef in sync
  useEffect(() => { stepRef.current = step; }, [step]);

  // ── Health check ──────────────────────────────────────────────────────────────
  const checkHealth = useCallback(async () => {
    try {
      const h = await backend.getHealth();
      setServerStatus(h.status === 'ok' ? 'ok' : 'error');
    } catch { setServerStatus('error'); }
  }, [backend]);

  useEffect(() => { checkHealth(); }, [checkHealth]);

  // ── New session ───────────────────────────────────────────────────────────────
  const newSession = useCallback(async (level, seed, checkpoint, domainJson) => {
    if (runIntervalRef.current) clearInterval(runIntervalRef.current);
    setIsRunning(false);
    setIsDone(false);
    setClaims([]);
    setStep(0);
    stepRef.current = 0;
    setNodeStates({});
    setOverseers([]);
    setRewardDeltas([]);
    setEpisodeReward(null);
    setBudget(20);
    setReward({ claim: 0, counterfactual: 0, tom: 0, chain: 0, consistency: 0, budget: 0, verdict: 0, total: 0 });
    setHypotheses({
      h1: { label: 'PROXY DISCRIM.', type: 'proxy_discrimination', status: 'ACTIVE' },
      h2: { label: 'ADV. INJECTION', type: 'adversarial_injection', status: 'ACTIVE' },
      h3: { label: 'MODEL DRIFT',    type: 'model_drift',          status: 'ACTIVE' },
    });
    try {
      const sess = await backend.createSession(level, seed, checkpoint, domainJson || null);
      const sid = sess.session_id;
      setSessionId(sid);
      const obs = await backend.resetSession(sid, seed);
      const render = await backend.renderSession(sid);
      _applyRender(render);
    } catch (e) {
      console.error('newSession error:', e);
    }
  }, [backend]);

  // ── Render → graph state ──────────────────────────────────────────────────────
  const _applyRender = useCallback((render) => {
    if (!render) return;
    const rawNodes = render.graph_nodes || [];
    const rawEdges = render.graph_edges || [];
    const positioned = _autoLayout(rawNodes);
    setGraphState({ nodes: positioned, edges: rawEdges });

    // Mark queried nodes yellow
    const qs = {};
    (render.queried_nodes || []).forEach(n => { qs[n] = 'queried'; });
    if (Object.keys(qs).length > 0) {
      setNodeStates(prev => ({ ...prev, ...qs }));
    }

    if (render.claims && render.claims.length > 0) setClaims(render.claims);
    if (typeof render.step === 'number')  setStep(render.step);
    if (typeof render.running_reward === 'number') {
      setReward(prev => ({ ...prev, total: render.running_reward }));
    }
  }, []);

  // ── Auto-layout ───────────────────────────────────────────────────────────────
  const _autoLayout = (rawNodes) => {
    const W = 860, H = 560;
    const layers = { input: [], proxy: [], hidden: [], decision: [], policy: [], outcome: [] };

    rawNodes.forEach(n => {
      const t = (n.node_type || n.type || 'input').toLowerCase();
      if (t.includes('proxy'))                       layers.proxy.push(n);
      else if (t.includes('decision') || t.includes('record')) layers.decision.push(n);
      else if (t.includes('outcome'))                layers.outcome.push(n);
      else if (t.includes('policy'))                 layers.policy.push(n);
      else if (t.includes('hidden') || t.includes('latent')) layers.hidden.push(n);
      else                                           layers.input.push(n);
    });

    const layerOrder = ['input', 'proxy', 'hidden', 'decision', 'policy', 'outcome'];
    const xStep = W / (layerOrder.length + 1);
    const positioned = [];

    layerOrder.forEach((layer, li) => {
      const nodes = layers[layer];
      if (!nodes.length) return;
      const yStep = H / (nodes.length + 1);
      nodes.forEach((n, ni) => {
        positioned.push({ ...n, x: xStep * (li + 1), y: yStep * (ni + 1) });
      });
    });

    const placedIds = new Set(positioned.map(n => n.id));
    rawNodes.filter(n => !placedIds.has(n.id)).forEach((n, i) => {
      positioned.push({ ...n, x: 50 + (i % 8) * 110, y: 30 + Math.floor(i / 8) * 100 });
    });
    return positioned;
  };

  // ── Scripted demo action sequence ─────────────────────────────────────────────
  // Each step issues a realistic agent action that produces visible UI output.
  //
  // HOW IT WILL CHANGE WITH REAL CHECKPOINTS:
  // When lora_sft / lora_grpo checkpoints are available, this entire scripted
  // sequence gets replaced with a single call:
  //   const action = await backend.agentStep(sessionId)
  // The server will run the LLM's forward pass and return whatever action it chose.
  // The frontend code below (node coloring, claim card rendering, reward deltas)
  // stays exactly the same — it just receives real LLM actions instead of scripted ones.
  //
  // Current state: scripted sequence mirrors what a well-trained ARBITER agent does.
  const _getScriptedAction = (stepNum) => {
    // STRUCTURE: each action has:
    //   claim: {}   → sent verbatim to backend (**claim in Python) — MUST match dataclass exactly
    //   _ui:  {}    → frontend-only display labels, stripped before POST
    const ACTIONS = [
      // Step 0: Query all denied records → all feature nodes turn yellow
      { type: 'QUERY_RECORDS', feature_filter: {}, outcome_filter: 'denied' },

      // Step 1: Distribution query on zip_code_cluster → the proxy variable
      { type: 'QUERY_FEATURE_DISTRIBUTION', feature_id: 'zip_code_cluster', group_by: 'loan_outcome' },

      // Step 2: CLAIM_CAUSAL — CausalLinkClaim fields: cause_feature, effect_outcome, mechanism, direction, confidence, basis_records, anomaly_type
      { type: 'CLAIM_CAUSAL',
        claim: {
          cause_feature:  'zip_code_cluster',
          effect_outcome: 'denial_rate_overall',
          mechanism:      'internal_risk_score',
          direction:      'positive',
          confidence:     'HIGH',
          basis_records:  ['rec_001', 'rec_007', 'rec_012'],
          anomaly_type:   'proxy_discrimination',
        },
        _ui: { cause_feature: 'zip_code_cluster', effect_feature: 'loan_denied' },
      },

      // Step 3: Counterfactual query
      { type: 'QUERY_COUNTERFACTUAL', record_id: 'rec_001', feature_id: 'zip_code_cluster', counterfactual_value: 3 },

      // Step 4: CLAIM_COUNTERFACTUAL — CounterfactualClaim fields: subject_record, counterfactual_feature, predicted_outcome_change, confidence, basis
      { type: 'CLAIM_COUNTERFACTUAL',
        claim: {
          subject_record:           'rec_001',
          counterfactual_feature:   'zip_code_cluster',
          predicted_outcome_change: 'approved',
          confidence:               'HIGH',
          basis:                    'cf_query_step_3',
        },
        _ui: { record_id: 'rec_001', feature_id: 'zip_code_cluster', actual_outcome: 'denied', predicted_outcome: 'approved' },
      },

      // Step 5: Distribution query on credit_score (confirm it's not the driver)
      { type: 'QUERY_FEATURE_DISTRIBUTION', feature_id: 'credit_score', group_by: 'zip_code_cluster' },

      // Step 6: CLAIM_CAUSAL — mechanism link: zip_code → internal_risk_score
      { type: 'CLAIM_CAUSAL',
        claim: {
          cause_feature:  'zip_code_cluster',
          effect_outcome: 'internal_risk_score',
          mechanism:      'proxy_laundering',
          direction:      'positive',
          confidence:     'HIGH',
          basis_records:  ['rec_001', 'rec_007', 'rec_012', 'rec_019'],
          anomaly_type:   'proxy_discrimination',
        },
        _ui: { cause_feature: 'zip_code_cluster', effect_feature: 'internal_risk_score' },
      },

      // Step 7: Flag H1 (proxy_discrimination) as CONFIRMED → card turns green
      { type: 'FLAG_HYPOTHESIS', hypothesis_type: 'proxy_discrimination', status: 'CONFIRMED' },

      // Step 8: Flag H3 (model_drift) as ELIMINATED → card fades out
      { type: 'FLAG_HYPOTHESIS', hypothesis_type: 'model_drift', status: 'ELIMINATED' },

      // Step 9: CLAIM_THEORY_OF_MIND — TheoryOfMindClaim fields: defender_action, target_link, obfuscation_method, confidence, basis
      { type: 'CLAIM_THEORY_OF_MIND',
        claim: {
          defender_action:    'obfuscating',
          target_link:        'zip_code_cluster->internal_risk_score',
          obfuscation_method: 'timestamp_manipulation',
          confidence:         'HIGH',
          basis:              'timestamp_clustering_around_denial_events',
        },
        _ui: { defender_action: 'obfuscating', obfuscation_method: 'timestamp_manipulation' },
      },

      // Step 10: Second counterfactual query on a different record
      { type: 'QUERY_COUNTERFACTUAL', record_id: 'rec_007', feature_id: 'zip_code_cluster', counterfactual_value: 1 },

      // Step 11: Submit final report → episode ends, total reward shown
      { type: 'SUBMIT_REPORT',
        anomaly_type: 'proxy_discrimination',
        primary_evidence_chain: ['zip_code_cluster', 'internal_risk_score', 'denial_rate_overall'],
        affected_demographic: 'zip_code_cluster_7',
        recommended_action: 'audit_risk_score_model',
      },
    ];
    return ACTIONS[Math.min(stepNum, ACTIONS.length - 1)];
  };

  // ── Single step ───────────────────────────────────────────────────────────────
  const doStep = useCallback(async (overrideSessionId, overrideDone) => {
    const sid     = overrideSessionId !== undefined ? overrideSessionId : sessionId;
    const epDone  = overrideDone      !== undefined ? overrideDone      : isDone;
    if (!sid || epDone) return;

    const currentStep = stepRef.current;
    const action = _getScriptedAction(currentStep);

    // Strip _ui before sending to backend — Python dataclass rejects unknown kwargs
    const { _ui: ui = {}, ...actionForBackend } = action;

    try {
      const result = await backend.stepSession(sid, actionForBackend);
      const { observation, reward: stepReward, done, info } = result;

      // ── Observation updates ─────────────────────────────────────────────────
      if (observation) {
        const newStep = observation.step || 0;
        setStep(newStep);
        stepRef.current = newStep;

        // Forward-compatible: update budget from backend when available
        if (typeof observation.remaining_budget === 'number') {
          setBudget(observation.remaining_budget);
        } else {
          // Estimate budget from step count
          const cost = (action.type === 'QUERY_COUNTERFACTUAL') ? 2 : (action.type.startsWith('QUERY_') ? 1 : 0);
          setBudget(prev => Math.max(0, prev - cost));
        }

        if (observation.hypothesis_flags) {
          setHypotheses(prev => {
            const next = { ...prev };
            Object.entries(observation.hypothesis_flags).forEach(([k, v]) => {
              const key = k === 'proxy_discrimination' ? 'h1'
                : k === 'adversarial_injection' ? 'h2' : 'h3';
              if (next[key]) next[key] = { ...next[key], status: v };
            });
            return next;
          });
        }

        // Mark queried nodes yellow immediately (don't wait for render refresh)
        if (observation.queried_nodes && observation.queried_nodes.length > 0) {
          setNodeStates(prev => {
            const next = { ...prev };
            observation.queried_nodes.forEach(nid => {
              if (!next[nid] || next[nid] === 'default') next[nid] = 'queried';
            });
            return next;
          });
        }

        // Also: if this was a QUERY action, explicitly mark the queried feature yellow
        if (action.type === 'QUERY_FEATURE_DISTRIBUTION' && action.feature_id) {
          setNodeStates(prev => ({ ...prev, [action.feature_id]: 'queried' }));
        }
        if (action.type === 'QUERY_COUNTERFACTUAL' && action.feature_id) {
          setNodeStates(prev => ({ ...prev, [action.feature_id]: 'queried' }));
        }
      }

      // ── Claim verification → add card + color cause/effect nodes ────────────
      if (info && info.verification) {
        const v = info.verification;
        const atype = action.type;
        const claimType = atype === 'CLAIM_CAUSAL' ? 'causal'
          : atype === 'CLAIM_COUNTERFACTUAL' ? 'counterfactual'
          : atype === 'CLAIM_THEORY_OF_MIND' ? 'theory_of_mind' : 'causal';

        const trueCount  = Object.values(v).filter(x => x === true).length;
        const falseCount = Object.values(v).filter(x => x === false).length;
        const isCorrect  = trueCount >= falseCount;
        // claimData = backend fields (clean); ui = display overrides
        const claimData  = actionForBackend.claim || {};

        const newClaim = {
          claim_type:     claimType,
          step:           currentStep,
          confidence:     claimData.confidence || 'HIGH',
          verification:   v,
          correct:        isCorrect,
          reward_delta:   stepReward,
          // Causal display (ui overrides, fall back to backend field names)
          cause_feature:  ui.cause_feature  || claimData.cause_feature,
          effect_feature: ui.effect_feature || claimData.effect_outcome,
          mechanism:      claimData.mechanism,
          basis_records:  claimData.basis_records || [],
          // Counterfactual display
          record_id:        ui.record_id        || claimData.subject_record,
          feature_id:       ui.feature_id       || claimData.counterfactual_feature,
          predicted_outcome: ui.predicted_outcome || claimData.predicted_outcome_change,
          actual_outcome:   ui.actual_outcome,
          // ToM display
          defender_action:    ui.defender_action    || claimData.defender_action,
          obfuscation_method: ui.obfuscation_method || claimData.obfuscation_method,
          target_link:        claimData.target_link,
        };
        setClaims(prev => [...prev, newClaim]);

        // Color cause/effect nodes green (correct) or orange (incorrect)
        const nodesToColor = [
          claimData.cause_feature,
          claimData.effect_outcome,
          ui.effect_feature,
        ].filter(Boolean);
        if (nodesToColor.length > 0) {
          setNodeStates(prev => {
            const next = { ...prev };
            nodesToColor.forEach(nid => { next[nid] = isCorrect ? 'correct' : 'incorrect'; });
            return next;
          });
        }
      }

      // ── FLAG_HYPOTHESIS → update card state immediately ─────────────────────
      if (action.type === 'FLAG_HYPOTHESIS') {
        setHypotheses(prev => {
          const next = { ...prev };
          Object.entries(next).forEach(([k, h]) => {
            if (h.type === action.hypothesis_type) next[k] = { ...h, status: action.status };
          });
          return next;
        });
        // No claim card for FLAG, but we just keep going
      }

      // ── Reward delta popup ──────────────────────────────────────────────────
      if (typeof stepReward === 'number' && stepReward !== 0) {
        const id   = deltaIdRef.current++;
        const comp = action.type.includes('COUNTERFACTUAL') ? 'counterfactual'
          : action.type.includes('THEORY') ? 'tom'
          : stepReward > 0 ? 'claim' : 'consistency';
        setRewardDeltas(prev => [...prev, { id, component: comp, value: stepReward }]);
        setTimeout(() => setRewardDeltas(prev => prev.filter(d => d.id !== id)), 900);
        setReward(prev => ({
          ...prev,
          [comp]: prev[comp] + stepReward,
          total: prev.total + stepReward,
        }));
      }

      // ── Overseer contradiction ──────────────────────────────────────────────
      if (info && info.consistency && info.consistency.num_violations > 0) {
        (info.consistency.violations || []).forEach(vi => {
          setOverseers(prev => [...prev, {
            id: Date.now(),
            message: vi.description || 'Contradiction detected between claims',
            penalty: vi.penalty || -1.0,
            step: currentStep,
          }]);
        });
      }

      // ── Episode done ────────────────────────────────────────────────────────
      if (done) {
        setIsDone(true);
        if (runIntervalRef.current) clearInterval(runIntervalRef.current);
        setIsRunning(false);
        // Add to episode history for sparkline
        const finalTotal = reward?.total ?? total ?? 0;
        setEpisodeHistory(prev => [...prev.slice(-19), { episode: prev.length, reward: typeof stepReward === 'number' ? (reward?.total ?? 0) + stepReward : 0 }]);
        if (info && info.episode_reward) {
          const er = info.episode_reward;
          setEpisodeReward(er);
          setReward({
            claim:          er.intermediate?.claim_reward    || 0,
            counterfactual: er.intermediate?.counterfactual_reward || 0,
            tom:            er.intermediate?.tom_reward       || 0,
            chain:          er.terminal?.chain_score          || 0,
            consistency:    er.terminal?.consistency_penalty  || 0,
            budget:         er.terminal?.budget_efficiency    || 0,
            verdict:        er.terminal?.verdict_correct ? 5 : 0,
            total:          er.total || 0,
          });
        }
        const render = await backend.renderSession(sid);
        _applyRender(render);
        return;
      }

      // ── Refresh graph to sync server-side queried_nodes ────────────────────
      const render = await backend.renderSession(sid);
      _applyRender(render);

    } catch (e) {
      console.error('step error:', e);
    }
  }, [sessionId, isDone, backend, _applyRender]);

  // ── Auto-run ──────────────────────────────────────────────────────────────────
  const stopAuto = useCallback(() => {
    if (runIntervalRef.current) {
      clearInterval(runIntervalRef.current);
      runIntervalRef.current = null;
    }
    setIsRunning(false);
  }, []);

  const startAuto = useCallback((speed) => {
    stopAuto();
    if (!sessionId) return;
    const ms = speed === '5x' ? 500 : speed === '2x' ? 1000 : 1800;
    setIsRunning(true);
    // Capture current sid/done in closure to avoid stale refs
    const sid = sessionId;
    runIntervalRef.current = setInterval(async () => {
      if (stepRef.current >= 20) {
        stopAuto();
        return;
      }
      await doStep(sid, false);
    }, ms);
  }, [sessionId, doStep, stopAuto]);

  useEffect(() => () => { if (runIntervalRef.current) clearInterval(runIntervalRef.current); }, []);

  // ── Anomaly detection from graph metadata ─────────────────────────────────────
  useEffect(() => {
    const anodes = graphState.nodes.filter(n => n.is_anomalous || n.anomalous).map(n => n.id);
    const aedges = graphState.edges
      .filter(e => e.is_anomalous || e.anomalous)
      .map(e => `${e.source}->${e.target}`);
    setAnomalyNodes(anodes);
    setAnomalyEdges(aedges);
  }, [graphState]);

  return {
    sessionId, graphState, claims, reward, hypotheses,
    step, maxSteps, isRunning, isDone, nodeStates,
    anomalyNodes, anomalyEdges, overseers, rewardDeltas, episodeReward,
    serverStatus, budget, episodeHistory,
    newSession, doStep, startAuto, stopAuto, checkHealth,
  };
};
