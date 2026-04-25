// App.jsx — Root: Landing → (Loan or DomainBuilder) → Episode
(function () {
  const { useState, useCallback, useEffect, useRef } = React;

  function App() {
    // ── Screen routing: 'landing' | 'domain_builder' | 'episode' ──────────
    const [screen, setScreen]       = useState('landing');
    const [domainMode, setDomainMode] = useState('loan');   // 'loan' | 'custom'
    const [customDomain, setCustomDomain] = useState(null); // DomainConfig JSON

    // ── Episode UI state ──────────────────────────────────────────────────
    const [activeTab, setActiveTab] = useState('LIVE');
    const [level,     setLevel]     = useState(1);
    const [modelMode, setModelMode] = useState('UNTRAINED');
    const [seed,      setSeed]      = useState(42);
    const [speed,     setSpeed]     = useState('MANUAL');
    const [judgeView, setJudgeView] = useState(true);
    const [domainLabel, setDomainLabel] = useState('Loan Approval');

    const backend = window.useBackend();
    const episode = window.useEpisode(backend);

    const checkpointFor = (model) => {
      if (model === 'SFT ONLY')     return 'lora_sft';
      if (model === 'FULL ARBITER') return 'lora_grpo';
      return 'base';
    };

    // ── Landing: Loan selected ─────────────────────────────────────────────
    const handleSelectLoan = () => {
      setDomainMode('loan');
      setCustomDomain(null);
      setLevel(1);
      setScreen('episode');
    };

    // ── Landing: Custom domain selected ────────────────────────────────────
    const handleSelectCustom = () => {
      setDomainMode('custom');
      setScreen('domain_builder');
    };

    // ── DomainBuilder: user finished configuring, launch episode ──────────
    const handleDomainLaunch = useCallback(({ domainJson, level: lvl, seed: s }) => {
      setCustomDomain(domainJson);
      setLevel(lvl);
      setSeed(s);
      setDomainMode('custom');
      setDomainLabel(domainJson?.domain_name || 'Custom Domain');
      setScreen('episode');
    }, []);

    // ── Episode: reset/start session ──────────────────────────────────────
    const handleReset = useCallback(() => {
      const levelNum = typeof level === 'number' ? level : (parseInt((level + '').replace('L', '')) || 1);
      const domainJson = domainMode === 'custom' ? customDomain : null;
      episode.newSession(levelNum, seed, checkpointFor(modelMode), domainJson);
    }, [level, seed, modelMode, episode, domainMode, customDomain]);

    // Auto-start session when screen becomes 'episode' (only for Live tab)
    useEffect(() => {
      if (screen === 'episode') {
        setTimeout(handleReset, 200);
        const lbl = domainMode === 'custom' && customDomain
          ? (customDomain.domain_name || 'Custom Domain')
          : 'Loan Approval';
        setDomainLabel(lbl);
      }
    }, [screen]);

    const handleStep = useCallback(() => {
      if (speed === 'MANUAL') {
        episode.doStep();
      } else {
        const s = speed.replace('×', 'x');
        if (episode.isRunning) episode.stopAuto();
        else episode.startAuto(s);
      }
    }, [speed, episode]);

    const handlePause = useCallback(() => episode.stopAuto(), [episode]);

    useEffect(() => {
      if (episode.isRunning && speed !== 'MANUAL') {
        episode.startAuto(speed.replace('×', 'x'));
      }
    }, [speed]);

    // Keyboard shortcuts (only in episode screen, not on training tab)
    useEffect(() => {
      if (screen !== 'episode') return;
      const handler = (e) => {
        if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
        if (e.key === '1') setActiveTab('LIVE');
        if (e.key === '2') setActiveTab('ARMS_RACE');
        if (e.key === '3') setActiveTab('COMPARISON');
        if (e.key === '4') setActiveTab('STATS');
        if (e.key === '5') setActiveTab('TRAINING');
        if (e.key === 'j') setJudgeView(v => !v);
        if (e.key === ' ' && activeTab !== 'TRAINING') { e.preventDefault(); handleStep(); }
        if (e.key === 'r' && activeTab !== 'TRAINING') handleReset();
        if (e.key === 'Escape') setScreen('landing');
      };
      window.addEventListener('keydown', handler);
      return () => window.removeEventListener('keydown', handler);
    }, [screen, activeTab, handleStep, handleReset]);

    // Tab content fade
    const [tabVisible, setTabVisible] = useState(true);
    const prevTab = useRef(activeTab);
    useEffect(() => {
      if (prevTab.current !== activeTab) {
        setTabVisible(false);
        const t = setTimeout(() => { setTabVisible(true); prevTab.current = activeTab; }, 80);
        return () => clearTimeout(t);
      }
    }, [activeTab]);

    const contentStyle = {
      opacity: tabVisible ? 1 : 0,
      transform: tabVisible ? 'translateY(0)' : 'translateY(8px)',
      transition: 'opacity 0.2s ease-out, transform 0.2s ease-out',
    };

    // ── LANDING ────────────────────────────────────────────────────────────
    if (screen === 'landing') {
      return (
        <window.LandingScreen
          onSelectLoan={handleSelectLoan}
          onSelectCustom={handleSelectCustom}
        />
      );
    }

    // ── DOMAIN BUILDER ─────────────────────────────────────────────────────
    if (screen === 'domain_builder') {
      return (
        <window.DomainBuilder
          onLaunch={handleDomainLaunch}
          onBack={() => setScreen('landing')}
        />
      );
    }

    // ── EPISODE (LIVE / ARMS RACE / COMPARISON / STATS) ────────────────────
    return (
      <div id="app-root" style={{ height: '100vh', display: 'flex', flexDirection: 'column', overflow: 'hidden', background: 'var(--bg-shell)' }}>

        {/* Topbar + Controls Strip — now with domain label + back button */}
        <window.EpisodeControls
          level={'L' + level}      onLevelChange={l => setLevel(parseInt(l.replace('L','')) || 1)}
          modelMode={modelMode}    onModelChange={setModelMode}
          seed={seed}              onSeedChange={setSeed}
          speed={speed}            onSpeedChange={setSpeed}
          step={episode.step}      maxSteps={episode.maxSteps}
          isRunning={episode.isRunning}
          isDone={episode.isDone}
          onStep={handleStep}
          onPause={handlePause}
          onReset={handleReset}
          activeTab={activeTab}    onTabChange={setActiveTab}
          serverStatus={episode.serverStatus}
          domainLabel={domainLabel}
          onBackToLanding={() => setScreen('landing')}
          onChangeDomain={domainMode === 'custom'
            ? () => setScreen('domain_builder')
            : handleSelectCustom}
        />

        {/* Main content */}
        <div style={{ flex: 1, overflow: 'hidden', padding: '16px 24px', display: 'flex', flexDirection: 'column', gap: '16px' }}>
          <div style={{ ...contentStyle, flex: 1, minHeight: 0, display: 'flex', flexDirection: 'column', gap: '16px' }}>

            {/* LIVE */}
            {activeTab === 'LIVE' && (
              <>
                <div style={{ flex: 1, display: 'flex', gap: '16px', minHeight: 0 }}>
                  <div style={{ flex: '0 0 58%', display: 'flex', flexDirection: 'column', gap: '16px', minWidth: 0 }}>
                    <div style={{ flex: 1, minHeight: 0 }}>
                      <window.CausalGraph
                        nodes={episode.graphState.nodes}
                        edges={episode.graphState.edges}
                        nodeStates={episode.nodeStates}
                        anomalyNodes={episode.anomalyNodes}
                        anomalyEdges={episode.anomalyEdges}
                        judgeView={judgeView}
                        onJudgeToggle={() => setJudgeView(v => !v)}
                      />
                    </div>
                  </div>
                  <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: '16px', minWidth: 0 }}>
                    <div style={{ flex: '0 0 55%', minHeight: 0 }}>
                      <window.ClaimChain claims={episode.claims} overseers={episode.overseers} />
                    </div>
                    <div style={{ flex: '0 0 25%', minHeight: 0 }}>
                      <window.RewardPanel
                        reward={episode.reward}
                        rewardDeltas={episode.rewardDeltas}
                        episodeReward={episode.episodeReward}
                      />
                    </div>
                    <div style={{ flex: 1, minHeight: 0 }}>
                      <window.HypothesisTracker hypotheses={episode.hypotheses} />
                    </div>
                  </div>
                </div>
                <div style={{ flexShrink: 0 }}>
                  <window.BottomMetrics
                    step={episode.step}
                    maxSteps={episode.maxSteps}
                    claims={episode.claims}
                    reward={episode.reward}
                    budget={episode.budget}
                    maxBudget={20}
                    level={'L' + level}
                    episodeHistory={episode.episodeHistory}
                  />
                </div>
              </>
            )}

            {activeTab === 'ARMS_RACE' && (
              <div style={{ flex: 1, minHeight: 0 }}>
                <window.ArmsRaceChart />
              </div>
            )}

            {activeTab === 'COMPARISON' && (
              <div style={{ flex: 1, minHeight: 0 }}>
                <window.ContrastPanel />
              </div>
            )}

            {activeTab === 'STATS' && (
              <div style={{ flex: 1, minHeight: 0, overflowY: 'auto' }}>
                <window.StatsPage metricsData={null} />
              </div>
            )}

            {activeTab === 'TRAINING' && (
              <div style={{ flex: 1, minHeight: 0 }}>
                <window.TrainingMonitor />
              </div>
            )}
          </div>
        </div>
      </div>
    );
  }

  const root = ReactDOM.createRoot(document.getElementById('root'));
  root.render(React.createElement(App));
})();
