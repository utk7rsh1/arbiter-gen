// App.jsx — Root component: BioSync-inspired master grid layout with tab routing

(function() {
  const { useState, useCallback, useEffect, useRef } = React;

  function App() {
    // ── Global UI state ─────────────────────────────────────────────
    const [activeTab, setActiveTab] = useState('LIVE');
    const [level,     setLevel]     = useState('L1');
    const [modelMode, setModelMode] = useState('UNTRAINED');
    const [seed,      setSeed]      = useState(42);
    const [speed,     setSpeed]     = useState('MANUAL');
    const [judgeView, setJudgeView] = useState(true);

    // ── Backend + Episode hooks ─────────────────────────────────────
    const backend = window.useBackend();
    const episode = window.useEpisode(backend);

    // ── Checkpoint map ──────────────────────────────────────────────
    const checkpointFor = (model) => {
      if (model === 'SFT ONLY')      return 'lora_sft';
      if (model === 'FULL ARBITER')  return 'lora_grpo';
      return 'base';
    };

    // ── Actions ─────────────────────────────────────────────────────
    const handleReset = useCallback(() => {
      const levelNum = parseInt(level.replace('L','')) || 1;
      episode.newSession(levelNum, seed, checkpointFor(modelMode));
    }, [level, seed, modelMode, episode]);

    const handleStep = useCallback(() => {
      if (speed === 'MANUAL') {
        episode.doStep();
      } else {
        const s = speed.replace('×','x');
        if (episode.isRunning) {
          episode.stopAuto();
        } else {
          episode.startAuto(s);
        }
      }
    }, [speed, episode]);

    const handlePause = useCallback(() => {
      episode.stopAuto();
    }, [episode]);

    // ── Speed changes restart auto-run ──────────────────────────────
    useEffect(() => {
      if (episode.isRunning && speed !== 'MANUAL') {
        const s = speed.replace('×', 'x');
        episode.startAuto(s);
      }
    }, [speed]);

    // ── Keyboard shortcuts ──────────────────────────────────────────
    useEffect(() => {
      const handler = (e) => {
        if (e.target.tagName === 'INPUT') return;
        if (e.key === '1') setActiveTab('LIVE');
        if (e.key === '2') setActiveTab('ARMS_RACE');
        if (e.key === '3') setActiveTab('COMPARISON');
        if (e.key === '4') setActiveTab('STATS');
        if (e.key === 'j') setJudgeView(v => !v);
        if (e.key === ' ') { e.preventDefault(); handleStep(); }
        if (e.key === 'r') handleReset();
      };
      window.addEventListener('keydown', handler);
      return () => window.removeEventListener('keydown', handler);
    }, [handleStep, handleReset]);

    // ── Tab content transition ──────────────────────────────────────
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

    // ── Layout ──────────────────────────────────────────────────────
    return (
      <div id="app-root" style={{ height: '100vh', display: 'flex', flexDirection: 'column', overflow: 'hidden', background: 'var(--bg-shell)' }}>

        {/* Topbar + Controls Strip */}
        <window.EpisodeControls
          level={level}            onLevelChange={setLevel}
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
        />

        {/* Main content area */}
        <div style={{ flex: 1, overflow: 'hidden', padding: '16px 24px', display: 'flex', flexDirection: 'column', gap: '16px' }}>
          <div style={{ ...contentStyle, flex: 1, minHeight: 0, display: 'flex', flexDirection: 'column', gap: '16px' }}>

            {/* ── LIVE DEMO TAB ───────────────────────────────────────── */}
            {activeTab === 'LIVE' && (
              <>
                {/* Main content: 58% graph / 42% right column */}
                <div style={{ flex: 1, display: 'flex', gap: '16px', minHeight: 0 }}>
                  {/* Left: Causal Graph dark panel (58%) */}
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

                  {/* Right column (42%): Claim Chain + Reward + Hypothesis */}
                  <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: '16px', minWidth: 0 }}>
                    {/* Claim chain (55%) */}
                    <div style={{ flex: '0 0 55%', minHeight: 0 }}>
                      <window.ClaimChain
                        claims={episode.claims}
                        overseers={episode.overseers}
                      />
                    </div>
                    {/* Reward panel (25%) */}
                    <div style={{ flex: '0 0 25%', minHeight: 0 }}>
                      <window.RewardPanel
                        reward={episode.reward}
                        rewardDeltas={episode.rewardDeltas}
                        episodeReward={episode.episodeReward}
                      />
                    </div>
                    {/* Hypothesis tracker (20%) */}
                    <div style={{ flex: 1, minHeight: 0 }}>
                      <window.HypothesisTracker hypotheses={episode.hypotheses} />
                    </div>
                  </div>
                </div>

                {/* Bottom metric strip */}
                <div style={{ flexShrink: 0 }}>
                  <window.BottomMetrics
                    step={episode.step}
                    maxSteps={episode.maxSteps}
                    claims={episode.claims}
                    reward={episode.reward}
                    budget={episode.budget}
                    maxBudget={20}
                    level={level}
                    episodeHistory={episode.episodeHistory}
                  />
                </div>
              </>
            )}

            {/* ── ARMS RACE TAB ───────────────────────────────────────── */}
            {activeTab === 'ARMS_RACE' && (
              <div style={{ flex: 1, minHeight: 0 }}>
                <window.ArmsRaceChart />
              </div>
            )}

            {/* ── COMPARISON TAB ──────────────────────────────────────── */}
            {activeTab === 'COMPARISON' && (
              <div style={{ flex: 1, minHeight: 0 }}>
                <window.ContrastPanel />
              </div>
            )}

            {/* ── STATS TAB ──────────────────────────────────────────── */}
            {activeTab === 'STATS' && (
              <div style={{ flex: 1, minHeight: 0, overflowY: 'auto' }}>
                <window.StatsPage metricsData={null} />
              </div>
            )}
          </div>
        </div>
      </div>
    );
  }

  // Mount
  const root = ReactDOM.createRoot(document.getElementById('root'));
  root.render(React.createElement(App));
})();
