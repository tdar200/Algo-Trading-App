import React, { useState, useCallback, useEffect, useRef } from 'react';
import { io } from 'socket.io-client';
import Box from '@mui/material/Box';
import Paper from '@mui/material/Paper';
import Alert from '@mui/material/Alert';
import OptimizationConfigPanel from './OptimizationConfigPanel';
import OptimizationProgress from './OptimizationProgress';
import OptimizationResults from './OptimizationResults';
import AdvancedStatsModal from './AdvancedStatsModal';

const API_URL = process.env.REACT_APP_API_URL || "http://127.0.0.1:5000";

function OptimizationPage({ theme }) {
  const isDark = theme === 'dark';
  const socketRef = useRef(null);
  const optimizationIdRef = useRef(null);

  // Configuration state
  const [config, setConfig] = useState({
    symbol: 'AAPL',
    startDate: new Date(Date.now() - 365 * 24 * 60 * 60 * 1000).toISOString().split('T')[0],
    endDate: new Date().toISOString().split('T')[0],
    initialCapital: 100000,
    commission: 0.1,
    cpuCores: 4,
    useHybrid: false,
    screeningFilters: {
      min_sharpe: 0.5,
      max_drawdown: 30,
      min_win_rate: 40,
      min_trades: 5
    }
  });

  // Parameter ranges for optimization
  const [paramRanges, setParamRanges] = useState({
    firstRetracement: { start: 3, end: 10, step: 1, enabled: true },
    secondRetracement: { start: 3, end: 10, step: 1, enabled: true },
    touchCount: { start: 1, end: 3, step: 1, enabled: false },
    breakoutBuffer: { start: 0, end: 10, step: 2, enabled: false },
    takeProfit: { start: 5, end: 25, step: 5, enabled: true },
    stopLoss: { start: 3, end: 10, step: 1, enabled: true },
  });

  // Optimization state
  const [status, setStatus] = useState('idle'); // idle, running, completed, error
  const [isHybridRunning, setIsHybridRunning] = useState(false);
  const [progress, setProgress] = useState({
    current: 0,
    total: 0,
    percent: 0,
    currentParams: null,
    bestSoFar: null,
  });
  const [hybridProgress, setHybridProgress] = useState({
    phase: 'screening',
    screening: { current: 0, total: 0, percent: 0, candidates_found: 0 },
    validation: { current: 0, total: 0, percent: 0 }
  });
  const [results, setResults] = useState([]);
  const [error, setError] = useState(null);
  const [sortConfig, setSortConfig] = useState({ key: 'net_profit', direction: 'desc' });
  const [selectedResult, setSelectedResult] = useState(null);
  const [modalOpen, setModalOpen] = useState(false);

  // Calculate total combinations
  const calculateCombinations = useCallback(() => {
    let total = 1;
    Object.entries(paramRanges).forEach(([key, range]) => {
      if (range.enabled) {
        const count = Math.floor((range.end - range.start) / range.step) + 1;
        total *= Math.max(1, count);
      }
    });
    return total;
  }, [paramRanges]);

  const totalCombinations = calculateCombinations();

  // Initialize WebSocket connection
  useEffect(() => {
    socketRef.current = io(API_URL, {
      transports: ['websocket', 'polling'],
    });

    socketRef.current.on('connect', () => {
      console.log('Optimization WebSocket connected');
    });

    // Standard optimization events
    socketRef.current.on('optimization_progress', (data) => {
      setProgress({
        current: data.current,
        total: data.total,
        percent: data.percent,
        currentParams: data.current_params,
        bestSoFar: data.best_so_far,
        rate: data.rate,
        eta_seconds: data.eta_seconds,
      });
    });

    socketRef.current.on('optimization_complete', (data) => {
      console.log('Optimization complete:', data);
      setResults(data.results);
      setStatus('completed');
      setProgress(prev => ({ ...prev, percent: 100 }));
    });

    socketRef.current.on('optimization_error', (data) => {
      setError(data.message);
      setStatus('error');
    });

    // Hybrid optimization events
    socketRef.current.on('hybrid_screening_progress', (data) => {
      setHybridProgress(prev => ({
        ...prev,
        phase: 'screening',
        screening: {
          current: data.current,
          total: data.total,
          percent: data.percent,
          candidates_found: data.candidates_found
        }
      }));
    });

    socketRef.current.on('hybrid_validation_progress', (data) => {
      setHybridProgress(prev => ({
        ...prev,
        phase: 'validation',
        validation: {
          current: data.current,
          total: data.total,
          percent: data.percent
        }
      }));
    });

    socketRef.current.on('hybrid_optimization_complete', (data) => {
      console.log('Hybrid optimization complete:', data);
      setResults(data.results);
      setStatus('completed');
      setIsHybridRunning(false);
    });

    socketRef.current.on('hybrid_optimization_error', (data) => {
      setError(data.message);
      setStatus('error');
      setIsHybridRunning(false);
    });

    socketRef.current.on('disconnect', () => {
      console.log('Optimization WebSocket disconnected');
    });

    return () => {
      if (socketRef.current) {
        socketRef.current.disconnect();
        socketRef.current = null;
      }
    };
  }, []);

  const startOptimization = useCallback(async () => {
    setStatus('running');
    setIsHybridRunning(false);
    setError(null);
    setResults([]);
    setProgress({
      current: 0,
      total: totalCombinations,
      percent: 0,
      currentParams: null,
      bestSoFar: null,
    });

    // Build parameter ranges object for enabled parameters
    const enabledRanges = {};
    Object.entries(paramRanges).forEach(([key, range]) => {
      if (range.enabled) {
        enabledRanges[key] = {
          start: range.start,
          end: range.end,
          step: range.step,
        };
      }
    });

    // Ensure socket is connected
    if (!socketRef.current?.connected) {
      await new Promise((resolve) => {
        socketRef.current.once('connect', resolve);
        setTimeout(resolve, 2000);
      });
    }

    socketRef.current.emit('start_optimization', {
      symbol: config.symbol,
      start_date: config.startDate,
      end_date: config.endDate,
      initial_capital: config.initialCapital,
      commission: config.commission / 100,
      parameter_ranges: enabledRanges,
      cpu_cores: config.cpuCores,
    });
  }, [config, paramRanges, totalCombinations]);

  const startHybridOptimization = useCallback(async () => {
    setStatus('running');
    setIsHybridRunning(true);
    setError(null);
    setResults([]);
    setHybridProgress({
      phase: 'screening',
      screening: { current: 0, total: totalCombinations, percent: 0, candidates_found: 0 },
      validation: { current: 0, total: 0, percent: 0 }
    });

    // Build parameter ranges object for enabled parameters
    const enabledRanges = {};
    Object.entries(paramRanges).forEach(([key, range]) => {
      if (range.enabled) {
        enabledRanges[key] = {
          start: range.start,
          end: range.end,
          step: range.step,
        };
      }
    });

    // Ensure socket is connected
    if (!socketRef.current?.connected) {
      await new Promise((resolve) => {
        socketRef.current.once('connect', resolve);
        setTimeout(resolve, 2000);
      });
    }

    socketRef.current.emit('start_hybrid_optimization', {
      symbol: config.symbol,
      start_date: config.startDate,
      end_date: config.endDate,
      initial_capital: config.initialCapital,
      commission: config.commission / 100,
      parameter_ranges: enabledRanges,
      screening_filters: config.screeningFilters,
      top_n_candidates: 100,
      cpu_cores: config.cpuCores,
    });
  }, [config, paramRanges, totalCombinations]);

  const cancelOptimization = useCallback(() => {
    if (socketRef.current && optimizationIdRef.current) {
      if (isHybridRunning) {
        socketRef.current.emit('cancel_hybrid_optimization', {
          optimization_id: optimizationIdRef.current,
        });
      } else {
        socketRef.current.emit('cancel_optimization', {
          optimization_id: optimizationIdRef.current,
        });
      }
    }
    setStatus('idle');
    setIsHybridRunning(false);
  }, [isHybridRunning]);

  const handleConfigChange = useCallback((field, value) => {
    setConfig(prev => ({
      ...prev,
      [field]: value,
    }));
  }, []);

  const handleParamRangeChange = useCallback((param, field, value) => {
    setParamRanges(prev => ({
      ...prev,
      [param]: {
        ...prev[param],
        [field]: value,
      },
    }));
  }, []);

  const handleSort = useCallback((key) => {
    setSortConfig(prev => ({
      key,
      direction: prev.key === key && prev.direction === 'desc' ? 'asc' : 'desc',
    }));
  }, []);

  const handleResultClick = useCallback((result) => {
    setSelectedResult(result);
    setModalOpen(true);
  }, []);

  const handleUseSettings = useCallback((params) => {
    // This would ideally navigate to backtesting tab with these settings
    // For now, we'll just log it - could use a global state or context
    console.log('Use settings:', params);
    alert(`Settings copied!\n\nSwitch to Backtesting tab and update parameters:\n${Object.entries(params).map(([k, v]) => `${k}: ${v}`).join('\n')}`);
  }, []);

  // Sort results
  const sortedResults = [...results].sort((a, b) => {
    const aVal = a.statistics[sortConfig.key] || 0;
    const bVal = b.statistics[sortConfig.key] || 0;
    return sortConfig.direction === 'desc' ? bVal - aVal : aVal - bVal;
  });

  return (
    <div className="backtest-page">
      {/* Left Sidebar - Configuration */}
      <div className="backtest-sidebar">
        <OptimizationConfigPanel
          config={config}
          paramRanges={paramRanges}
          onConfigChange={handleConfigChange}
          onParamRangeChange={handleParamRangeChange}
          onStartOptimization={startOptimization}
          onStartHybridOptimization={startHybridOptimization}
          isRunning={status === 'running'}
          totalCombinations={totalCombinations}
          theme={theme}
        />
      </div>

      {/* Main Content Area */}
      <div className="backtest-main">
        {/* Error Alert */}
        {error && (
          <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError(null)}>
            {error}
          </Alert>
        )}

        {/* Progress Section */}
        {status === 'running' && (
          <OptimizationProgress
            progress={progress}
            hybridProgress={hybridProgress}
            isHybrid={isHybridRunning}
            onCancel={cancelOptimization}
            theme={theme}
          />
        )}

        {/* Results Section */}
        {status === 'idle' && (
          <Paper
            sx={{
              p: 4,
              backgroundColor: isDark ? '#1e1e1e' : '#fff',
              minHeight: '400px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              borderRadius: '12px',
            }}
          >
            <Box sx={{ textAlign: 'center', color: isDark ? 'rgba(255,255,255,0.5)' : 'rgba(0,0,0,0.5)' }}>
              <svg width="64" height="64" viewBox="0 0 24 24" fill="currentColor" style={{ opacity: 0.3 }}>
                <path d="M19.14 12.94c.04-.31.06-.63.06-.94 0-.31-.02-.63-.06-.94l2.03-1.58c.18-.14.23-.41.12-.61l-1.92-3.32c-.12-.22-.37-.29-.59-.22l-2.39.96c-.5-.38-1.03-.7-1.62-.94l-.36-2.54c-.04-.24-.24-.41-.48-.41h-3.84c-.24 0-.43.17-.47.41l-.36 2.54c-.59.24-1.13.57-1.62.94l-2.39-.96c-.22-.08-.47 0-.59.22L2.74 8.87c-.12.21-.08.47.12.61l2.03 1.58c-.04.31-.06.63-.06.94s.02.63.06.94l-2.03 1.58c-.18.14-.23.41-.12.61l1.92 3.32c.12.22.37.29.59.22l2.39-.96c.5.38 1.03.7 1.62.94l.36 2.54c.05.24.24.41.48.41h3.84c.24 0 .44-.17.47-.41l.36-2.54c.59-.24 1.13-.56 1.62-.94l2.39.96c.22.08.47 0 .59-.22l1.92-3.32c.12-.22.07-.47-.12-.61l-2.01-1.58zM12 15.6c-1.98 0-3.6-1.62-3.6-3.6s1.62-3.6 3.6-3.6 3.6 1.62 3.6 3.6-1.62 3.6-3.6 3.6z"/>
              </svg>
              <p style={{ marginTop: '16px' }}>
                Configure parameter ranges and click "Start Optimization" to find the best settings
              </p>
              <p style={{ marginTop: '8px', fontSize: '14px' }}>
                {totalCombinations} combinations will be tested
              </p>
            </Box>
          </Paper>
        )}

        {status === 'completed' && (
          <OptimizationResults
            results={sortedResults}
            sortConfig={sortConfig}
            onSort={handleSort}
            onResultClick={handleResultClick}
            theme={theme}
          />
        )}

        {/* Advanced Stats Modal */}
        <AdvancedStatsModal
          open={modalOpen}
          onClose={() => setModalOpen(false)}
          result={selectedResult}
          theme={theme}
          onUseSettings={handleUseSettings}
        />
      </div>
    </div>
  );
}

export default OptimizationPage;
