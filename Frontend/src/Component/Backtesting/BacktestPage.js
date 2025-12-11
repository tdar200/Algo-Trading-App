import React, { useState, useCallback, useEffect, useRef } from 'react';
import axios from 'axios';
import { io } from 'socket.io-client';
import Box from '@mui/material/Box';
import Paper from '@mui/material/Paper';
import Tabs from '@mui/material/Tabs';
import Tab from '@mui/material/Tab';
import CircularProgress from '@mui/material/CircularProgress';
import Alert from '@mui/material/Alert';
import BacktestConfigPanel from './BacktestConfigPanel';
import BacktestChart from './BacktestChart';
import PlaybackControls from './PlaybackControls';
import TradeStats from './BacktestResults/TradeStats';
import EquityChart from './BacktestResults/EquityChart';
import TradeHistory from './BacktestResults/TradeHistory';
import RiskMetrics from './BacktestResults/RiskMetrics';

const API_URL = process.env.REACT_APP_API_URL || "http://127.0.0.1:5000";

function BacktestPage({ theme }) {
  const isDark = theme === 'dark';
  const socketRef = useRef(null);
  const backtestIdRef = useRef(null);

  // Configuration state
  const [config, setConfig] = useState({
    symbol: 'AAPL',
    startDate: new Date(Date.now() - 365 * 24 * 60 * 60 * 1000).toISOString().split('T')[0],
    endDate: new Date().toISOString().split('T')[0],
    initialCapital: 100000,
    commission: 0.1,
    strategy: 'SupResStrategy',
    strategyParams: {
      firstRetracement: 5,
      secondRetracement: 5,
      touchCount: 1,
      levelRange: 0.001,
      breakoutBuffer: 0,
      takeProfit: 10,
      stopLoss: 5,
    },
    mode: 'fast',
  });

  // Backtest state
  const [status, setStatus] = useState('idle'); // idle, running, completed, error
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);
  const [resultsTab, setResultsTab] = useState(0);

  // Visual mode state
  const [visualState, setVisualState] = useState({
    isPlaying: false,
    speed: 1,
    currentBarIndex: 0,
    totalBars: 0,
  });

  // Visual mode data
  const [visualData, setVisualData] = useState({
    visibleCandles: [],
    trades: [],
    equityCurve: [],
    balance: 0,
    equity: 0,
    openTrade: null,
  });

  // Initialize WebSocket connection for visual mode
  useEffect(() => {
    if (config.mode === 'visual' && !socketRef.current) {
      socketRef.current = io(API_URL, {
        transports: ['websocket', 'polling'],
      });

      socketRef.current.on('connect', () => {
        console.log('WebSocket connected');
      });

      socketRef.current.on('backtest_initialized', (data) => {
        console.log('Backtest initialized:', data);
        backtestIdRef.current = data.backtest_id;
        setVisualState(prev => ({
          ...prev,
          totalBars: data.total_bars,
          currentBarIndex: 0,
        }));
        setStatus('completed');
      });

      socketRef.current.on('bar_update', (data) => {
        setVisualState(prev => ({
          ...prev,
          currentBarIndex: data.bar_index,
          totalBars: data.total_bars,
        }));

        setVisualData(prev => ({
          visibleCandles: [...prev.visibleCandles, data.candle],
          trades: data.new_trades?.length ? [...prev.trades, ...data.new_trades] : prev.trades,
          equityCurve: [...prev.equityCurve, { date: data.candle.date, balance: data.balance, equity: data.equity }],
          balance: data.balance,
          equity: data.equity,
          openTrade: data.open_trade,
        }));
      });

      socketRef.current.on('backtest_complete', (data) => {
        console.log('Backtest complete:', data);
        setResults({
          statistics: data.statistics,
          trades: data.trades,
          equity_curve: data.equity_curve,
          stock_data: visualData.visibleCandles,
        });
        setVisualState(prev => ({ ...prev, isPlaying: false }));
      });

      socketRef.current.on('error', (data) => {
        setError(data.message);
        setStatus('error');
      });

      socketRef.current.on('disconnect', () => {
        console.log('WebSocket disconnected');
      });
    }

    return () => {
      if (socketRef.current) {
        socketRef.current.disconnect();
        socketRef.current = null;
      }
    };
  }, [config.mode]);

  const runBacktest = useCallback(async () => {
    setStatus('running');
    setError(null);
    setResults(null);
    setVisualData({
      visibleCandles: [],
      trades: [],
      equityCurve: [],
      balance: config.initialCapital,
      equity: config.initialCapital,
      openTrade: null,
    });

    if (config.mode === 'visual') {
      // Visual mode - use WebSocket
      if (!socketRef.current) {
        socketRef.current = io(API_URL, {
          transports: ['websocket', 'polling'],
        });
      }

      // Wait for connection
      if (!socketRef.current.connected) {
        await new Promise((resolve) => {
          socketRef.current.once('connect', resolve);
          setTimeout(resolve, 2000); // Timeout after 2s
        });
      }

      socketRef.current.emit('start_visual_backtest', {
        symbol: config.symbol,
        start_date: config.startDate,
        end_date: config.endDate,
        initial_capital: config.initialCapital,
        commission: config.commission / 100,
        strategy: config.strategy,
        strategy_params: {
          FIRST_RETRACEMENT: config.strategyParams.firstRetracement,
          SECOND_RETRACEMENT: config.strategyParams.secondRetracement,
          TOUCH_COUNT: config.strategyParams.touchCount,
          RES_SUP_RANGE: config.strategyParams.levelRange,
          BREAKOUT_BUFFER: config.strategyParams.breakoutBuffer,
          TAKE_PROFIT: config.strategyParams.takeProfit,
          STOP_LOSS: config.strategyParams.stopLoss,
          RISK_PERCENTAGE: 5,
        },
      });
    } else {
      // Fast mode - use REST API
      try {
        const response = await axios.post(`${API_URL}/api/backtest/run`, {
          symbol: config.symbol,
          start_date: config.startDate,
          end_date: config.endDate,
          initial_capital: config.initialCapital,
          commission: config.commission / 100,
          strategy: config.strategy,
          strategy_params: {
            FIRST_RETRACEMENT: config.strategyParams.firstRetracement,
            SECOND_RETRACEMENT: config.strategyParams.secondRetracement,
            TOUCH_COUNT: config.strategyParams.touchCount,
            RES_SUP_RANGE: config.strategyParams.levelRange,
            BREAKOUT_BUFFER: config.strategyParams.breakoutBuffer,
            TAKE_PROFIT: config.strategyParams.takeProfit,
            STOP_LOSS: config.strategyParams.stopLoss,
          },
        });

        setResults(response.data);
        setStatus('completed');
      } catch (err) {
        const errorMessage = err.response?.data?.error || err.message || 'Backtest failed';
        setError(errorMessage);
        setStatus('error');
      }
    }
  }, [config]);

  // Handle visual mode playback controls
  const handleVisualStateChange = useCallback((updater) => {
    setVisualState(prev => {
      const newState = typeof updater === 'function' ? updater(prev) : updater;

      // Send control commands to server
      if (socketRef.current && backtestIdRef.current) {
        if (newState.isPlaying !== prev.isPlaying) {
          socketRef.current.emit('control', {
            backtest_id: backtestIdRef.current,
            action: newState.isPlaying ? 'play' : 'pause',
          });
        }

        if (newState.speed !== prev.speed) {
          socketRef.current.emit('control', {
            backtest_id: backtestIdRef.current,
            action: 'set_speed',
            value: newState.speed,
          });
        }

        if (newState.currentBarIndex !== prev.currentBarIndex && !newState.isPlaying) {
          socketRef.current.emit('control', {
            backtest_id: backtestIdRef.current,
            action: 'step',
          });
        }
      }

      return newState;
    });
  }, []);

  const handleConfigChange = useCallback((field, value) => {
    setConfig(prev => ({
      ...prev,
      [field]: value,
    }));
  }, []);

  const handleStrategyParamChange = useCallback((param, value) => {
    setConfig(prev => ({
      ...prev,
      strategyParams: {
        ...prev.strategyParams,
        [param]: value,
      },
    }));
  }, []);

  // Get display data based on mode
  const displayData = config.mode === 'visual'
    ? {
        stock_data: visualData.visibleCandles,
        trades: visualData.trades,
        equity_curve: visualData.equityCurve,
        statistics: results?.statistics || null,
        risk_metrics: results?.risk_metrics || null,
      }
    : results;

  return (
    <div className="backtest-page">
      {/* Left Sidebar - Configuration */}
      <div className="backtest-sidebar">
        <BacktestConfigPanel
          config={config}
          onConfigChange={handleConfigChange}
          onStrategyParamChange={handleStrategyParamChange}
          onRunBacktest={runBacktest}
          isRunning={status === 'running'}
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

        {/* Chart Area */}
        <Paper
          className="backtest-chart-container"
          sx={{
            p: 2,
            mb: 2,
            backgroundColor: isDark ? '#1e1e1e' : '#fff',
            minHeight: '400px',
            display: 'flex',
            flexDirection: 'column',
          }}
        >
          {status === 'idle' && (
            <Box
              sx={{
                flex: 1,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                color: isDark ? 'rgba(255,255,255,0.5)' : 'rgba(0,0,0,0.5)',
              }}
            >
              <div style={{ textAlign: 'center' }}>
                <svg width="64" height="64" viewBox="0 0 24 24" fill="currentColor" style={{ opacity: 0.3 }}>
                  <path d="M3 13h2v-2H3v2zm0 4h2v-2H3v2zm0-8h2V7H3v2zm4 4h14v-2H7v2zm0 4h14v-2H7v2zM7 7v2h14V7H7z"/>
                </svg>
                <p style={{ marginTop: '16px' }}>Configure settings and click "Run Backtest" to start</p>
              </div>
            </Box>
          )}

          {status === 'running' && (
            <Box
              sx={{
                flex: 1,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
              }}
            >
              <CircularProgress sx={{ color: '#2962FF' }} />
              <span style={{ marginLeft: '16px', color: isDark ? '#fff' : '#000' }}>
                {config.mode === 'visual' ? 'Initializing visual backtest...' : 'Running backtest...'}
              </span>
            </Box>
          )}

          {status === 'completed' && displayData && (
            <>
              <BacktestChart
                data={displayData.stock_data || []}
                trades={displayData.trades || []}
                theme={theme}
              />
              {config.mode === 'visual' && (
                <PlaybackControls
                  visualState={visualState}
                  onStateChange={handleVisualStateChange}
                  theme={theme}
                />
              )}
            </>
          )}
        </Paper>

        {/* Results Panels */}
        {status === 'completed' && displayData && (
          <Paper
            className="backtest-results-container"
            sx={{
              backgroundColor: isDark ? '#1e1e1e' : '#fff',
            }}
          >
            <Tabs
              value={resultsTab}
              onChange={(e, newValue) => setResultsTab(newValue)}
              sx={{
                borderBottom: 1,
                borderColor: isDark ? 'rgba(255,255,255,0.1)' : 'divider',
                '& .MuiTab-root': {
                  textTransform: 'none',
                  color: isDark ? 'rgba(255,255,255,0.7)' : 'inherit',
                  '&.Mui-selected': {
                    color: isDark ? '#fff' : '#1976d2',
                  },
                },
              }}
            >
              <Tab label="Trade Statistics" />
              <Tab label="Risk Metrics" />
              <Tab label="Equity Curve" />
              <Tab label="Trade History" />
            </Tabs>

            <Box sx={{ p: 2 }}>
              {resultsTab === 0 && (
                <TradeStats statistics={displayData.statistics} theme={theme} />
              )}
              {resultsTab === 1 && (
                <RiskMetrics riskMetrics={displayData.risk_metrics} theme={theme} />
              )}
              {resultsTab === 2 && (
                <EquityChart equityCurve={displayData.equity_curve || []} theme={theme} />
              )}
              {resultsTab === 3 && (
                <TradeHistory trades={displayData.trades || []} theme={theme} />
              )}
            </Box>
          </Paper>
        )}
      </div>
    </div>
  );
}

export default BacktestPage;
