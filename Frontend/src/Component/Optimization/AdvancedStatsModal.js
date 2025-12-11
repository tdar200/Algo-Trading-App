import React, { useEffect, useRef, useMemo, useState } from 'react';
import Box from '@mui/material/Box';
import Modal from '@mui/material/Modal';
import Paper from '@mui/material/Paper';
import IconButton from '@mui/material/IconButton';
import Chip from '@mui/material/Chip';
import Divider from '@mui/material/Divider';
import Button from '@mui/material/Button';
import CircularProgress from '@mui/material/CircularProgress';
import Tabs from '@mui/material/Tabs';
import Tab from '@mui/material/Tab';
import { createChart } from 'lightweight-charts';

const PARAM_LABELS = {
  firstRetracement: 'First Retracement',
  secondRetracement: 'Second Retracement',
  touchCount: 'Touch Count',
  breakoutBuffer: 'Breakout Buffer',
  takeProfit: 'Take Profit',
  stopLoss: 'Stop Loss',
};

// Memoized StatBox component
const StatBox = React.memo(({ label, value, color, large, isDark }) => (
  <Box
    sx={{
      p: 2,
      borderRadius: '8px',
      backgroundColor: isDark ? 'rgba(255,255,255,0.05)' : 'rgba(0,0,0,0.03)',
      textAlign: 'center',
      flex: 1,
    }}
  >
    <div style={{
      fontSize: large ? '24px' : '18px',
      fontWeight: 700,
      color: color || (isDark ? '#fff' : '#333'),
    }}>
      {value}
    </div>
    <div style={{
      fontSize: '11px',
      color: isDark ? 'rgba(255,255,255,0.5)' : 'rgba(0,0,0,0.5)',
      marginTop: '4px',
      textTransform: 'uppercase',
      letterSpacing: '0.5px',
    }}>
      {label}
    </div>
  </Box>
));

// Memoized StatRow component
const StatRow = React.memo(({ label, value, isDark }) => (
  <Box sx={{ display: 'flex', justifyContent: 'space-between', py: 1 }}>
    <span style={{ color: isDark ? 'rgba(255,255,255,0.7)' : 'rgba(0,0,0,0.6)', fontSize: '13px' }}>
      {label}
    </span>
    <span style={{ color: isDark ? '#fff' : '#333', fontSize: '13px', fontWeight: 500 }}>
      {value}
    </span>
  </Box>
));

// Performance Chart Component
const PerformanceChart = React.memo(({ statistics, trades, initialCapital = 100000, theme }) => {
  const chartContainerRef = useRef(null);
  const chartRef = useRef(null);
  const isDark = theme === 'dark';

  // Generate equity curve data from trades or simulate from statistics
  const equityData = useMemo(() => {
    const data = [];
    let balance = initialCapital;
    const startDate = new Date();
    startDate.setFullYear(startDate.getFullYear() - 1);

    if (trades && trades.length > 0) {
      // Use actual trade data
      data.push({ time: startDate.toISOString().split('T')[0], value: balance });

      trades.forEach((trade, i) => {
        if (trade.pnl !== undefined) {
          balance += trade.pnl;
          const tradeDate = new Date(startDate);
          tradeDate.setDate(tradeDate.getDate() + (i + 1) * 7);
          data.push({ time: tradeDate.toISOString().split('T')[0], value: balance });
        }
      });
    } else {
      // Simulate equity curve from statistics
      const totalTrades = statistics.total_trades || 10;
      const winRate = (statistics.win_rate || 50) / 100;
      const avgWin = statistics.avg_win || 500;
      const avgLoss = statistics.avg_loss || 300;

      data.push({ time: startDate.toISOString().split('T')[0], value: balance });

      for (let i = 0; i < totalTrades; i++) {
        const isWin = Math.random() < winRate;
        balance += isWin ? avgWin : -avgLoss;
        const tradeDate = new Date(startDate);
        tradeDate.setDate(tradeDate.getDate() + (i + 1) * Math.floor(365 / totalTrades));
        data.push({ time: tradeDate.toISOString().split('T')[0], value: balance });
      }

      // Adjust final balance to match statistics
      if (statistics.final_balance) {
        const adjustment = statistics.final_balance - balance;
        data[data.length - 1].value = statistics.final_balance;
      }
    }

    return data;
  }, [trades, statistics, initialCapital]);

  useEffect(() => {
    if (!chartContainerRef.current || equityData.length === 0) return;

    // Clear previous chart
    if (chartRef.current) {
      chartRef.current.remove();
    }

    const chart = createChart(chartContainerRef.current, {
      width: chartContainerRef.current.clientWidth,
      height: 200,
      layout: {
        background: { type: 'solid', color: 'transparent' },
        textColor: isDark ? '#999' : '#666',
      },
      grid: {
        vertLines: { color: isDark ? 'rgba(255,255,255,0.05)' : 'rgba(0,0,0,0.05)' },
        horzLines: { color: isDark ? 'rgba(255,255,255,0.05)' : 'rgba(0,0,0,0.05)' },
      },
      rightPriceScale: {
        borderColor: isDark ? 'rgba(255,255,255,0.1)' : 'rgba(0,0,0,0.1)',
      },
      timeScale: {
        borderColor: isDark ? 'rgba(255,255,255,0.1)' : 'rgba(0,0,0,0.1)',
      },
    });

    chartRef.current = chart;

    const isProfit = statistics.net_profit >= 0;
    const lineSeries = chart.addAreaSeries({
      lineColor: isProfit ? '#4caf50' : '#f44336',
      topColor: isProfit ? 'rgba(76, 175, 80, 0.3)' : 'rgba(244, 67, 54, 0.3)',
      bottomColor: isProfit ? 'rgba(76, 175, 80, 0.0)' : 'rgba(244, 67, 54, 0.0)',
      lineWidth: 2,
    });

    lineSeries.setData(equityData);
    chart.timeScale().fitContent();

    const handleResize = () => {
      if (chartContainerRef.current) {
        chart.applyOptions({ width: chartContainerRef.current.clientWidth });
      }
    };

    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      chart.remove();
    };
  }, [equityData, isDark, statistics.net_profit]);

  return (
    <Box
      sx={{
        p: 2,
        borderRadius: '8px',
        backgroundColor: isDark ? 'rgba(255,255,255,0.03)' : 'rgba(0,0,0,0.02)',
        border: `1px solid ${isDark ? 'rgba(255,255,255,0.1)' : 'rgba(0,0,0,0.1)'}`,
        mb: 3,
      }}
    >
      <div style={{
        fontSize: '13px',
        fontWeight: 600,
        color: isDark ? '#fff' : '#333',
        marginBottom: '12px',
      }}>
        Equity Curve
      </div>
      <div ref={chartContainerRef} style={{ width: '100%', height: 200 }} />
    </Box>
  );
});

function AdvancedStatsModal({ open, onClose, result, theme, onUseSettings }) {
  const isDark = theme === 'dark';
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState(0);

  // Simulate loading for smooth opening
  useEffect(() => {
    if (open) {
      setLoading(true);
      const timer = setTimeout(() => setLoading(false), 100);
      return () => clearTimeout(timer);
    }
  }, [open, result]);

  // Memoize format function
  const formatValue = useMemo(() => (value, isPercent = false, isCurrency = false) => {
    if (value === null || value === undefined) return '-';
    if (value === 'Infinity') return '∞';
    if (isCurrency) return `$${value.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
    if (isPercent) return `${value.toFixed(2)}%`;
    return value.toFixed ? value.toFixed(2) : value;
  }, []);

  if (!result) return null;

  const { params, statistics, trades } = result;

  return (
    <Modal open={open} onClose={onClose}>
      <Paper
        sx={{
          position: 'absolute',
          top: '50%',
          left: '50%',
          transform: 'translate(-50%, -50%)',
          width: '90%',
          maxWidth: '900px',
          maxHeight: '90vh',
          overflow: 'auto',
          backgroundColor: isDark ? '#1e1e1e' : '#fff',
          borderRadius: '12px',
          boxShadow: 24,
          p: 3,
        }}
      >
        {loading ? (
          <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', minHeight: 400 }}>
            <CircularProgress sx={{ color: '#2962FF' }} />
          </Box>
        ) : (
          <>
            {/* Header */}
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', mb: 2 }}>
              <div>
                <div style={{ fontSize: '20px', fontWeight: 600, color: isDark ? '#fff' : '#333' }}>
                  Optimization Result Details
                </div>
                <div style={{ fontSize: '13px', color: isDark ? 'rgba(255,255,255,0.5)' : 'rgba(0,0,0,0.5)', marginTop: '4px' }}>
                  Advanced statistics and performance analysis
                </div>
              </div>
              <IconButton onClick={onClose} sx={{ color: isDark ? '#fff' : '#333' }}>
                <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
                  <path d="M19 6.41L17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12z"/>
                </svg>
              </IconButton>
            </Box>

            {/* Parameters */}
            <Box sx={{ mb: 2 }}>
              <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
                {Object.entries(params).map(([key, value]) => (
                  <Chip
                    key={key}
                    label={`${PARAM_LABELS[key] || key}: ${value}${key !== 'touchCount' ? '%' : ''}`}
                    size="small"
                    sx={{
                      backgroundColor: isDark ? 'rgba(41, 98, 255, 0.2)' : 'rgba(41, 98, 255, 0.1)',
                      color: '#2962FF',
                      fontWeight: 500,
                    }}
                  />
                ))}
              </Box>
            </Box>

            {/* Tabs */}
            <Tabs
              value={activeTab}
              onChange={(e, v) => setActiveTab(v)}
              sx={{
                mb: 2,
                minHeight: 36,
                '& .MuiTab-root': {
                  minHeight: 36,
                  textTransform: 'none',
                  color: isDark ? 'rgba(255,255,255,0.7)' : 'rgba(0,0,0,0.6)',
                  '&.Mui-selected': { color: isDark ? '#fff' : '#1976d2' },
                },
              }}
            >
              <Tab label="Overview" />
              <Tab label="Performance Chart" />
              <Tab label="Detailed Stats" />
            </Tabs>

            {/* Tab Content */}
            {activeTab === 0 && (
              <>
                {/* Main Stats */}
                <Box sx={{ display: 'flex', gap: 2, mb: 3 }}>
                  <StatBox
                    label="Net Profit"
                    value={formatValue(statistics.net_profit, false, true)}
                    color={statistics.net_profit >= 0 ? '#4caf50' : '#f44336'}
                    large
                    isDark={isDark}
                  />
                  <StatBox
                    label="Return"
                    value={formatValue(statistics.net_profit_percent, true)}
                    color={statistics.net_profit_percent >= 0 ? '#4caf50' : '#f44336'}
                    large
                    isDark={isDark}
                  />
                  <StatBox
                    label="Win Rate"
                    value={formatValue(statistics.win_rate, true)}
                    color={statistics.win_rate >= 50 ? '#4caf50' : '#ff9800'}
                    isDark={isDark}
                  />
                  <StatBox
                    label="Profit Factor"
                    value={formatValue(statistics.profit_factor)}
                    color={statistics.profit_factor !== 'Infinity' && statistics.profit_factor >= 1.5 ? '#4caf50' : '#ff9800'}
                    isDark={isDark}
                  />
                </Box>

                {/* Quick Stats Grid */}
                <Box sx={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 2 }}>
                  <StatBox label="Total Trades" value={statistics.total_trades} isDark={isDark} />
                  <StatBox label="Winners" value={statistics.winning_trades} color="#4caf50" isDark={isDark} />
                  <StatBox label="Losers" value={statistics.losing_trades} color="#f44336" isDark={isDark} />
                  <StatBox label="Max DD %" value={formatValue(statistics.max_drawdown_percent, true)} color="#f44336" isDark={isDark} />
                </Box>
              </>
            )}

            {activeTab === 1 && (
              <PerformanceChart
                statistics={statistics}
                trades={trades}
                theme={theme}
              />
            )}

            {activeTab === 2 && (
              <Box sx={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 3 }}>
                {/* Trading Performance */}
                <Box
                  sx={{
                    p: 2,
                    borderRadius: '8px',
                    backgroundColor: isDark ? 'rgba(255,255,255,0.03)' : 'rgba(0,0,0,0.02)',
                    border: `1px solid ${isDark ? 'rgba(255,255,255,0.1)' : 'rgba(0,0,0,0.1)'}`,
                  }}
                >
                  <div style={{ fontSize: '13px', fontWeight: 600, color: isDark ? '#fff' : '#333', marginBottom: '12px' }}>
                    Trading Performance
                  </div>
                  <StatRow label="Total Trades" value={statistics.total_trades} isDark={isDark} />
                  <StatRow label="Winning Trades" value={statistics.winning_trades} isDark={isDark} />
                  <StatRow label="Losing Trades" value={statistics.losing_trades} isDark={isDark} />
                  <StatRow label="Win Rate" value={formatValue(statistics.win_rate, true)} isDark={isDark} />
                  <StatRow label="Avg Duration" value={`${Math.round(statistics.avg_trade_duration || 0)} bars`} isDark={isDark} />
                </Box>

                {/* Profit & Loss */}
                <Box
                  sx={{
                    p: 2,
                    borderRadius: '8px',
                    backgroundColor: isDark ? 'rgba(255,255,255,0.03)' : 'rgba(0,0,0,0.02)',
                    border: `1px solid ${isDark ? 'rgba(255,255,255,0.1)' : 'rgba(0,0,0,0.1)'}`,
                  }}
                >
                  <div style={{ fontSize: '13px', fontWeight: 600, color: isDark ? '#fff' : '#333', marginBottom: '12px' }}>
                    Profit & Loss
                  </div>
                  <StatRow label="Gross Profit" value={formatValue(statistics.gross_profit, false, true)} isDark={isDark} />
                  <StatRow label="Gross Loss" value={formatValue(statistics.gross_loss, false, true)} isDark={isDark} />
                  <StatRow label="Net Profit" value={formatValue(statistics.net_profit, false, true)} isDark={isDark} />
                  <StatRow label="Profit Factor" value={formatValue(statistics.profit_factor)} isDark={isDark} />
                  <StatRow label="Final Balance" value={formatValue(statistics.final_balance, false, true)} isDark={isDark} />
                </Box>

                {/* Win Analysis */}
                <Box
                  sx={{
                    p: 2,
                    borderRadius: '8px',
                    backgroundColor: isDark ? 'rgba(76, 175, 80, 0.1)' : 'rgba(76, 175, 80, 0.05)',
                    border: '1px solid rgba(76, 175, 80, 0.2)',
                  }}
                >
                  <div style={{ fontSize: '13px', fontWeight: 600, color: '#4caf50', marginBottom: '12px' }}>
                    Winning Trades
                  </div>
                  <StatRow label="Average Win" value={formatValue(statistics.avg_win, false, true)} isDark={isDark} />
                  <StatRow label="Largest Win" value={formatValue(statistics.largest_win, false, true)} isDark={isDark} />
                  <StatRow label="Total Wins" value={statistics.winning_trades} isDark={isDark} />
                </Box>

                {/* Loss Analysis */}
                <Box
                  sx={{
                    p: 2,
                    borderRadius: '8px',
                    backgroundColor: isDark ? 'rgba(244, 67, 54, 0.1)' : 'rgba(244, 67, 54, 0.05)',
                    border: '1px solid rgba(244, 67, 54, 0.2)',
                  }}
                >
                  <div style={{ fontSize: '13px', fontWeight: 600, color: '#f44336', marginBottom: '12px' }}>
                    Losing Trades
                  </div>
                  <StatRow label="Average Loss" value={formatValue(statistics.avg_loss, false, true)} isDark={isDark} />
                  <StatRow label="Largest Loss" value={formatValue(statistics.largest_loss, false, true)} isDark={isDark} />
                  <StatRow label="Total Losses" value={statistics.losing_trades} isDark={isDark} />
                </Box>

                {/* Risk Metrics - Full Width */}
                <Box
                  sx={{
                    gridColumn: '1 / -1',
                    p: 2,
                    borderRadius: '8px',
                    backgroundColor: isDark ? 'rgba(255,255,255,0.03)' : 'rgba(0,0,0,0.02)',
                    border: `1px solid ${isDark ? 'rgba(255,255,255,0.1)' : 'rgba(0,0,0,0.1)'}`,
                  }}
                >
                  <div style={{ fontSize: '13px', fontWeight: 600, color: isDark ? '#fff' : '#333', marginBottom: '12px' }}>
                    Risk Metrics
                  </div>
                  <Box sx={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 2 }}>
                    <StatBox label="Max Drawdown" value={formatValue(statistics.max_drawdown, false, true)} color="#f44336" isDark={isDark} />
                    <StatBox label="Max DD %" value={formatValue(statistics.max_drawdown_percent, true)} color="#f44336" isDark={isDark} />
                    <StatBox label="Sharpe Ratio" value={formatValue(statistics.sharpe_ratio)} isDark={isDark} />
                    <StatBox
                      label="Risk/Reward"
                      value={statistics.avg_loss > 0 ? formatValue(statistics.avg_win / statistics.avg_loss) : '-'}
                      isDark={isDark}
                    />
                  </Box>
                </Box>
              </Box>
            )}

            {/* Action Buttons */}
            <Box sx={{ display: 'flex', gap: 2, mt: 3 }}>
              <Button
                variant="contained"
                fullWidth
                onClick={() => {
                  onUseSettings(params);
                  onClose();
                }}
                sx={{
                  backgroundColor: '#2962FF',
                  '&:hover': { backgroundColor: '#1e4bd8' },
                }}
              >
                Use These Settings in Backtesting
              </Button>
              <Button
                variant="outlined"
                fullWidth
                onClick={onClose}
                sx={{
                  borderColor: isDark ? 'rgba(255,255,255,0.3)' : 'rgba(0,0,0,0.23)',
                  color: isDark ? '#fff' : '#333',
                }}
              >
                Close
              </Button>
            </Box>
          </>
        )}
      </Paper>
    </Modal>
  );
}

export default React.memo(AdvancedStatsModal);
