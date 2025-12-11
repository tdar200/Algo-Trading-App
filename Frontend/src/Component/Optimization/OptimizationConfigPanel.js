import React, { useState, useEffect } from 'react';
import Box from '@mui/material/Box';
import TextField from '@mui/material/TextField';
import Button from '@mui/material/Button';
import Switch from '@mui/material/Switch';
import Slider from '@mui/material/Slider';
import InputAdornment from '@mui/material/InputAdornment';
import CircularProgress from '@mui/material/CircularProgress';
import Tooltip from '@mui/material/Tooltip';
import ToggleButton from '@mui/material/ToggleButton';
import ToggleButtonGroup from '@mui/material/ToggleButtonGroup';
import StockSearch from '../Common/StockSearch';

const API_URL = process.env.REACT_APP_API_URL || "http://127.0.0.1:5000";

const PARAM_LABELS = {
  firstRetracement: { label: 'First Retracement', unit: '%', tooltip: 'Min % move from low to high to detect a swing' },
  secondRetracement: { label: 'Second Retracement', unit: '%', tooltip: 'Min % pullback to confirm the swing' },
  touchCount: { label: 'Touch Count', unit: '', tooltip: 'Min touches required before trading a level' },
  breakoutBuffer: { label: 'Breakout Buffer', unit: '%', tooltip: '% above resistance to trigger buy' },
  takeProfit: { label: 'Take Profit', unit: '%', tooltip: 'Take profit target percentage' },
  stopLoss: { label: 'Stop Loss', unit: '%', tooltip: 'Stop loss percentage' },
};

function OptimizationConfigPanel({
  config,
  paramRanges,
  onConfigChange,
  onParamRangeChange,
  onStartOptimization,
  onStartHybridOptimization,
  isRunning,
  totalCombinations,
  theme,
}) {
  const isDark = theme === 'dark';
  const [hybridAvailable, setHybridAvailable] = useState(false);
  const [gpuAvailable, setGpuAvailable] = useState(false);
  const [statusChecking, setStatusChecking] = useState(true);
  const [statusMessage, setStatusMessage] = useState('');

  // Check VectorBT/GPU availability on mount
  useEffect(() => {
    const checkStatus = async () => {
      try {
        const response = await fetch(`${API_URL}/api/gpu/status`);
        const data = await response.json();
        setHybridAvailable(data.hybrid_available);
        setGpuAvailable(data.gpu_available);
        setStatusMessage(data.message);
      } catch (error) {
        console.log('Status check failed:', error);
        setHybridAvailable(false);
        setGpuAvailable(false);
        setStatusMessage('Backend unavailable');
      } finally {
        setStatusChecking(false);
      }
    };
    checkStatus();
  }, []);

  const inputStyles = {
    '& .MuiOutlinedInput-root': {
      backgroundColor: isDark ? 'rgba(255,255,255,0.05)' : '#fff',
      '& fieldset': {
        borderColor: isDark ? 'rgba(255,255,255,0.2)' : 'rgba(0,0,0,0.23)',
      },
      '&:hover fieldset': {
        borderColor: isDark ? 'rgba(255,255,255,0.3)' : 'rgba(0,0,0,0.5)',
      },
    },
    '& .MuiInputLabel-root': {
      color: isDark ? 'rgba(255,255,255,0.7)' : 'inherit',
    },
    '& .MuiInputBase-input': {
      color: isDark ? '#fff' : 'inherit',
    },
  };

  const smallInputStyles = {
    ...inputStyles,
    '& .MuiOutlinedInput-root': {
      ...inputStyles['& .MuiOutlinedInput-root'],
      '& input': {
        padding: '8px 10px',
        fontSize: '13px',
      },
    },
    '& .MuiInputLabel-root': {
      ...inputStyles['& .MuiInputLabel-root'],
      fontSize: '12px',
    },
  };

  return (
    <Box className="config-panel">
      {/* Symbol Selection */}
      <div className="control-card">
        <div className="control-card-title">Symbol</div>
        <StockSearch
          value={config.symbol}
          onChange={(symbol) => onConfigChange('symbol', symbol)}
          theme={theme}
          label="Search Symbol"
          showQuickPicks={true}
        />
      </div>

      {/* Date Range */}
      <div className="control-card">
        <div className="control-card-title">Date Range</div>

        <TextField
          label="Start Date"
          type="date"
          size="small"
          fullWidth
          value={config.startDate}
          onChange={(e) => onConfigChange('startDate', e.target.value)}
          InputLabelProps={{ shrink: true }}
          sx={{ ...inputStyles, mb: 1.5 }}
        />

        <TextField
          label="End Date"
          type="date"
          size="small"
          fullWidth
          value={config.endDate}
          onChange={(e) => onConfigChange('endDate', e.target.value)}
          InputLabelProps={{ shrink: true }}
          sx={inputStyles}
        />
      </div>

      {/* Account Settings */}
      <div className="control-card">
        <div className="control-card-title">Account Settings</div>

        <TextField
          label="Initial Capital"
          type="number"
          size="small"
          fullWidth
          value={config.initialCapital}
          onChange={(e) => onConfigChange('initialCapital', parseFloat(e.target.value) || 0)}
          InputProps={{
            startAdornment: <InputAdornment position="start">$</InputAdornment>,
          }}
          sx={{ ...inputStyles, mb: 1.5 }}
        />

        <TextField
          label="Commission"
          type="number"
          size="small"
          fullWidth
          value={config.commission}
          onChange={(e) => onConfigChange('commission', parseFloat(e.target.value) || 0)}
          InputProps={{
            endAdornment: <InputAdornment position="end">%</InputAdornment>,
          }}
          inputProps={{ step: 0.01, min: 0, max: 10 }}
          sx={inputStyles}
        />
      </div>

      {/* Parameter Ranges */}
      <div className="control-card">
        <div className="control-card-title">Parameter Ranges</div>

        {Object.entries(paramRanges).map(([param, range]) => (
          <Tooltip key={param} title={PARAM_LABELS[param]?.tooltip || ''} placement="top">
            <Box
              sx={{
                mb: 2,
                p: 1.5,
                borderRadius: '8px',
                backgroundColor: range.enabled
                  ? (isDark ? 'rgba(25, 118, 210, 0.1)' : 'rgba(25, 118, 210, 0.05)')
                  : (isDark ? 'rgba(255,255,255,0.02)' : 'rgba(0,0,0,0.02)'),
                border: `1px solid ${range.enabled
                  ? (isDark ? 'rgba(25, 118, 210, 0.3)' : 'rgba(25, 118, 210, 0.2)')
                  : (isDark ? 'rgba(255,255,255,0.1)' : 'rgba(0,0,0,0.1)')}`,
                opacity: range.enabled ? 1 : 0.6,
              }}
            >
              <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 1 }}>
                <span style={{
                  fontSize: '13px',
                  fontWeight: 500,
                  color: isDark ? '#fff' : '#333',
                }}>
                  {PARAM_LABELS[param]?.label || param}
                </span>
                <Switch
                  size="small"
                  checked={range.enabled}
                  onChange={(e) => onParamRangeChange(param, 'enabled', e.target.checked)}
                />
              </Box>

              <Box sx={{ display: 'flex', gap: 1, alignItems: 'center' }}>
                <TextField
                  label="Start"
                  type="number"
                  size="small"
                  value={range.start}
                  onChange={(e) => onParamRangeChange(param, 'start', parseFloat(e.target.value) || 0)}
                  disabled={!range.enabled}
                  sx={{ ...smallInputStyles, flex: 1 }}
                  inputProps={{ step: param === 'touchCount' ? 1 : 0.5 }}
                />
                <span style={{ color: isDark ? 'rgba(255,255,255,0.5)' : 'rgba(0,0,0,0.5)' }}>to</span>
                <TextField
                  label="End"
                  type="number"
                  size="small"
                  value={range.end}
                  onChange={(e) => onParamRangeChange(param, 'end', parseFloat(e.target.value) || 0)}
                  disabled={!range.enabled}
                  sx={{ ...smallInputStyles, flex: 1 }}
                  inputProps={{ step: param === 'touchCount' ? 1 : 0.5 }}
                />
                <span style={{ color: isDark ? 'rgba(255,255,255,0.5)' : 'rgba(0,0,0,0.5)' }}>step</span>
                <TextField
                  label="Step"
                  type="number"
                  size="small"
                  value={range.step}
                  onChange={(e) => onParamRangeChange(param, 'step', parseFloat(e.target.value) || 1)}
                  disabled={!range.enabled}
                  sx={{ ...smallInputStyles, width: '70px' }}
                  inputProps={{ step: param === 'touchCount' ? 1 : 0.5, min: 0.1 }}
                />
              </Box>
            </Box>
          </Tooltip>
        ))}
      </div>

      {/* Processing Settings */}
      <div className="control-card">
        <div className="control-card-title">Processing</div>

        <Box sx={{ mb: 2 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 1 }}>
            <span style={{ fontSize: '13px', color: isDark ? '#fff' : '#333' }}>
              Mode
            </span>
            {statusChecking && (
              <CircularProgress size={14} sx={{ color: isDark ? '#90caf9' : '#1976d2' }} />
            )}
          </Box>
          <ToggleButtonGroup
            value={config.useHybrid ? 'hybrid' : 'standard'}
            exclusive
            onChange={(e, value) => value && onConfigChange('useHybrid', value === 'hybrid')}
            fullWidth
            size="small"
            sx={{
              '& .MuiToggleButton-root': {
                color: isDark ? 'rgba(255,255,255,0.7)' : 'inherit',
                borderColor: isDark ? 'rgba(255,255,255,0.2)' : 'rgba(0,0,0,0.12)',
                '&.Mui-selected': {
                  backgroundColor: isDark ? 'rgba(25, 118, 210, 0.3)' : 'rgba(25, 118, 210, 0.12)',
                  color: isDark ? '#90caf9' : '#1976d2',
                },
                '&.Mui-disabled': {
                  color: isDark ? 'rgba(255,255,255,0.3)' : 'rgba(0,0,0,0.26)',
                },
              },
            }}
          >
            <ToggleButton value="standard">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor" style={{ marginRight: 6 }}>
                <path d="M15 21h-2v-2h2v2zm-2-7h-2v5h2v-5zm8-2h-2v4h2v-4zm-2-2h-2v2h2v-2zm-4 0h-2v2h2v-2zm-4 0H9v2h2v-2zm-4 0H5v2h2v-2zm16-2V4h-2v2h-2V4h-2v2h-2V4h-2v2h-2V4H9v2H7V4H5v2H3v2h2v2H3v2h2v2H3v2h2v2H3v2h2v-2h2v2h2v-2h2v2h2v-2h2v2h2v-2h2v2h2v-2h-2v-2h2v-2h-2v-2h2V8zm-4 8H7V8h10v8z"/>
              </svg>
              Standard
            </ToggleButton>
            <ToggleButton value="hybrid" disabled={!hybridAvailable}>
              <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor" style={{ marginRight: 6 }}>
                <path d="M7 4h10v15H7z M4 8h3v7H4z M17 8h3v7h-3z M9 2h6v2H9z M9 20h6v2H9z"/>
              </svg>
              Hybrid {gpuAvailable ? '(GPU)' : '(CPU)'}
            </ToggleButton>
          </ToggleButtonGroup>
          <div style={{ fontSize: '11px', color: isDark ? 'rgba(255,255,255,0.5)' : 'rgba(0,0,0,0.5)', marginTop: '4px', textAlign: 'center' }}>
            {hybridAvailable
              ? (gpuAvailable ? 'GPU accelerated' : 'VectorBT screening + Backtrader validation')
              : 'VectorBT not installed - Hybrid mode unavailable'}
          </div>
        </Box>

        {/* Screening Filters (only for Hybrid mode) */}
        {config.useHybrid && hybridAvailable && (
          <Box sx={{
            mb: 2,
            p: 1.5,
            borderRadius: '8px',
            backgroundColor: isDark ? 'rgba(76, 175, 80, 0.1)' : 'rgba(76, 175, 80, 0.05)',
            border: `1px solid ${isDark ? 'rgba(76, 175, 80, 0.3)' : 'rgba(76, 175, 80, 0.2)'}`,
          }}>
            <div style={{
              fontSize: '12px',
              fontWeight: 500,
              color: '#4caf50',
              marginBottom: '8px',
              textTransform: 'uppercase',
              letterSpacing: '0.5px',
            }}>
              Screening Filters
            </div>
            <Box sx={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 1 }}>
              <TextField
                label="Min Sharpe"
                type="number"
                size="small"
                value={config.screeningFilters?.min_sharpe || 1.0}
                onChange={(e) => onConfigChange('screeningFilters', {
                  ...config.screeningFilters,
                  min_sharpe: parseFloat(e.target.value) || 0
                })}
                inputProps={{ step: 0.1, min: 0 }}
                sx={smallInputStyles}
              />
              <TextField
                label="Max DD %"
                type="number"
                size="small"
                value={config.screeningFilters?.max_drawdown || 20}
                onChange={(e) => onConfigChange('screeningFilters', {
                  ...config.screeningFilters,
                  max_drawdown: parseFloat(e.target.value) || 0
                })}
                inputProps={{ step: 1, min: 0, max: 100 }}
                sx={smallInputStyles}
              />
              <TextField
                label="Min Win %"
                type="number"
                size="small"
                value={config.screeningFilters?.min_win_rate || 45}
                onChange={(e) => onConfigChange('screeningFilters', {
                  ...config.screeningFilters,
                  min_win_rate: parseFloat(e.target.value) || 0
                })}
                inputProps={{ step: 1, min: 0, max: 100 }}
                sx={smallInputStyles}
              />
              <TextField
                label="Min Trades"
                type="number"
                size="small"
                value={config.screeningFilters?.min_trades || 50}
                onChange={(e) => onConfigChange('screeningFilters', {
                  ...config.screeningFilters,
                  min_trades: parseInt(e.target.value) || 0
                })}
                inputProps={{ step: 10, min: 0 }}
                sx={smallInputStyles}
              />
            </Box>
          </Box>
        )}

        <Box>
          <Tooltip title="Number of parallel workers for optimization" placement="top">
            <div className="slider-label">
              <span className="slider-name">CPU Workers</span>
              <span className="slider-value">{config.cpuCores}</span>
            </div>
          </Tooltip>
          <Slider
            value={config.cpuCores}
            onChange={(e, val) => onConfigChange('cpuCores', val)}
            min={1}
            max={16}
            marks={[
              { value: 1, label: '1' },
              { value: 4, label: '4' },
              { value: 8, label: '8' },
              { value: 16, label: '16' },
            ]}
            size="small"
            sx={{
              '& .MuiSlider-markLabel': {
                fontSize: '10px',
                color: isDark ? 'rgba(255,255,255,0.5)' : 'rgba(0,0,0,0.5)',
              },
            }}
          />
        </Box>
      </div>

      {/* Combinations Count */}
      <Box
        sx={{
          p: 2,
          mb: 2,
          borderRadius: '8px',
          backgroundColor: totalCombinations > 500
            ? (isDark ? 'rgba(255, 152, 0, 0.1)' : 'rgba(255, 152, 0, 0.1)')
            : (isDark ? 'rgba(76, 175, 80, 0.1)' : 'rgba(76, 175, 80, 0.1)'),
          border: `1px solid ${totalCombinations > 500
            ? 'rgba(255, 152, 0, 0.3)'
            : 'rgba(76, 175, 80, 0.3)'}`,
          textAlign: 'center',
        }}
      >
        <div style={{
          fontSize: '24px',
          fontWeight: 700,
          color: totalCombinations > 500 ? '#ff9800' : '#4caf50',
        }}>
          {totalCombinations.toLocaleString()}
        </div>
        <div style={{
          fontSize: '12px',
          color: isDark ? 'rgba(255,255,255,0.7)' : 'rgba(0,0,0,0.6)',
          marginTop: '4px',
        }}>
          Total Combinations
        </div>
        {totalCombinations > 500 && (
          <div style={{
            fontSize: '11px',
            color: '#ff9800',
            marginTop: '8px',
          }}>
            Using {config.cpuCores} worker{config.cpuCores > 1 ? 's' : ''} for parallel processing
          </div>
        )}
      </Box>

      {/* Run Button */}
      <Button
        variant="contained"
        fullWidth
        size="large"
        onClick={config.useHybrid && hybridAvailable ? onStartHybridOptimization : onStartOptimization}
        disabled={isRunning || totalCombinations < 1}
        sx={{
          py: 1.5,
          fontWeight: 600,
          backgroundColor: config.useHybrid && hybridAvailable ? '#4caf50' : '#2962FF',
          '&:hover': {
            backgroundColor: config.useHybrid && hybridAvailable ? '#388e3c' : '#1e4bd8',
          },
        }}
      >
        {isRunning ? (
          <>
            <CircularProgress size={20} sx={{ color: '#fff', mr: 1 }} />
            {config.useHybrid ? 'Running Hybrid...' : 'Optimizing...'}
          </>
        ) : (
          config.useHybrid && hybridAvailable ? 'Start Hybrid Optimization' : 'Start Optimization'
        )}
      </Button>
    </Box>
  );
}

export default OptimizationConfigPanel;
