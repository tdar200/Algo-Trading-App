import React from 'react';
import Box from '@mui/material/Box';
import TextField from '@mui/material/TextField';
import Slider from '@mui/material/Slider';
import Button from '@mui/material/Button';
import FormControl from '@mui/material/FormControl';
import InputLabel from '@mui/material/InputLabel';
import Select from '@mui/material/Select';
import MenuItem from '@mui/material/MenuItem';
import ToggleButton from '@mui/material/ToggleButton';
import ToggleButtonGroup from '@mui/material/ToggleButtonGroup';
import Tooltip from '@mui/material/Tooltip';
import InputAdornment from '@mui/material/InputAdornment';
import CircularProgress from '@mui/material/CircularProgress';
import StockSearch from '../Common/StockSearch';

const STRATEGIES = [
  {
    id: 'SupResStrategy',
    name: 'Support/Resistance Breakout',
    description: 'Trades breakouts above resistance and breakdowns below support',
  },
];

function BacktestConfigPanel({
  config,
  onConfigChange,
  onStrategyParamChange,
  onRunBacktest,
  isRunning,
  theme,
}) {
  const isDark = theme === 'dark';

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

      {/* Strategy Selection */}
      <div className="control-card">
        <div className="control-card-title">Strategy</div>

        <FormControl fullWidth size="small" sx={inputStyles}>
          <InputLabel>Strategy</InputLabel>
          <Select
            value={config.strategy}
            label="Strategy"
            onChange={(e) => onConfigChange('strategy', e.target.value)}
          >
            {STRATEGIES.map((strategy) => (
              <MenuItem key={strategy.id} value={strategy.id}>
                {strategy.name}
              </MenuItem>
            ))}
          </Select>
        </FormControl>

        {/* Strategy Parameters */}
        <Box sx={{ mt: 2 }}>
          <div className="parameter-group">
            <Tooltip title="Minimum % move from low to high to detect a swing" placement="top">
              <div className="slider-label">
                <span className="slider-name">First Retracement</span>
                <span className="slider-value">{config.strategyParams.firstRetracement}%</span>
              </div>
            </Tooltip>
            <Slider
              value={config.strategyParams.firstRetracement}
              onChange={(e, val) => onStrategyParamChange('firstRetracement', val)}
              min={1}
              max={50}
              size="small"
            />
          </div>

          <div className="parameter-group">
            <Tooltip title="Minimum % pullback to confirm the swing" placement="top">
              <div className="slider-label">
                <span className="slider-name">Second Retracement</span>
                <span className="slider-value">{config.strategyParams.secondRetracement}%</span>
              </div>
            </Tooltip>
            <Slider
              value={config.strategyParams.secondRetracement}
              onChange={(e, val) => onStrategyParamChange('secondRetracement', val)}
              min={1}
              max={50}
              size="small"
            />
          </div>

          <div className="parameter-group">
            <Tooltip title="Minimum touches required before trading a level" placement="top">
              <div className="slider-label">
                <span className="slider-name">Touch Count</span>
                <span className="slider-value">{config.strategyParams.touchCount}</span>
              </div>
            </Tooltip>
            <Slider
              value={config.strategyParams.touchCount}
              onChange={(e, val) => onStrategyParamChange('touchCount', val)}
              min={1}
              max={10}
              size="small"
            />
          </div>

          <div className="parameter-group">
            <Tooltip title="% above resistance level to trigger buy (e.g., 10% means buy at resistance + 10%)" placement="top">
              <div className="slider-label">
                <span className="slider-name">Breakout Buffer</span>
                <span className="slider-value">{config.strategyParams.breakoutBuffer}%</span>
              </div>
            </Tooltip>
            <Slider
              value={config.strategyParams.breakoutBuffer}
              onChange={(e, val) => onStrategyParamChange('breakoutBuffer', val)}
              min={0}
              max={20}
              step={0.5}
              size="small"
            />
          </div>

          <div className="parameter-group">
            <Tooltip title="Take profit target percentage" placement="top">
              <div className="slider-label">
                <span className="slider-name">Take Profit</span>
                <span className="slider-value">{config.strategyParams.takeProfit}%</span>
              </div>
            </Tooltip>
            <Slider
              value={config.strategyParams.takeProfit}
              onChange={(e, val) => onStrategyParamChange('takeProfit', val)}
              min={1}
              max={50}
              size="small"
            />
          </div>

          <div className="parameter-group">
            <Tooltip title="Stop loss percentage" placement="top">
              <div className="slider-label">
                <span className="slider-name">Stop Loss</span>
                <span className="slider-value">{config.strategyParams.stopLoss}%</span>
              </div>
            </Tooltip>
            <Slider
              value={config.strategyParams.stopLoss}
              onChange={(e, val) => onStrategyParamChange('stopLoss', val)}
              min={1}
              max={30}
              size="small"
            />
          </div>
        </Box>
      </div>

      {/* Backtest Mode */}
      <div className="control-card">
        <div className="control-card-title">Mode</div>

        <ToggleButtonGroup
          value={config.mode}
          exclusive
          onChange={(e, newMode) => newMode && onConfigChange('mode', newMode)}
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
            },
          }}
        >
          <ToggleButton value="fast">
            Fast
          </ToggleButton>
          <ToggleButton value="visual">
            Visual
          </ToggleButton>
        </ToggleButtonGroup>

        <p style={{
          fontSize: '12px',
          color: isDark ? 'rgba(255,255,255,0.5)' : 'rgba(0,0,0,0.5)',
          marginTop: '8px',
          marginBottom: 0,
        }}>
          {config.mode === 'fast'
            ? 'Run backtest quickly and show final results'
            : 'Watch trades happen in real-time with playback controls'
          }
        </p>
      </div>

      {/* Run Button */}
      <Button
        variant="contained"
        fullWidth
        size="large"
        onClick={onRunBacktest}
        disabled={isRunning}
        sx={{
          mt: 2,
          py: 1.5,
          fontWeight: 600,
          backgroundColor: '#2962FF',
          '&:hover': {
            backgroundColor: '#1e4bd8',
          },
        }}
      >
        {isRunning ? (
          <>
            <CircularProgress size={20} sx={{ color: '#fff', mr: 1 }} />
            Running...
          </>
        ) : (
          'Run Backtest'
        )}
      </Button>
    </Box>
  );
}

export default BacktestConfigPanel;
