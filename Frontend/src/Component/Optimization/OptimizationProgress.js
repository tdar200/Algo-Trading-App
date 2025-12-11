import React from 'react';
import Box from '@mui/material/Box';
import Paper from '@mui/material/Paper';
import LinearProgress from '@mui/material/LinearProgress';
import Button from '@mui/material/Button';
import Chip from '@mui/material/Chip';

function OptimizationProgress({ progress, hybridProgress, isHybrid, onCancel, theme }) {
  const isDark = theme === 'dark';

  const formatParams = (params) => {
    if (!params) return '-';
    return Object.entries(params)
      .map(([key, value]) => {
        const label = key.replace(/([A-Z])/g, ' $1').trim();
        return `${label}: ${value}`;
      })
      .join(' | ');
  };

  // Standard optimization progress
  if (!isHybrid) {
    return (
      <Paper
        sx={{
          p: 3,
          mb: 2,
          backgroundColor: isDark ? '#1e1e1e' : '#fff',
          borderRadius: '12px',
        }}
      >
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
          <div style={{ fontSize: '18px', fontWeight: 600, color: isDark ? '#fff' : '#333' }}>
            Optimization in Progress
          </div>
          <Button
            variant="outlined"
            color="error"
            size="small"
            onClick={onCancel}
          >
            Cancel
          </Button>
        </Box>

        {/* Progress Bar */}
        <Box sx={{ mb: 3 }}>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
            <span style={{ fontSize: '14px', color: isDark ? 'rgba(255,255,255,0.7)' : 'rgba(0,0,0,0.6)' }}>
              {progress.current} / {progress.total} combinations
            </span>
            <Box sx={{ display: 'flex', gap: 2, alignItems: 'center' }}>
              {progress.rate > 0 && (
                <span style={{ fontSize: '12px', color: isDark ? 'rgba(255,255,255,0.5)' : 'rgba(0,0,0,0.5)' }}>
                  {progress.rate} tests/sec
                </span>
              )}
              {progress.eta_seconds > 0 && (
                <span style={{ fontSize: '12px', color: '#2962FF' }}>
                  ETA: {progress.eta_seconds < 60 ? `${progress.eta_seconds}s` : `${Math.floor(progress.eta_seconds / 60)}m ${progress.eta_seconds % 60}s`}
                </span>
              )}
              <span style={{ fontSize: '14px', fontWeight: 600, color: isDark ? '#fff' : '#333' }}>
                {progress.percent.toFixed(1)}%
              </span>
            </Box>
          </Box>
          <LinearProgress
            variant="determinate"
            value={progress.percent}
            sx={{
              height: 10,
              borderRadius: 5,
              backgroundColor: isDark ? 'rgba(255,255,255,0.1)' : 'rgba(0,0,0,0.1)',
              '& .MuiLinearProgress-bar': {
                backgroundColor: '#2962FF',
                borderRadius: 5,
              },
            }}
          />
        </Box>

        {/* Current Parameters */}
        <Box
          sx={{
            p: 2,
            borderRadius: '8px',
            backgroundColor: isDark ? 'rgba(255,255,255,0.05)' : 'rgba(0,0,0,0.03)',
            mb: 2,
          }}
        >
          <div style={{
            fontSize: '12px',
            textTransform: 'uppercase',
            letterSpacing: '0.5px',
            color: isDark ? 'rgba(255,255,255,0.5)' : 'rgba(0,0,0,0.5)',
            marginBottom: '8px',
          }}>
            Currently Testing
          </div>
          <div style={{
            fontSize: '13px',
            color: isDark ? '#fff' : '#333',
            fontFamily: 'monospace',
          }}>
            {formatParams(progress.currentParams)}
          </div>
        </Box>

        {/* Best So Far */}
        {progress.bestSoFar && (
          <Box
            sx={{
              p: 2,
              borderRadius: '8px',
              backgroundColor: isDark ? 'rgba(76, 175, 80, 0.1)' : 'rgba(76, 175, 80, 0.1)',
              border: '1px solid rgba(76, 175, 80, 0.3)',
            }}
          >
            <div style={{
              fontSize: '12px',
              textTransform: 'uppercase',
              letterSpacing: '0.5px',
              color: '#4caf50',
              marginBottom: '8px',
            }}>
              Best Result So Far
            </div>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <div style={{
                fontSize: '13px',
                color: isDark ? '#fff' : '#333',
                fontFamily: 'monospace',
              }}>
                {formatParams(progress.bestSoFar.params)}
              </div>
              <div style={{
                fontSize: '18px',
                fontWeight: 700,
                color: progress.bestSoFar.net_profit >= 0 ? '#4caf50' : '#f44336',
              }}>
                ${progress.bestSoFar.net_profit?.toLocaleString() || '0'}
              </div>
            </Box>
          </Box>
        )}
      </Paper>
    );
  }

  // Hybrid optimization progress (two phases)
  const screening = hybridProgress?.screening || { current: 0, total: 0, percent: 0, candidates_found: 0 };
  const validation = hybridProgress?.validation || { current: 0, total: 0, percent: 0 };
  const phase = hybridProgress?.phase || 'screening';

  return (
    <Paper
      sx={{
        p: 3,
        mb: 2,
        backgroundColor: isDark ? '#1e1e1e' : '#fff',
        borderRadius: '12px',
      }}
    >
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
          <div style={{ fontSize: '18px', fontWeight: 600, color: isDark ? '#fff' : '#333' }}>
            Hybrid Optimization
          </div>
          <Chip
            label={phase === 'screening' ? 'Phase 1: Screening' : 'Phase 2: Validation'}
            size="small"
            sx={{
              backgroundColor: phase === 'screening'
                ? (isDark ? 'rgba(41, 98, 255, 0.2)' : 'rgba(41, 98, 255, 0.1)')
                : (isDark ? 'rgba(76, 175, 80, 0.2)' : 'rgba(76, 175, 80, 0.1)'),
              color: phase === 'screening' ? '#2962FF' : '#4caf50',
              fontWeight: 600,
            }}
          />
        </Box>
        <Button
          variant="outlined"
          color="error"
          size="small"
          onClick={onCancel}
        >
          Cancel
        </Button>
      </Box>

      {/* Phase 1: VectorBT Screening */}
      <Box sx={{ mb: 3 }}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <svg width="16" height="16" viewBox="0 0 24 24" fill={phase === 'screening' ? '#2962FF' : '#4caf50'}>
              <path d="M7 4h10v15H7z M4 8h3v7H4z M17 8h3v7h-3z M9 2h6v2H9z M9 20h6v2H9z"/>
            </svg>
            <span style={{
              fontSize: '13px',
              fontWeight: 500,
              color: isDark ? '#fff' : '#333',
            }}>
              VectorBT Screening
            </span>
          </Box>
          <Box sx={{ display: 'flex', gap: 2, alignItems: 'center' }}>
            {screening.candidates_found > 0 && (
              <Chip
                label={`${screening.candidates_found} candidates`}
                size="small"
                sx={{
                  backgroundColor: isDark ? 'rgba(76, 175, 80, 0.2)' : 'rgba(76, 175, 80, 0.1)',
                  color: '#4caf50',
                  fontSize: '11px',
                }}
              />
            )}
            <span style={{
              fontSize: '13px',
              color: isDark ? 'rgba(255,255,255,0.7)' : 'rgba(0,0,0,0.6)',
            }}>
              {screening.current} / {screening.total}
            </span>
            <span style={{
              fontSize: '13px',
              fontWeight: 600,
              color: phase === 'screening' ? '#2962FF' : '#4caf50',
            }}>
              {screening.percent.toFixed(1)}%
            </span>
          </Box>
        </Box>
        <LinearProgress
          variant="determinate"
          value={screening.percent}
          sx={{
            height: 8,
            borderRadius: 4,
            backgroundColor: isDark ? 'rgba(255,255,255,0.1)' : 'rgba(0,0,0,0.1)',
            '& .MuiLinearProgress-bar': {
              backgroundColor: phase === 'screening' ? '#2962FF' : '#4caf50',
              borderRadius: 4,
            },
          }}
        />
      </Box>

      {/* Phase 2: Backtrader Validation */}
      <Box sx={{ mb: 2, opacity: phase === 'validation' ? 1 : 0.5 }}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <svg width="16" height="16" viewBox="0 0 24 24" fill={phase === 'validation' ? '#4caf50' : (isDark ? 'rgba(255,255,255,0.5)' : 'rgba(0,0,0,0.3)')}>
              <path d="M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41L9 16.17z"/>
            </svg>
            <span style={{
              fontSize: '13px',
              fontWeight: 500,
              color: isDark ? '#fff' : '#333',
            }}>
              Backtrader Validation
            </span>
          </Box>
          <Box sx={{ display: 'flex', gap: 2, alignItems: 'center' }}>
            <span style={{
              fontSize: '13px',
              color: isDark ? 'rgba(255,255,255,0.7)' : 'rgba(0,0,0,0.6)',
            }}>
              {validation.current} / {validation.total}
            </span>
            <span style={{
              fontSize: '13px',
              fontWeight: 600,
              color: phase === 'validation' ? '#4caf50' : (isDark ? 'rgba(255,255,255,0.5)' : 'rgba(0,0,0,0.3)'),
            }}>
              {validation.percent.toFixed(1)}%
            </span>
          </Box>
        </Box>
        <LinearProgress
          variant="determinate"
          value={validation.percent}
          sx={{
            height: 8,
            borderRadius: 4,
            backgroundColor: isDark ? 'rgba(255,255,255,0.1)' : 'rgba(0,0,0,0.1)',
            '& .MuiLinearProgress-bar': {
              backgroundColor: '#4caf50',
              borderRadius: 4,
            },
          }}
        />
      </Box>

      {/* Info Box */}
      <Box
        sx={{
          p: 2,
          borderRadius: '8px',
          backgroundColor: isDark ? 'rgba(255,255,255,0.05)' : 'rgba(0,0,0,0.03)',
        }}
      >
        <div style={{
          fontSize: '12px',
          color: isDark ? 'rgba(255,255,255,0.6)' : 'rgba(0,0,0,0.5)',
          lineHeight: 1.5,
        }}>
          {phase === 'screening' ? (
            <>
              <strong style={{ color: '#2962FF' }}>Phase 1:</strong> VectorBT is rapidly screening parameter combinations using GPU acceleration.
              Candidates that pass the screening filters will be validated in Phase 2.
            </>
          ) : (
            <>
              <strong style={{ color: '#4caf50' }}>Phase 2:</strong> Backtrader is validating {validation.total} candidates with event-driven simulation.
              Results with {'<'}1% discrepancy from VectorBT will be included in final results.
            </>
          )}
        </div>
      </Box>
    </Paper>
  );
}

export default OptimizationProgress;
