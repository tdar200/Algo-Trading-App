import React, { useState, useMemo, useCallback, useTransition } from 'react';
import Box from '@mui/material/Box';
import Paper from '@mui/material/Paper';
import Table from '@mui/material/Table';
import TableBody from '@mui/material/TableBody';
import TableCell from '@mui/material/TableCell';
import TableContainer from '@mui/material/TableContainer';
import TableHead from '@mui/material/TableHead';
import TableRow from '@mui/material/TableRow';
import Chip from '@mui/material/Chip';
import Collapse from '@mui/material/Collapse';
import CircularProgress from '@mui/material/CircularProgress';
import Fade from '@mui/material/Fade';

const COLUMNS = [
  { key: 'rank', label: '#', sortable: false, width: 50 },
  { key: 'net_profit', label: 'Net Profit', sortable: true },
  { key: 'net_profit_percent', label: 'Return %', sortable: true },
  { key: 'win_rate', label: 'Win Rate', sortable: true },
  { key: 'profit_factor', label: 'Profit Factor', sortable: true },
  { key: 'max_drawdown_percent', label: 'Max DD %', sortable: true },
  { key: 'total_trades', label: 'Trades', sortable: true },
  { key: 'sharpe_ratio', label: 'Sharpe', sortable: true },
];

const PARAM_SHORT_LABELS = {
  firstRetracement: '1st Ret',
  secondRetracement: '2nd Ret',
  touchCount: 'Touch',
  breakoutBuffer: 'Buffer',
  takeProfit: 'TP',
  stopLoss: 'SL',
};

// Memoized format function - defined outside component
const formatValue = (key, value) => {
  if (value === null || value === undefined) return '-';

  switch (key) {
    case 'net_profit':
      return `$${value.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
    case 'net_profit_percent':
    case 'win_rate':
    case 'max_drawdown_percent':
      return `${value.toFixed(2)}%`;
    case 'profit_factor':
    case 'sharpe_ratio':
      return value === 'Infinity' ? '∞' : value.toFixed(2);
    case 'total_trades':
      return value.toString();
    default:
      return value;
  }
};

// Memoized color function - defined outside component
const getValueColor = (key, value, isDark) => {
  if (key === 'net_profit' || key === 'net_profit_percent') {
    return value >= 0 ? '#4caf50' : '#f44336';
  }
  if (key === 'win_rate') {
    return value >= 50 ? '#4caf50' : value >= 40 ? '#ff9800' : '#f44336';
  }
  if (key === 'max_drawdown_percent') {
    return value <= 10 ? '#4caf50' : value <= 20 ? '#ff9800' : '#f44336';
  }
  return isDark ? '#fff' : '#333';
};

// Memoized Result Row Component
const ResultRow = React.memo(({ result, index, isDark, isExpanded, onRowClick, onResultClick }) => {
  return (
    <React.Fragment>
      <TableRow
        hover
        onClick={() => onResultClick ? onResultClick(result) : onRowClick(index)}
        onDoubleClick={() => onResultClick && onResultClick(result)}
        sx={{
          cursor: 'pointer',
          backgroundColor: index === 0
            ? (isDark ? 'rgba(76, 175, 80, 0.1)' : 'rgba(76, 175, 80, 0.05)')
            : 'inherit',
          '&:hover': {
            backgroundColor: isDark ? 'rgba(255,255,255,0.05)' : 'rgba(0,0,0,0.03)',
          },
        }}
      >
        <TableCell sx={{ color: isDark ? '#fff' : '#333', fontWeight: index === 0 ? 700 : 400 }}>
          {index === 0 ? '🏆' : index + 1}
        </TableCell>
        {COLUMNS.slice(1).map((column) => (
          <TableCell
            key={column.key}
            sx={{
              color: getValueColor(column.key, result.statistics[column.key], isDark),
              fontWeight: column.key === 'net_profit' ? 600 : 400,
              fontSize: '13px',
            }}
          >
            {formatValue(column.key, result.statistics[column.key])}
          </TableCell>
        ))}
      </TableRow>
      <TableRow>
        <TableCell colSpan={COLUMNS.length} sx={{ py: 0, border: 0 }}>
          <Collapse in={isExpanded} timeout="auto" unmountOnExit>
            <Box sx={{ py: 2, px: 1 }}>
              <div style={{
                fontSize: '12px',
                textTransform: 'uppercase',
                letterSpacing: '0.5px',
                color: isDark ? 'rgba(255,255,255,0.5)' : 'rgba(0,0,0,0.5)',
                marginBottom: '8px',
              }}>
                Parameters
              </div>
              <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
                {Object.entries(result.params).map(([key, value]) => (
                  <Chip
                    key={key}
                    label={`${PARAM_SHORT_LABELS[key] || key}: ${value}`}
                    size="small"
                    sx={{
                      backgroundColor: isDark ? 'rgba(255,255,255,0.1)' : 'rgba(0,0,0,0.08)',
                      color: isDark ? '#fff' : '#333',
                      fontSize: '12px',
                    }}
                  />
                ))}
              </Box>
            </Box>
          </Collapse>
        </TableCell>
      </TableRow>
    </React.Fragment>
  );
});

ResultRow.displayName = 'ResultRow';

// Memoized Header Cell Component
const HeaderCell = React.memo(({ column, sortConfig, onSort, isDark }) => {
  const isActive = sortConfig.key === column.key;

  return (
    <TableCell
      sx={{
        backgroundColor: isDark ? '#252525' : '#f5f5f5',
        color: isDark ? '#fff' : '#333',
        fontWeight: 600,
        fontSize: '12px',
        cursor: column.sortable ? 'pointer' : 'default',
        width: column.width,
        whiteSpace: 'nowrap',
        transition: 'background-color 0.15s ease',
        '&:hover': column.sortable ? {
          backgroundColor: isDark ? '#333' : '#eee',
        } : {},
      }}
      onClick={() => column.sortable && onSort(column.key)}
    >
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
        {column.label}
        {column.sortable && isActive && (
          <span style={{ fontSize: '10px' }}>
            {sortConfig.direction === 'desc' ? '▼' : '▲'}
          </span>
        )}
      </Box>
    </TableCell>
  );
});

HeaderCell.displayName = 'HeaderCell';

function OptimizationResults({ results, sortConfig, onSort, onResultClick, theme }) {
  const isDark = theme === 'dark';
  const [expandedRow, setExpandedRow] = useState(null);
  const [isPending, startTransition] = useTransition();

  // Memoize the row click handler
  const handleRowClick = useCallback((index) => {
    setExpandedRow(prev => prev === index ? null : index);
  }, []);

  // Wrap sorting in transition for smoother UX
  const handleSort = useCallback((key) => {
    startTransition(() => {
      onSort(key);
    });
  }, [onSort]);

  // Memoize visible results (limit to first 100 for performance)
  const visibleResults = useMemo(() => {
    return results.slice(0, 100);
  }, [results]);

  return (
    <Paper
      sx={{
        backgroundColor: isDark ? '#1e1e1e' : '#fff',
        borderRadius: '12px',
        overflow: 'hidden',
        position: 'relative',
      }}
    >
      {/* Sorting Overlay */}
      <Fade in={isPending}>
        <Box
          sx={{
            position: 'absolute',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            backgroundColor: isDark ? 'rgba(0,0,0,0.5)' : 'rgba(255,255,255,0.7)',
            zIndex: 10,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
          }}
        >
          <Box sx={{ textAlign: 'center' }}>
            <CircularProgress size={32} sx={{ color: '#2962FF' }} />
            <div style={{
              marginTop: '8px',
              fontSize: '13px',
              color: isDark ? 'rgba(255,255,255,0.7)' : 'rgba(0,0,0,0.6)'
            }}>
              Sorting...
            </div>
          </Box>
        </Box>
      </Fade>

      <Box sx={{ p: 2, borderBottom: `1px solid ${isDark ? 'rgba(255,255,255,0.1)' : 'rgba(0,0,0,0.1)'}` }}>
        <div style={{ fontSize: '18px', fontWeight: 600, color: isDark ? '#fff' : '#333' }}>
          Optimization Results
        </div>
        <div style={{ fontSize: '13px', color: isDark ? 'rgba(255,255,255,0.5)' : 'rgba(0,0,0,0.5)', marginTop: '4px' }}>
          {results.length} combinations tested{results.length > 100 ? ` (showing top 100)` : ''} - Click a row to view detailed statistics
        </div>
      </Box>

      <TableContainer sx={{ maxHeight: 600 }}>
        <Table stickyHeader size="small">
          <TableHead>
            <TableRow>
              {COLUMNS.map((column) => (
                <HeaderCell
                  key={column.key}
                  column={column}
                  sortConfig={sortConfig}
                  onSort={handleSort}
                  isDark={isDark}
                />
              ))}
            </TableRow>
          </TableHead>
          <TableBody>
            {visibleResults.map((result, index) => (
              <ResultRow
                key={`${result.params?.takeProfit}-${result.params?.stopLoss}-${index}`}
                result={result}
                index={index}
                isDark={isDark}
                isExpanded={expandedRow === index}
                onRowClick={handleRowClick}
                onResultClick={onResultClick}
              />
            ))}
          </TableBody>
        </Table>
      </TableContainer>

      {results.length === 0 && (
        <Box sx={{ p: 4, textAlign: 'center', color: isDark ? 'rgba(255,255,255,0.5)' : 'rgba(0,0,0,0.5)' }}>
          No results yet
        </Box>
      )}
    </Paper>
  );
}

export default React.memo(OptimizationResults);
