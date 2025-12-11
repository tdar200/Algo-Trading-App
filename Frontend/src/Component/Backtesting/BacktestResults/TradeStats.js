import React from 'react';
import Box from '@mui/material/Box';
import Grid from '@mui/material/Grid';
import Paper from '@mui/material/Paper';

function StatCard({ label, value, color, isDark, prefix = '', suffix = '' }) {
  return (
    <Paper
      elevation={0}
      sx={{
        p: 2,
        backgroundColor: isDark ? 'rgba(255,255,255,0.05)' : 'rgba(0,0,0,0.02)',
        borderRadius: 2,
        textAlign: 'center',
      }}
    >
      <div style={{
        fontSize: '12px',
        color: isDark ? 'rgba(255,255,255,0.6)' : 'rgba(0,0,0,0.6)',
        marginBottom: '4px',
        textTransform: 'uppercase',
        letterSpacing: '0.5px',
      }}>
        {label}
      </div>
      <div style={{
        fontSize: '20px',
        fontWeight: 600,
        color: color || (isDark ? '#fff' : '#000'),
      }}>
        {prefix}{typeof value === 'number' ? value.toLocaleString(undefined, {
          minimumFractionDigits: 2,
          maximumFractionDigits: 2
        }) : value}{suffix}
      </div>
    </Paper>
  );
}

function TradeStats({ statistics, theme }) {
  const isDark = theme === 'dark';

  if (!statistics) {
    return (
      <Box sx={{ textAlign: 'center', py: 4, color: isDark ? 'rgba(255,255,255,0.5)' : 'rgba(0,0,0,0.5)' }}>
        No statistics available
      </Box>
    );
  }

  const profitColor = statistics.net_profit >= 0 ? '#4caf50' : '#f44336';

  return (
    <Box>
      {/* Key Metrics Row */}
      <Grid container spacing={2} sx={{ mb: 3 }}>
        <Grid item xs={6} sm={3}>
          <StatCard
            label="Net Profit"
            value={statistics.net_profit || 0}
            color={profitColor}
            isDark={isDark}
            prefix="$"
          />
        </Grid>
        <Grid item xs={6} sm={3}>
          <StatCard
            label="Return"
            value={statistics.net_profit_percent || 0}
            color={profitColor}
            isDark={isDark}
            suffix="%"
          />
        </Grid>
        <Grid item xs={6} sm={3}>
          <StatCard
            label="Profit Factor"
            value={statistics.profit_factor === Infinity ? '∞' : (statistics.profit_factor || 0)}
            isDark={isDark}
          />
        </Grid>
        <Grid item xs={6} sm={3}>
          <StatCard
            label="Max Drawdown"
            value={statistics.max_drawdown_percent || 0}
            color="#f44336"
            isDark={isDark}
            suffix="%"
          />
        </Grid>
      </Grid>

      {/* Trade Summary */}
      <Grid container spacing={2} sx={{ mb: 3 }}>
        <Grid item xs={4} sm={2}>
          <StatCard
            label="Total Trades"
            value={statistics.total_trades || 0}
            isDark={isDark}
          />
        </Grid>
        <Grid item xs={4} sm={2}>
          <StatCard
            label="Winning"
            value={statistics.winning_trades || 0}
            color="#4caf50"
            isDark={isDark}
          />
        </Grid>
        <Grid item xs={4} sm={2}>
          <StatCard
            label="Losing"
            value={statistics.losing_trades || 0}
            color="#f44336"
            isDark={isDark}
          />
        </Grid>
        <Grid item xs={4} sm={2}>
          <StatCard
            label="Win Rate"
            value={statistics.win_rate || 0}
            isDark={isDark}
            suffix="%"
          />
        </Grid>
        <Grid item xs={4} sm={2}>
          <StatCard
            label="Avg Win"
            value={statistics.average_win || 0}
            color="#4caf50"
            isDark={isDark}
            prefix="$"
          />
        </Grid>
        <Grid item xs={4} sm={2}>
          <StatCard
            label="Avg Loss"
            value={Math.abs(statistics.average_loss || 0)}
            color="#f44336"
            isDark={isDark}
            prefix="-$"
          />
        </Grid>
      </Grid>

      {/* Additional Stats */}
      <Grid container spacing={2}>
        <Grid item xs={6} sm={3}>
          <StatCard
            label="Gross Profit"
            value={statistics.gross_profit || 0}
            color="#4caf50"
            isDark={isDark}
            prefix="$"
          />
        </Grid>
        <Grid item xs={6} sm={3}>
          <StatCard
            label="Gross Loss"
            value={Math.abs(statistics.gross_loss || 0)}
            color="#f44336"
            isDark={isDark}
            prefix="-$"
          />
        </Grid>
        <Grid item xs={6} sm={3}>
          <StatCard
            label="Largest Win"
            value={statistics.largest_win || 0}
            color="#4caf50"
            isDark={isDark}
            prefix="$"
          />
        </Grid>
        <Grid item xs={6} sm={3}>
          <StatCard
            label="Largest Loss"
            value={Math.abs(statistics.largest_loss || 0)}
            color="#f44336"
            isDark={isDark}
            prefix="-$"
          />
        </Grid>
      </Grid>
    </Box>
  );
}

export default TradeStats;
