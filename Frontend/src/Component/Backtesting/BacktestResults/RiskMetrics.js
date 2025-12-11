import React from 'react';
import Box from '@mui/material/Box';
import Grid from '@mui/material/Grid';
import Paper from '@mui/material/Paper';
import Tooltip from '@mui/material/Tooltip';

function MetricCard({ label, value, tooltip, color, isDark, prefix = '', suffix = '', size = 'normal' }) {
  const card = (
    <Paper
      elevation={0}
      sx={{
        p: size === 'small' ? 1.5 : 2,
        backgroundColor: isDark ? 'rgba(255,255,255,0.05)' : 'rgba(0,0,0,0.02)',
        borderRadius: 2,
        textAlign: 'center',
        height: '100%',
      }}
    >
      <div style={{
        fontSize: size === 'small' ? '10px' : '11px',
        color: isDark ? 'rgba(255,255,255,0.6)' : 'rgba(0,0,0,0.6)',
        marginBottom: '4px',
        textTransform: 'uppercase',
        letterSpacing: '0.5px',
        lineHeight: 1.2,
      }}>
        {label}
      </div>
      <div style={{
        fontSize: size === 'small' ? '16px' : '18px',
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

  if (tooltip) {
    return (
      <Tooltip title={tooltip} placement="top" arrow>
        {card}
      </Tooltip>
    );
  }
  return card;
}

function SectionTitle({ children, isDark }) {
  return (
    <div style={{
      fontSize: '12px',
      fontWeight: 600,
      color: isDark ? 'rgba(255,255,255,0.8)' : 'rgba(0,0,0,0.7)',
      marginBottom: '12px',
      marginTop: '20px',
      textTransform: 'uppercase',
      letterSpacing: '1px',
      borderBottom: `1px solid ${isDark ? 'rgba(255,255,255,0.1)' : 'rgba(0,0,0,0.1)'}`,
      paddingBottom: '8px',
    }}>
      {children}
    </div>
  );
}

function RiskMetrics({ riskMetrics, theme }) {
  const isDark = theme === 'dark';

  if (!riskMetrics) {
    return (
      <Box sx={{ textAlign: 'center', py: 4, color: isDark ? 'rgba(255,255,255,0.5)' : 'rgba(0,0,0,0.5)' }}>
        No risk metrics available
      </Box>
    );
  }

  const getColor = (value, thresholds) => {
    if (thresholds.good !== undefined && value >= thresholds.good) return '#4caf50';
    if (thresholds.bad !== undefined && value <= thresholds.bad) return '#f44336';
    if (thresholds.warn !== undefined && value <= thresholds.warn) return '#ff9800';
    return isDark ? '#fff' : '#000';
  };

  return (
    <Box>
      {/* Risk-Adjusted Returns */}
      <SectionTitle isDark={isDark}>Risk-Adjusted Returns</SectionTitle>
      <Grid container spacing={2}>
        <Grid item xs={6} sm={3}>
          <MetricCard
            label="Sharpe Ratio"
            value={riskMetrics.sharpe_ratio}
            tooltip="Risk-adjusted return (excess return per unit of volatility). >1 is good, >2 is excellent"
            color={getColor(riskMetrics.sharpe_ratio, { good: 1, warn: 0.5, bad: 0 })}
            isDark={isDark}
          />
        </Grid>
        <Grid item xs={6} sm={3}>
          <MetricCard
            label="Sortino Ratio"
            value={riskMetrics.sortino_ratio}
            tooltip="Like Sharpe but only considers downside volatility. >2 is good, >3 is excellent"
            color={getColor(riskMetrics.sortino_ratio, { good: 2, warn: 1, bad: 0 })}
            isDark={isDark}
          />
        </Grid>
        <Grid item xs={6} sm={3}>
          <MetricCard
            label="Calmar Ratio"
            value={riskMetrics.calmar_ratio}
            tooltip="Annualized return divided by max drawdown. >1 is good, >3 is excellent"
            color={getColor(riskMetrics.calmar_ratio, { good: 1, warn: 0.5, bad: 0 })}
            isDark={isDark}
          />
        </Grid>
        <Grid item xs={6} sm={3}>
          <MetricCard
            label="Recovery Factor"
            value={riskMetrics.recovery_factor}
            tooltip="Total return divided by max drawdown. Higher is better"
            color={getColor(riskMetrics.recovery_factor, { good: 2, warn: 1, bad: 0 })}
            isDark={isDark}
          />
        </Grid>
      </Grid>

      {/* Volatility & Risk */}
      <SectionTitle isDark={isDark}>Volatility & Drawdown</SectionTitle>
      <Grid container spacing={2}>
        <Grid item xs={6} sm={3}>
          <MetricCard
            label="Daily Volatility"
            value={riskMetrics.daily_volatility}
            tooltip="Standard deviation of daily returns"
            isDark={isDark}
            suffix="%"
          />
        </Grid>
        <Grid item xs={6} sm={3}>
          <MetricCard
            label="Annual Volatility"
            value={riskMetrics.annualized_volatility}
            tooltip="Annualized standard deviation of returns"
            color={riskMetrics.annualized_volatility > 30 ? '#f44336' : undefined}
            isDark={isDark}
            suffix="%"
          />
        </Grid>
        <Grid item xs={6} sm={3}>
          <MetricCard
            label="Max Drawdown"
            value={riskMetrics.max_drawdown_percent}
            tooltip="Largest peak-to-trough decline"
            color={riskMetrics.max_drawdown_percent > 20 ? '#f44336' : riskMetrics.max_drawdown_percent > 10 ? '#ff9800' : '#4caf50'}
            isDark={isDark}
            suffix="%"
          />
        </Grid>
        <Grid item xs={6} sm={3}>
          <MetricCard
            label="Trading Days"
            value={riskMetrics.trading_days}
            tooltip="Number of trading days in the backtest"
            isDark={isDark}
          />
        </Grid>
      </Grid>

      {/* Value at Risk */}
      <SectionTitle isDark={isDark}>Value at Risk (VaR)</SectionTitle>
      <Grid container spacing={2}>
        <Grid item xs={6} sm={4}>
          <MetricCard
            label="VaR (95%)"
            value={riskMetrics.var_95}
            tooltip="Maximum expected daily loss with 95% confidence"
            color="#f44336"
            isDark={isDark}
            prefix="$"
          />
        </Grid>
        <Grid item xs={6} sm={4}>
          <MetricCard
            label="VaR (99%)"
            value={riskMetrics.var_99}
            tooltip="Maximum expected daily loss with 99% confidence"
            color="#f44336"
            isDark={isDark}
            prefix="$"
          />
        </Grid>
        <Grid item xs={12} sm={4}>
          <MetricCard
            label="CVaR (95%)"
            value={riskMetrics.cvar_95}
            tooltip="Expected loss when loss exceeds VaR (Expected Shortfall)"
            color="#f44336"
            isDark={isDark}
            prefix="$"
          />
        </Grid>
      </Grid>

      {/* Trade Statistics */}
      <SectionTitle isDark={isDark}>Trade Analysis</SectionTitle>
      <Grid container spacing={2}>
        <Grid item xs={4} sm={2}>
          <MetricCard
            label="Risk/Reward"
            value={riskMetrics.risk_reward_ratio}
            tooltip="Average win / Average loss ratio"
            color={getColor(riskMetrics.risk_reward_ratio, { good: 2, warn: 1, bad: 0.5 })}
            isDark={isDark}
            size="small"
          />
        </Grid>
        <Grid item xs={4} sm={2}>
          <MetricCard
            label="Expected Value"
            value={riskMetrics.expected_value}
            tooltip="Average profit/loss per trade"
            color={riskMetrics.expected_value >= 0 ? '#4caf50' : '#f44336'}
            isDark={isDark}
            prefix="$"
            size="small"
          />
        </Grid>
        <Grid item xs={4} sm={2}>
          <MetricCard
            label="Profit Factor"
            value={riskMetrics.profit_factor}
            tooltip="Gross profit / Gross loss. >1.5 is good, >2 is excellent"
            color={getColor(typeof riskMetrics.profit_factor === 'number' ? riskMetrics.profit_factor : 0, { good: 1.5, warn: 1, bad: 0.8 })}
            isDark={isDark}
            size="small"
          />
        </Grid>
        <Grid item xs={6} sm={3}>
          <MetricCard
            label="Max Consec. Wins"
            value={riskMetrics.max_consecutive_wins}
            tooltip="Longest winning streak"
            color="#4caf50"
            isDark={isDark}
            size="small"
          />
        </Grid>
        <Grid item xs={6} sm={3}>
          <MetricCard
            label="Max Consec. Losses"
            value={riskMetrics.max_consecutive_losses}
            tooltip="Longest losing streak"
            color="#f44336"
            isDark={isDark}
            size="small"
          />
        </Grid>
      </Grid>

      {/* Position Sizing Recommendations */}
      <SectionTitle isDark={isDark}>Position Sizing Recommendations</SectionTitle>
      <Grid container spacing={2}>
        <Grid item xs={4}>
          <MetricCard
            label="Kelly Criterion"
            value={riskMetrics.kelly_criterion}
            tooltip="Optimal position size based on win rate and risk/reward. Often considered aggressive."
            color="#2962FF"
            isDark={isDark}
            suffix="%"
          />
        </Grid>
        <Grid item xs={4}>
          <MetricCard
            label="Half Kelly"
            value={riskMetrics.half_kelly}
            tooltip="Conservative position size (half of Kelly). Recommended for most traders."
            color="#4caf50"
            isDark={isDark}
            suffix="%"
          />
        </Grid>
        <Grid item xs={4}>
          <MetricCard
            label="VaR-Based"
            value={riskMetrics.var_based_position}
            tooltip="Position size targeting 2% daily VaR"
            isDark={isDark}
            suffix="%"
          />
        </Grid>
      </Grid>

      {/* Recommendation Box */}
      <Paper
        elevation={0}
        sx={{
          mt: 3,
          p: 2,
          backgroundColor: isDark ? 'rgba(41, 98, 255, 0.1)' : 'rgba(41, 98, 255, 0.05)',
          border: `1px solid ${isDark ? 'rgba(41, 98, 255, 0.3)' : 'rgba(41, 98, 255, 0.2)'}`,
          borderRadius: 2,
        }}
      >
        <div style={{
          fontSize: '12px',
          fontWeight: 600,
          color: '#2962FF',
          marginBottom: '8px',
          textTransform: 'uppercase',
          letterSpacing: '0.5px',
        }}>
          Recommendation
        </div>
        <div style={{
          fontSize: '13px',
          color: isDark ? 'rgba(255,255,255,0.8)' : 'rgba(0,0,0,0.7)',
          lineHeight: 1.5,
        }}>
          {getRecommendation(riskMetrics)}
        </div>
      </Paper>
    </Box>
  );
}

function getRecommendation(metrics) {
  const recommendations = [];

  // Sharpe ratio assessment
  if (metrics.sharpe_ratio >= 2) {
    recommendations.push("Excellent risk-adjusted returns (Sharpe > 2).");
  } else if (metrics.sharpe_ratio >= 1) {
    recommendations.push("Good risk-adjusted returns (Sharpe > 1).");
  } else if (metrics.sharpe_ratio > 0) {
    recommendations.push("Positive but modest risk-adjusted returns.");
  } else {
    recommendations.push("Poor risk-adjusted returns. Consider adjusting strategy parameters.");
  }

  // Drawdown assessment
  if (metrics.max_drawdown_percent > 25) {
    recommendations.push("High drawdown risk - consider tighter stop losses or smaller positions.");
  } else if (metrics.max_drawdown_percent > 15) {
    recommendations.push("Moderate drawdown - ensure position sizing accounts for this.");
  }

  // Position sizing
  if (metrics.half_kelly > 0) {
    recommendations.push(`Suggested position size: ${metrics.half_kelly.toFixed(1)}% of capital per trade (Half-Kelly).`);
  }

  // Risk/reward
  if (metrics.risk_reward_ratio < 1) {
    recommendations.push("Risk/reward ratio below 1:1 - wins need to be larger than losses for profitability.");
  }

  return recommendations.join(" ");
}

export default RiskMetrics;
