import React, { useEffect, useRef } from 'react';
import Box from '@mui/material/Box';
import { createChart } from 'lightweight-charts';

function EquityChart({ equityCurve, theme }) {
  const chartContainerRef = useRef(null);
  const chartRef = useRef(null);
  const isDark = theme === 'dark';

  useEffect(() => {
    if (!chartContainerRef.current) return;

    // Create chart
    const chart = createChart(chartContainerRef.current, {
      layout: {
        background: { type: 'solid', color: isDark ? '#1e1e1e' : '#ffffff' },
        textColor: isDark ? '#d1d4dc' : '#333',
      },
      grid: {
        vertLines: { color: isDark ? 'rgba(255,255,255,0.1)' : 'rgba(0,0,0,0.1)' },
        horzLines: { color: isDark ? 'rgba(255,255,255,0.1)' : 'rgba(0,0,0,0.1)' },
      },
      width: chartContainerRef.current.clientWidth,
      height: 300,
      rightPriceScale: {
        borderColor: isDark ? 'rgba(255,255,255,0.2)' : 'rgba(0,0,0,0.2)',
      },
      timeScale: {
        borderColor: isDark ? 'rgba(255,255,255,0.2)' : 'rgba(0,0,0,0.2)',
        timeVisible: true,
      },
      crosshair: {
        mode: 1,
      },
    });

    chartRef.current = chart;

    // Balance line (realized P&L)
    const balanceSeries = chart.addLineSeries({
      color: '#2962FF',
      lineWidth: 2,
      title: 'Balance',
    });

    // Equity line (balance + unrealized P&L)
    const equitySeries = chart.addLineSeries({
      color: '#4caf50',
      lineWidth: 2,
      title: 'Equity',
    });

    // Format data for the chart
    if (equityCurve && equityCurve.length > 0) {
      const balanceData = equityCurve.map(point => ({
        time: point.date,
        value: point.balance,
      }));

      const equityData = equityCurve.map(point => ({
        time: point.date,
        value: point.equity,
      }));

      balanceSeries.setData(balanceData);
      equitySeries.setData(equityData);

      chart.timeScale().fitContent();
    }

    // Handle resize
    const handleResize = () => {
      if (chartContainerRef.current && chartRef.current) {
        chartRef.current.applyOptions({
          width: chartContainerRef.current.clientWidth,
        });
      }
    };

    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      if (chartRef.current) {
        chartRef.current.remove();
        chartRef.current = null;
      }
    };
  }, [equityCurve, isDark]);

  if (!equityCurve || equityCurve.length === 0) {
    return (
      <Box
        sx={{
          height: 300,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          color: isDark ? 'rgba(255,255,255,0.5)' : 'rgba(0,0,0,0.5)',
        }}
      >
        No equity data available
      </Box>
    );
  }

  return (
    <Box>
      {/* Legend */}
      <Box sx={{ display: 'flex', gap: 3, mb: 2 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <Box sx={{ width: 16, height: 3, backgroundColor: '#2962FF', borderRadius: 1 }} />
          <span style={{ fontSize: '12px', color: isDark ? 'rgba(255,255,255,0.7)' : 'rgba(0,0,0,0.7)' }}>
            Balance
          </span>
        </Box>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <Box sx={{ width: 16, height: 3, backgroundColor: '#4caf50', borderRadius: 1 }} />
          <span style={{ fontSize: '12px', color: isDark ? 'rgba(255,255,255,0.7)' : 'rgba(0,0,0,0.7)' }}>
            Equity
          </span>
        </Box>
      </Box>

      {/* Chart */}
      <div ref={chartContainerRef} style={{ width: '100%' }} />
    </Box>
  );
}

export default EquityChart;
