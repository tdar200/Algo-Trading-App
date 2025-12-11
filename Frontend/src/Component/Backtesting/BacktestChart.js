import React, { useEffect, useRef } from 'react';
import { createChart, ColorType, CrosshairMode } from 'lightweight-charts';

function BacktestChart({ data, trades, theme }) {
  const chartContainerRef = useRef(null);
  const chartRef = useRef(null);
  const isDark = theme === 'dark';

  useEffect(() => {
    if (!chartContainerRef.current || !data || data.length === 0) return;

    // Theme-based colors
    const colors = isDark ? {
      background: '#1e1e1e',
      textColor: '#d1d4dc',
      gridColor: 'rgba(255,255,255,0.1)',
      borderColor: 'rgba(255,255,255,0.2)',
      crosshairColor: 'rgba(255,255,255,0.3)',
    } : {
      background: '#ffffff',
      textColor: '#1e293b',
      gridColor: 'rgba(0, 0, 0, 0.06)',
      borderColor: 'rgba(0, 0, 0, 0.1)',
      crosshairColor: 'rgba(0, 0, 0, 0.3)',
    };

    const chart = createChart(chartContainerRef.current, {
      layout: {
        background: { type: ColorType.Solid, color: colors.background },
        textColor: colors.textColor,
        fontSize: 12,
        fontFamily: "'Inter', -apple-system, BlinkMacSystemFont, sans-serif",
      },
      width: chartContainerRef.current.clientWidth,
      height: 350,
      grid: {
        vertLines: { color: colors.gridColor },
        horzLines: { color: colors.gridColor },
      },
      crosshair: {
        mode: CrosshairMode.Normal,
        vertLine: { color: colors.crosshairColor, labelBackgroundColor: '#2962FF' },
        horzLine: { color: colors.crosshairColor, labelBackgroundColor: '#2962FF' },
      },
      timeScale: {
        borderColor: colors.borderColor,
        timeVisible: true,
      },
      rightPriceScale: {
        borderColor: colors.borderColor,
        scaleMargins: { top: 0.1, bottom: 0.2 },
      },
    });

    chartRef.current = chart;

    // Candlestick series
    const candlestickSeries = chart.addCandlestickSeries({
      upColor: '#26a69a',
      downColor: '#ef5350',
      borderDownColor: '#ef5350',
      borderUpColor: '#26a69a',
      wickDownColor: '#ef5350',
      wickUpColor: '#26a69a',
    });

    // Format data
    const formattedData = data.map(item => ({
      time: item.date || item.Date,
      open: item.open || item.Open,
      high: item.high || item.High,
      low: item.low || item.Low,
      close: item.close || item.Close,
    })).sort((a, b) => new Date(a.time) - new Date(b.time));

    candlestickSeries.setData(formattedData);

    // Add trade markers
    if (trades && trades.length > 0) {
      const markers = [];

      trades.forEach(trade => {
        // Entry marker
        if (trade.entry_date) {
          markers.push({
            time: trade.entry_date,
            position: trade.type === 'long' ? 'belowBar' : 'aboveBar',
            color: '#2962FF',
            shape: trade.type === 'long' ? 'arrowUp' : 'arrowDown',
            text: `${trade.type === 'long' ? 'BUY' : 'SELL'} @ ${trade.entry_price?.toFixed(2)}`,
          });
        }

        // Exit marker
        if (trade.exit_date) {
          const isProfitable = trade.pnl >= 0;
          markers.push({
            time: trade.exit_date,
            position: trade.type === 'long' ? 'aboveBar' : 'belowBar',
            color: isProfitable ? '#4caf50' : '#f44336',
            shape: 'circle',
            text: `EXIT @ ${trade.exit_price?.toFixed(2)} (${isProfitable ? '+' : ''}${trade.pnl?.toFixed(2)})`,
          });
        }
      });

      // Sort markers by time
      markers.sort((a, b) => new Date(a.time) - new Date(b.time));
      candlestickSeries.setMarkers(markers);
    }

    // Volume series
    const volumeSeries = chart.addHistogramSeries({
      color: '#26a69a',
      priceFormat: { type: 'volume' },
      priceScaleId: '',
      scaleMargins: { top: 0.8, bottom: 0 },
    });

    const volumeData = data.map(item => ({
      time: item.date || item.Date,
      value: item.volume || item.Volume || 0,
      color: (item.close || item.Close) >= (item.open || item.Open)
        ? 'rgba(38, 166, 154, 0.5)'
        : 'rgba(239, 83, 80, 0.5)',
    })).sort((a, b) => new Date(a.time) - new Date(b.time));

    volumeSeries.setData(volumeData);

    chart.timeScale().fitContent();

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
  }, [data, trades, isDark]);

  if (!data || data.length === 0) {
    return (
      <div style={{
        height: 350,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        color: isDark ? 'rgba(255,255,255,0.5)' : 'rgba(0,0,0,0.5)',
      }}>
        No chart data available
      </div>
    );
  }

  return <div ref={chartContainerRef} style={{ width: '100%' }} />;
}

export default BacktestChart;
