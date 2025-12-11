import { useEffect, useRef, useCallback } from "react";
import { createChart, ColorType, CrosshairMode } from "lightweight-charts";

const CandlestickChart = ({
  data,
  trades = [],
  levels = [],
  resistanceLevels = [],
  supportLevels = [],
  resistanceSwings = [],
  supportSwings = [],
  absoluteResistance = null,
  absoluteSupport = null,
  theme = 'light'
}) => {
  const chartContainerRef = useRef();
  const chartRef = useRef(null);
  const candlestickSeriesRef = useRef(null);
  const levelMapRef = useRef(new Map()); // Maps price -> first occurrence date
  const isDark = theme === 'dark';

  useEffect(() => {
    if (!data || data.length === 0) return;

    // Theme-based colors
    const colors = isDark ? {
      background: '#131722',
      textColor: '#d1d4dc',
      gridColor: 'rgba(42, 46, 57, 0.6)',
      borderColor: 'rgba(42, 46, 57, 0.8)',
      crosshairColor: 'rgba(224, 227, 235, 0.4)',
    } : {
      background: '#ffffff',
      textColor: '#1e293b',
      gridColor: 'rgba(0, 0, 0, 0.06)',
      borderColor: 'rgba(0, 0, 0, 0.1)',
      crosshairColor: 'rgba(0, 0, 0, 0.3)',
    };

    const chartOptions = {
      layout: {
        background: { type: ColorType.Solid, color: colors.background },
        textColor: colors.textColor,
        fontSize: 12,
        fontFamily: "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
      },
      width: chartContainerRef.current.clientWidth,
      height: 600,
      grid: {
        vertLines: {
          color: colors.gridColor,
          style: 1,
        },
        horzLines: {
          color: colors.gridColor,
          style: 1,
        },
      },
      crosshair: {
        mode: CrosshairMode.Normal,
        vertLine: {
          width: 1,
          color: colors.crosshairColor,
          style: 0,
          labelBackgroundColor: '#2962FF',
        },
        horzLine: {
          width: 1,
          color: colors.crosshairColor,
          style: 0,
          labelBackgroundColor: '#2962FF',
        },
      },
      timeScale: {
        borderColor: colors.borderColor,
        timeVisible: true,
        secondsVisible: false,
        tickMarkFormatter: (time) => {
          const date = new Date(time);
          const month = date.toLocaleString('default', { month: 'short' });
          const day = date.getDate();
          return `${month} ${day}`;
        },
      },
      rightPriceScale: {
        borderColor: colors.borderColor,
        scaleMargins: {
          top: 0.1,
          bottom: 0.2,
        },
        autoScale: true,
      },
      handleScroll: {
        vertTouchDrag: false,
      },
      handleScale: {
        axisPressedMouseMove: true,
      },
    };

    const chart = createChart(chartContainerRef.current, chartOptions);
    chartRef.current = chart;

    // Build level map: for each level price, find the earliest swing date near that price
    const levelMap = new Map();

    // Helper to find earliest swing date for a given price (with tolerance)
    const findEarliestSwingDate = (targetPrice, swings) => {
      let earliestDate = null;
      const tolerance = targetPrice * 0.02; // 2% tolerance

      swings.forEach((swing) => {
        if (swing.middle?.date && swing.middle?.price) {
          if (Math.abs(swing.middle.price - targetPrice) <= tolerance) {
            if (!earliestDate || swing.middle.date < earliestDate) {
              earliestDate = swing.middle.date;
            }
          }
        }
      });
      return earliestDate;
    };

    // Map resistance levels to their first occurrence
    resistanceLevels.forEach((levelData) => {
      const price = typeof levelData === 'object' ? levelData.price : levelData;
      const date = findEarliestSwingDate(price, resistanceSwings);
      if (date) {
        levelMap.set(price, date);
      }
    });

    // Map support levels to their first occurrence
    supportLevels.forEach((levelData) => {
      const price = typeof levelData === 'object' ? levelData.price : levelData;
      const date = findEarliestSwingDate(price, supportSwings);
      if (date) {
        levelMap.set(price, date);
      }
    });

    // Add absolute levels
    if (absoluteResistance?.price && absoluteResistance?.date) {
      levelMap.set(absoluteResistance.price, absoluteResistance.date);
    }
    if (absoluteSupport?.price && absoluteSupport?.date) {
      levelMap.set(absoluteSupport.price, absoluteSupport.date);
    }

    levelMapRef.current = levelMap;

    // Add watermark
    chart.applyOptions({
      watermark: {
        visible: true,
        fontSize: 48,
        horzAlign: 'center',
        vertAlign: 'center',
        color: 'rgba(42, 46, 57, 0.5)',
        text: data.length > 0 ? '' : 'Loading...',
      },
    });

    // Candlestick Series with professional colors
    const candlestickSeries = chart.addCandlestickSeries({
      upColor: '#089981',
      downColor: '#F23645',
      borderUpColor: '#089981',
      borderDownColor: '#F23645',
      wickUpColor: '#089981',
      wickDownColor: '#F23645',
      borderVisible: true,
    });
    candlestickSeriesRef.current = candlestickSeries;

    // Price line styling
    candlestickSeries.applyOptions({
      priceLineVisible: true,
      priceLineWidth: 1,
      priceLineColor: '#2962FF',
      priceLineStyle: 2,
      lastValueVisible: true,
    });

    const candleData = data.map(d => ({
      time: d.date,
      open: d.Open,
      high: d.High,
      low: d.Low,
      close: d.Close,
    }));

    candleData.sort((a, b) => (a.time > b.time ? 1 : -1));
    candlestickSeries.setData(candleData);

    // Get first and last dates for horizontal lines
    const firstDate = candleData[0]?.time;
    const lastDate = candleData[candleData.length - 1]?.time;

    // Helper function to get color based on strength
    const getStrengthColor = (strength, isResistance) => {
      const alpha = Math.max(0.4, Math.min(1, strength / 100));
      if (isResistance) {
        return `rgba(242, 54, 69, ${alpha})`;
      } else {
        return `rgba(8, 153, 129, ${alpha})`;
      }
    };

    // Helper function to get line width based on strength
    const getStrengthWidth = (strength) => {
      if (strength >= 70) return 2;
      if (strength >= 50) return 1.5;
      return 1;
    };

    // Draw resistance levels
    resistanceLevels.forEach((levelData) => {
      const price = typeof levelData === 'object' ? levelData.price : levelData;
      const strength = typeof levelData === 'object' ? (levelData.strength || 50) : 50;

      const lineSeries = chart.addLineSeries({
        color: getStrengthColor(strength, true),
        lineWidth: getStrengthWidth(strength),
        lineStyle: 2,
        crosshairMarkerVisible: false,
        priceLineVisible: false,
        lastValueVisible: true,
        title: `R${strength}`,
      });
      lineSeries.setData([
        { time: firstDate, value: price },
        { time: lastDate, value: price },
      ]);
    });

    // Draw support levels
    supportLevels.forEach((levelData) => {
      const price = typeof levelData === 'object' ? levelData.price : levelData;
      const strength = typeof levelData === 'object' ? (levelData.strength || 50) : 50;

      const lineSeries = chart.addLineSeries({
        color: getStrengthColor(strength, false),
        lineWidth: getStrengthWidth(strength),
        lineStyle: 2,
        crosshairMarkerVisible: false,
        priceLineVisible: false,
        lastValueVisible: true,
        title: `S${strength}`,
      });
      lineSeries.setData([
        { time: firstDate, value: price },
        { time: lastDate, value: price },
      ]);
    });

    // Draw Absolute Resistance (prominent styling)
    if (absoluteResistance && absoluteResistance.price) {
      const absResLine = chart.addLineSeries({
        color: '#FF5252',
        lineWidth: 3,
        lineStyle: 0,
        crosshairMarkerVisible: true,
        priceLineVisible: false,
        lastValueVisible: true,
        title: 'ATH',
      });
      absResLine.setData([
        { time: firstDate, value: absoluteResistance.price },
        { time: lastDate, value: absoluteResistance.price },
      ]);
    }

    // Draw Absolute Support (prominent styling)
    if (absoluteSupport && absoluteSupport.price) {
      const absSupLine = chart.addLineSeries({
        color: '#00E676',
        lineWidth: 3,
        lineStyle: 0,
        crosshairMarkerVisible: true,
        priceLineVisible: false,
        lastValueVisible: true,
        title: 'ATL',
      });
      absSupLine.setData([
        { time: firstDate, value: absoluteSupport.price },
        { time: lastDate, value: absoluteSupport.price },
      ]);
    }

    // Draw Resistance Swing Patterns
    resistanceSwings.forEach((swing) => {
      if (!swing.start?.date || !swing.middle?.date || !swing.end?.date) return;
      if (swing.start.date === swing.middle.date || swing.middle.date === swing.end.date || swing.start.date === swing.end.date) return;

      const swingLine = chart.addLineSeries({
        color: 'rgba(242, 54, 69, 0.5)',
        lineWidth: 1.5,
        lineStyle: 0,
        crosshairMarkerVisible: false,
        priceLineVisible: false,
        lastValueVisible: false,
      });

      const swingData = [
        { time: swing.start.date, value: swing.start.price },
        { time: swing.middle.date, value: swing.middle.price },
        { time: swing.end.date, value: swing.end.price },
      ].sort((a, b) => (a.time > b.time ? 1 : -1));

      swingLine.setData(swingData);

      swingLine.setMarkers([
        {
          time: swing.start.date,
          position: 'belowBar',
          color: '#F23645',
          shape: 'circle',
          size: 0.5,
        },
        {
          time: swing.middle.date,
          position: 'aboveBar',
          color: '#F23645',
          shape: 'arrowDown',
          size: 1,
        },
        {
          time: swing.end.date,
          position: 'belowBar',
          color: '#F23645',
          shape: 'circle',
          size: 0.5,
        },
      ]);
    });

    // Draw Support Swing Patterns
    supportSwings.forEach((swing) => {
      if (!swing.start?.date || !swing.middle?.date || !swing.end?.date) return;
      if (swing.start.date === swing.middle.date || swing.middle.date === swing.end.date || swing.start.date === swing.end.date) return;

      const swingLine = chart.addLineSeries({
        color: 'rgba(8, 153, 129, 0.5)',
        lineWidth: 1.5,
        lineStyle: 0,
        crosshairMarkerVisible: false,
        priceLineVisible: false,
        lastValueVisible: false,
      });

      const swingData = [
        { time: swing.start.date, value: swing.start.price },
        { time: swing.middle.date, value: swing.middle.price },
        { time: swing.end.date, value: swing.end.price },
      ].sort((a, b) => (a.time > b.time ? 1 : -1));

      swingLine.setData(swingData);

      swingLine.setMarkers([
        {
          time: swing.start.date,
          position: 'aboveBar',
          color: '#089981',
          shape: 'circle',
          size: 0.5,
        },
        {
          time: swing.middle.date,
          position: 'belowBar',
          color: '#089981',
          shape: 'arrowUp',
          size: 1,
        },
        {
          time: swing.end.date,
          position: 'aboveBar',
          color: '#089981',
          shape: 'circle',
          size: 0.5,
        },
      ]);
    });

    // Volume Series with gradient effect
    const volumeSeries = chart.addHistogramSeries({
      priceFormat: {
        type: 'volume',
      },
      priceScaleId: 'volume',
    });

    chart.priceScale('volume').applyOptions({
      scaleMargins: {
        top: 0.85,
        bottom: 0,
      },
      visible: false,
    });

    const volumeData = data.map(d => ({
      time: d.date,
      value: d.Volume,
      color: d.Open > d.Close
        ? 'rgba(242, 54, 69, 0.5)'
        : 'rgba(8, 153, 129, 0.5)',
    }));
    volumeData.sort((a, b) => (a.time > b.time ? 1 : -1));
    volumeSeries.setData(volumeData);

    // Fit Content
    chart.timeScale().fitContent();

    // Click handler for price scale labels
    const handlePriceScaleClick = (e) => {
      const container = chartContainerRef.current;
      if (!container) return;

      const rect = container.getBoundingClientRect();
      const clickX = e.clientX - rect.left;
      const containerWidth = rect.width;

      // Check if click is on the right price scale area (last 80px)
      if (clickX < containerWidth - 80) return;

      // Get the price at the click position
      const clickY = e.clientY - rect.top;

      // Get all level prices
      const allLevels = [
        ...resistanceLevels.map(l => typeof l === 'object' ? l.price : l),
        ...supportLevels.map(l => typeof l === 'object' ? l.price : l),
        absoluteResistance?.price,
        absoluteSupport?.price,
      ].filter(Boolean);

      if (allLevels.length === 0) return;

      const series = candlestickSeriesRef.current;
      if (!series) return;

      // Find the level whose Y coordinate is closest to click
      let closestLevel = null;
      let closestDistance = Infinity;

      allLevels.forEach(price => {
        const y = series.priceToCoordinate(price);
        if (y !== null && y !== undefined) {
          const distance = Math.abs(y - clickY);
          if (distance < closestDistance && distance < 25) { // Within 25px
            closestDistance = distance;
            closestLevel = price;
          }
        }
      });

      if (closestLevel !== null) {
        // Find the first occurrence date for this level
        const levelMap = levelMapRef.current;
        let targetDate = levelMap.get(closestLevel);

        // If no exact match, find closest price in map
        if (!targetDate) {
          let closestMapPrice = null;
          let closestMapDiff = Infinity;
          for (const [mapPrice] of levelMap.entries()) {
            const diff = Math.abs(mapPrice - closestLevel);
            if (diff < closestMapDiff) {
              closestMapDiff = diff;
              closestMapPrice = mapPrice;
            }
          }
          if (closestMapPrice && closestMapDiff / closestLevel < 0.03) {
            targetDate = levelMap.get(closestMapPrice);
          }
        }

        if (targetDate) {
          // Find the bar index for this date
          const targetIndex = candleData.findIndex(d => d.time === targetDate);
          if (targetIndex !== -1) {
            // Scroll to show this date (position it at 1/4 from left)
            const timeScale = chart.timeScale();
            const visibleRange = timeScale.getVisibleLogicalRange();
            if (visibleRange) {
              const barsCount = Math.floor(visibleRange.to - visibleRange.from);
              const from = Math.max(0, targetIndex - Math.floor(barsCount / 4));
              const to = from + barsCount;
              timeScale.setVisibleLogicalRange({ from, to });
            }
          }
        }
      }
    };

    // Add click listener to container
    chartContainerRef.current.addEventListener('click', handlePriceScaleClick);
    chartContainerRef.current.style.cursor = 'default';

    // Change cursor when hovering over price scale
    const handleMouseMove = (e) => {
      const container = chartContainerRef.current;
      if (!container) return;

      const rect = container.getBoundingClientRect();
      const mouseX = e.clientX - rect.left;
      const mouseY = e.clientY - rect.top;
      const containerWidth = rect.width;

      // Check if mouse is on the right price scale area (last 80px)
      if (mouseX >= containerWidth - 80) {
        // Check if mouse is near a level label
        const series = candlestickSeriesRef.current;
        if (series) {
          const allLevels = [
            ...resistanceLevels.map(l => typeof l === 'object' ? l.price : l),
            ...supportLevels.map(l => typeof l === 'object' ? l.price : l),
            absoluteResistance?.price,
            absoluteSupport?.price,
          ].filter(Boolean);

          const isNearLabel = allLevels.some(price => {
            const y = series.priceToCoordinate(price);
            return y !== null && Math.abs(y - mouseY) < 15;
          });

          container.style.cursor = isNearLabel ? 'pointer' : 'default';
        }
      } else {
        container.style.cursor = 'default';
      }
    };

    chartContainerRef.current.addEventListener('mousemove', handleMouseMove);

    // Resize Handler
    const handleResize = () => {
      if (chartContainerRef.current) {
        chart.applyOptions({ width: chartContainerRef.current.clientWidth });
      }
    };

    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      if (chartContainerRef.current) {
        chartContainerRef.current.removeEventListener('click', handlePriceScaleClick);
        chartContainerRef.current.removeEventListener('mousemove', handleMouseMove);
      }
      chart.remove();
    };
  }, [data, resistanceLevels, supportLevels, resistanceSwings, supportSwings, absoluteResistance, absoluteSupport, theme, isDark]);

  return (
    <div
      ref={chartContainerRef}
      className="chart-container"
      style={{
        width: '100%',
        height: '600px',
        position: 'relative',
        borderRadius: '8px',
        overflow: 'hidden',
      }}
    />
  );
};

export default CandlestickChart;
