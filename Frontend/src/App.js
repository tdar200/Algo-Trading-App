import "./App.css";
import { useEffect, useState, useMemo, useRef, useCallback } from "react";
import axios from "axios";

import CircularProgress from "@mui/material/CircularProgress";
import Alert from "@mui/material/Alert";
import Slider from "@mui/material/Slider";
import FormControl from "@mui/material/FormControl";
import InputLabel from "@mui/material/InputLabel";
import Select from "@mui/material/Select";
import MenuItem from "@mui/material/MenuItem";
import Tooltip from "@mui/material/Tooltip";
import IconButton from "@mui/material/IconButton";
import Tabs from "@mui/material/Tabs";
import Tab from "@mui/material/Tab";
import CandlestickChart from "./Component/Candlestick";
import BacktestPage from "./Component/Backtesting/BacktestPage";
import OptimizationPage from "./Component/Optimization/OptimizationPage";
import StockSearch from "./Component/Common/StockSearch";
import { MLDashboard } from "./Component/ML";

const API_URL = process.env.REACT_APP_API_URL || "http://127.0.0.1:5000";
const DEBOUNCE_DELAY = 500;

// Helper to read query params
const getQueryParams = () => {
  const params = new URLSearchParams(window.location.search);
  return {
    stock: params.get('stock') || 'AMZN',
    timeframe: params.get('timeframe') || '2y',
    firstRetracement: parseFloat(params.get('fr')) || 5,
    secondRetracement: parseFloat(params.get('sr')) || 5,
    touchCount: parseInt(params.get('tc')) || 1,
    levelRange: parseFloat(params.get('lr')) || 0.001,
    theme: params.get('theme') || 'light',
  };
};

const initialParams = getQueryParams();

function App() {
  const [stockData, setStockData] = useState({});
  const [trades, setTrades] = useState([]);
  const [levels, setLevels] = useState([]);
  const [resistanceLevels, setResistanceLevels] = useState([]);
  const [supportLevels, setSupportLevels] = useState([]);
  const [resistanceSwings, setResistanceSwings] = useState([]);
  const [supportSwings, setSupportSwings] = useState([]);
  const [absoluteResistance, setAbsoluteResistance] = useState(null);
  const [absoluteSupport, setAbsoluteSupport] = useState(null);
  const [inputStock, setInputStock] = useState(initialParams.stock);
  const [theme, setTheme] = useState(initialParams.theme);
  const [activeTab, setActiveTab] = useState(0);

  const [firstRetracement, setFirstRetracement] = useState(initialParams.firstRetracement);
  const [secondRetracement, setSecondRetracement] = useState(initialParams.secondRetracement);
  const [touchCount, setTouchCount] = useState(initialParams.touchCount);
  const [levelRange, setLevelRange] = useState(initialParams.levelRange);
  const [timeframe, setTimeframe] = useState(initialParams.timeframe);

  const [debouncedParams, setDebouncedParams] = useState({
    firstRetracement: initialParams.firstRetracement,
    secondRetracement: initialParams.secondRetracement,
    touchCount: initialParams.touchCount,
    levelRange: initialParams.levelRange,
    timeframe: initialParams.timeframe
  });

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const debounceTimerRef = useRef(null);

  // Update URL query params when settings change
  const updateQueryParams = useCallback(() => {
    const params = new URLSearchParams();
    params.set('stock', inputStock);
    params.set('timeframe', timeframe);
    params.set('fr', firstRetracement);
    params.set('sr', secondRetracement);
    params.set('tc', touchCount);
    params.set('lr', levelRange);
    params.set('theme', theme);

    const newUrl = `${window.location.pathname}?${params.toString()}`;
    window.history.replaceState({}, '', newUrl);
  }, [inputStock, timeframe, firstRetracement, secondRetracement, touchCount, levelRange, theme]);

  const toggleTheme = () => {
    setTheme(prev => prev === 'light' ? 'dark' : 'light');
  };

  const isDark = theme === 'dark';

  // Update URL whenever params change (debounced)
  useEffect(() => {
    updateQueryParams();
  }, [updateQueryParams]);

  useEffect(() => {
    if (debounceTimerRef.current) {
      clearTimeout(debounceTimerRef.current);
    }

    debounceTimerRef.current = setTimeout(() => {
      setDebouncedParams({
        firstRetracement,
        secondRetracement,
        touchCount,
        levelRange,
        timeframe
      });
    }, DEBOUNCE_DELAY);

    return () => {
      if (debounceTimerRef.current) {
        clearTimeout(debounceTimerRef.current);
      }
    };
  }, [firstRetracement, secondRetracement, touchCount, levelRange, timeframe]);

  useEffect(() => {
    if (!inputStock) return;

    if (!/^[A-Za-z]{1,5}$/.test(inputStock)) {
      setError("Invalid stock symbol format");
      return;
    }

    setLoading(true);
    setError(null);
    axios
      .get(`${API_URL}/stock?symbol=${inputStock}&first_retracement=${debouncedParams.firstRetracement}&second_retracement=${debouncedParams.secondRetracement}&level_range=${debouncedParams.levelRange}&touch_count=${debouncedParams.touchCount}&period=${debouncedParams.timeframe}`)
      .then((res) => {
        if (res.data.stock_data) {
          setStockData(res.data.stock_data);
          setTrades(res.data.trades || []);
          setLevels(res.data.levels || []);
          setResistanceLevels(res.data.resistance_levels || []);
          setSupportLevels(res.data.support_levels || []);
          setResistanceSwings(res.data.resistance_swings || []);
          setSupportSwings(res.data.support_swings || []);
          setAbsoluteResistance(res.data.absolute_resistance || null);
          setAbsoluteSupport(res.data.absolute_support || null);
        } else {
          setStockData(res.data);
          setTrades([]);
          setLevels([]);
          setResistanceLevels([]);
          setSupportLevels([]);
          setResistanceSwings([]);
          setSupportSwings([]);
          setAbsoluteResistance(null);
          setAbsoluteSupport(null);
        }
      })
      .catch((err) => {
        const errorData = err.response?.data;
        const errorMessage = errorData?.error || (typeof errorData === 'string' ? errorData : err.message) || "Failed to fetch stock data";
        setError(errorMessage);
      })
      .finally(() => {
        setLoading(false);
      });
  }, [inputStock, debouncedParams]);

  const dataArray = useMemo(() => {
    if (!stockData || Object.keys(stockData).length === 0) return [];
    return Object.entries(stockData).map(([date, candlestickData]) => ({
      date,
      ...candlestickData,
    }));
  }, [stockData]);

  const timeframeLabels = {
    '1mo': '1 Month',
    '3mo': '3 Months',
    '6mo': '6 Months',
    '1y': '1 Year',
    '2y': '2 Years',
    '5y': '5 Years',
    '10y': '10 Years',
    'max': 'Max'
  };

  return (
    <div className={`App ${isDark ? 'dark' : 'light'}`}>
      {/* Header */}
      <header className="app-header">
        <div className="header-left">
          <div className="app-title">
            <span className="app-title-icon">&#9651;</span>
            Trading App
          </div>
          <Tabs
            value={activeTab}
            onChange={(e, newValue) => setActiveTab(newValue)}
            className="header-tabs"
            sx={{
              minHeight: '40px',
              '& .MuiTab-root': {
                minHeight: '40px',
                textTransform: 'none',
                fontWeight: 500,
                fontSize: '14px',
                color: isDark ? 'rgba(255,255,255,0.7)' : 'rgba(0,0,0,0.6)',
                '&.Mui-selected': {
                  color: isDark ? '#fff' : '#1976d2',
                },
              },
              '& .MuiTabs-indicator': {
                backgroundColor: isDark ? '#fff' : '#1976d2',
              },
            }}
          >
            <Tab label="Analyzer" />
            <Tab label="Backtesting" />
            <Tab label="Optimization" />
            <Tab label="ML Patterns" />
          </Tabs>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
          {activeTab === 0 && (
            <span className="data-points">
              {dataArray.length > 0 && `${dataArray.length} data points`}
            </span>
          )}
          <Tooltip title={isDark ? "Switch to Light Mode" : "Switch to Dark Mode"}>
            <IconButton onClick={toggleTheme} className="theme-toggle">
              {isDark ? (
                <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
                  <path d="M12 3a9 9 0 109 9c0-.46-.04-.92-.1-1.36a5.389 5.389 0 01-4.4 2.26 5.403 5.403 0 01-3.14-9.8c-.44-.06-.9-.1-1.36-.1z"/>
                </svg>
              ) : (
                <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
                  <path d="M12 7c-2.76 0-5 2.24-5 5s2.24 5 5 5 5-2.24 5-5-2.24-5-5-5zM2 13h2c.55 0 1-.45 1-1s-.45-1-1-1H2c-.55 0-1 .45-1 1s.45 1 1 1zm18 0h2c.55 0 1-.45 1-1s-.45-1-1-1h-2c-.55 0-1 .45-1 1s.45 1 1 1zM11 2v2c0 .55.45 1 1 1s1-.45 1-1V2c0-.55-.45-1-1-1s-1 .45-1 1zm0 18v2c0 .55.45 1 1 1s1-.45 1-1v-2c0-.55-.45-1-1-1s-1 .45-1 1zM5.99 4.58a.996.996 0 00-1.41 0 .996.996 0 000 1.41l1.06 1.06c.39.39 1.03.39 1.41 0s.39-1.03 0-1.41L5.99 4.58zm12.37 12.37a.996.996 0 00-1.41 0 .996.996 0 000 1.41l1.06 1.06c.39.39 1.03.39 1.41 0a.996.996 0 000-1.41l-1.06-1.06zm1.06-10.96a.996.996 0 000-1.41.996.996 0 00-1.41 0l-1.06 1.06c-.39.39-.39 1.03 0 1.41s1.03.39 1.41 0l1.06-1.06zM7.05 18.36a.996.996 0 000-1.41.996.996 0 00-1.41 0l-1.06 1.06c-.39.39-.39 1.03 0 1.41s1.03.39 1.41 0l1.06-1.06z"/>
                </svg>
              )}
            </IconButton>
          </Tooltip>
        </div>
      </header>

      {/* Main Content - All tabs stay mounted to preserve state */}
      <div className="main-content" style={{ display: activeTab === 0 ? 'flex' : 'none' }}>
        {/* Sidebar */}
        <div className="sidebar">
          {/* Stock Selection */}
          <div className="control-card">
            <div className="control-card-title">Stock Selection</div>

            <StockSearch
              value={inputStock}
              onChange={setInputStock}
              theme={theme}
              label="Search Symbol"
              showQuickPicks={true}
            />

            <FormControl fullWidth size="small" sx={{ mt: 2 }}>
              <InputLabel>Timeframe</InputLabel>
              <Select
                value={timeframe}
                label="Timeframe"
                onChange={(e) => setTimeframe(e.target.value)}
              >
                <MenuItem value="1mo">1 Month</MenuItem>
                <MenuItem value="3mo">3 Months</MenuItem>
                <MenuItem value="6mo">6 Months</MenuItem>
                <MenuItem value="1y">1 Year</MenuItem>
                <MenuItem value="2y">2 Years</MenuItem>
                <MenuItem value="5y">5 Years</MenuItem>
                <MenuItem value="10y">10 Years</MenuItem>
                <MenuItem value="max">Max (All Data)</MenuItem>
              </Select>
            </FormControl>
          </div>

          {/* Detection Parameters */}
          <div className="control-card">
            <div className="control-card-title">Detection Parameters</div>

            <div className="parameter-group">
              <Tooltip title="Minimum % move from low to high to detect a swing" placement="top">
                <div className="slider-label">
                  <span className="slider-name">First Retracement</span>
                  <span className="slider-value">{firstRetracement}%</span>
                </div>
              </Tooltip>
              <Slider
                value={firstRetracement}
                onChange={(e, val) => setFirstRetracement(val)}
                min={1}
                max={50}
                size="small"
              />
            </div>

            <div className="parameter-group">
              <Tooltip title="Minimum % pullback to confirm the swing" placement="top">
                <div className="slider-label">
                  <span className="slider-name">Second Retracement</span>
                  <span className="slider-value">{secondRetracement}%</span>
                </div>
              </Tooltip>
              <Slider
                value={secondRetracement}
                onChange={(e, val) => setSecondRetracement(val)}
                min={1}
                max={50}
                size="small"
              />
            </div>

            <div className="parameter-group">
              <Tooltip title="Minimum touches required before showing a level" placement="top">
                <div className="slider-label">
                  <span className="slider-name">Touch Count</span>
                  <span className="slider-value">{touchCount}</span>
                </div>
              </Tooltip>
              <Slider
                value={touchCount}
                onChange={(e, val) => setTouchCount(val)}
                min={1}
                max={10}
                size="small"
              />
            </div>

            <div className="parameter-group">
              <Tooltip title="% range to cluster similar price levels together" placement="top">
                <div className="slider-label">
                  <span className="slider-name">Level Range</span>
                  <span className="slider-value">{(levelRange * 100).toFixed(1)}%</span>
                </div>
              </Tooltip>
              <Slider
                value={levelRange}
                onChange={(e, val) => setLevelRange(val)}
                min={0.001}
                max={0.1}
                step={0.005}
                size="small"
              />
            </div>
          </div>

          {/* Legend */}
          <div className="control-card">
            <div className="control-card-title">Legend</div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
              <div className="legend-item">
                <div className="legend-color resistance" style={{ width: '24px' }}></div>
                <span>Resistance Levels (Red)</span>
              </div>
              <div className="legend-item">
                <div className="legend-color support" style={{ width: '24px' }}></div>
                <span>Support Levels (Blue)</span>
              </div>
              <div className="legend-item">
                <div className="legend-color resistance-swing" style={{ width: '24px' }}></div>
                <span>Resistance Swings</span>
              </div>
              <div className="legend-item">
                <div className="legend-color support-swing" style={{ width: '24px' }}></div>
                <span>Support Swings</span>
              </div>
            </div>
          </div>
        </div>

        {/* Chart Area */}
        <div className="chart-area">
          {error && (
            <Alert severity="error" sx={{ mb: 2 }}>
              {typeof error === 'object' ? JSON.stringify(error) : error}
            </Alert>
          )}

          <div className="chart-card">
            <div className="chart-header">
              <div>
                <div className="chart-title">{inputStock}</div>
                <div className="chart-subtitle">{timeframeLabels[timeframe]} Chart</div>
              </div>
              <div className="legend">
                <div className="legend-item">
                  <div className="legend-color resistance"></div>
                  <span>Resistance</span>
                </div>
                <div className="legend-item">
                  <div className="legend-color support"></div>
                  <span>Support</span>
                </div>
              </div>
            </div>

            <div style={{ position: 'relative', minHeight: '600px' }}>
              {loading && (
                <div className="chart-loading-overlay">
                  <CircularProgress sx={{ color: '#2962FF' }} />
                </div>
              )}

              {dataArray.length > 0 && (
                <CandlestickChart
                  data={dataArray}
                  trades={trades}
                  levels={levels}
                  resistanceLevels={resistanceLevels}
                  supportLevels={supportLevels}
                  resistanceSwings={resistanceSwings}
                  supportSwings={supportSwings}
                  absoluteResistance={absoluteResistance}
                  absoluteSupport={absoluteSupport}
                  theme={theme}
                />
              )}

              {!loading && dataArray.length === 0 && !error && (
                <div className="chart-empty-state">
                  <CircularProgress sx={{ color: '#2962FF' }} />
                </div>
              )}
            </div>

            {/* Absolute Levels */}
            {(absoluteResistance || absoluteSupport) && (
              <div className="absolute-levels-row">
                {absoluteResistance && (
                  <Tooltip title={`Highest detected resistance signal${absoluteResistance.date ? ` (${absoluteResistance.date})` : ''}`} placement="top">
                    <div className="absolute-level-item absolute-resistance">
                      <div className="absolute-level-label">Absolute Resistance</div>
                      <div className="absolute-level-value">${absoluteResistance.price.toFixed(2)}</div>
                      <div className="absolute-level-meta">
                        <span className="absolute-level-strength">{absoluteResistance.strength || 0}% strength</span>
                        {absoluteResistance.date && <span className="absolute-level-date">{absoluteResistance.date}</span>}
                      </div>
                    </div>
                  </Tooltip>
                )}
                {absoluteSupport && (
                  <Tooltip title={`Lowest detected support signal${absoluteSupport.date ? ` (${absoluteSupport.date})` : ''}`} placement="top">
                    <div className="absolute-level-item absolute-support">
                      <div className="absolute-level-label">Absolute Support</div>
                      <div className="absolute-level-value">${absoluteSupport.price.toFixed(2)}</div>
                      <div className="absolute-level-meta">
                        <span className="absolute-level-strength">{absoluteSupport.strength || 0}% strength</span>
                        {absoluteSupport.date && <span className="absolute-level-date">{absoluteSupport.date}</span>}
                      </div>
                    </div>
                  </Tooltip>
                )}
              </div>
            )}

            {/* Stats */}
            <div className="stats-row">
              <div className="stat-item resistance">
                <div className="stat-value">{resistanceLevels.length}</div>
                <div className="stat-label">Resistance</div>
              </div>
              <div className="stat-item support">
                <div className="stat-value">{supportLevels.length}</div>
                <div className="stat-label">Support</div>
              </div>
              <div className="stat-item">
                <div className="stat-value">{resistanceSwings.length}</div>
                <div className="stat-label">R Swings</div>
              </div>
              <div className="stat-item">
                <div className="stat-value">{supportSwings.length}</div>
                <div className="stat-label">S Swings</div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Backtesting Tab - Always mounted, hidden when not active */}
      <div style={{ display: activeTab === 1 ? 'block' : 'none' }}>
        <BacktestPage theme={theme} />
      </div>

      {/* Optimization Tab - Always mounted, hidden when not active */}
      <div style={{ display: activeTab === 2 ? 'block' : 'none' }}>
        <OptimizationPage theme={theme} />
      </div>

      {/* ML Patterns Tab - Always mounted, hidden when not active */}
      <div style={{ display: activeTab === 3 ? 'block' : 'none' }}>
        <MLDashboard />
      </div>
    </div>
  );
}

export default App;
