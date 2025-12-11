import React, { useState, useEffect, useRef, useCallback } from 'react';
import TextField from '@mui/material/TextField';
import Paper from '@mui/material/Paper';
import List from '@mui/material/List';
import ListItem from '@mui/material/ListItem';
import ListItemText from '@mui/material/ListItemText';
import CircularProgress from '@mui/material/CircularProgress';
import InputAdornment from '@mui/material/InputAdornment';
import IconButton from '@mui/material/IconButton';
import Chip from '@mui/material/Chip';
import Box from '@mui/material/Box';
import axios from 'axios';

const API_URL = process.env.REACT_APP_API_URL || "http://127.0.0.1:5000";

// Popular stocks for quick selection
const QUICK_PICKS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "SPY"];

function StockSearch({
  value,
  onChange,
  theme = 'light',
  label = "Stock Symbol",
  showQuickPicks = true,
  size = "small",
  fullWidth = true,
}) {
  const isDark = theme === 'dark';
  const [inputValue, setInputValue] = useState(value || '');
  const [suggestions, setSuggestions] = useState([]);
  const [loading, setLoading] = useState(false);
  const [showDropdown, setShowDropdown] = useState(false);
  const [selectedIndex, setSelectedIndex] = useState(-1);
  const [stockInfo, setStockInfo] = useState(null);
  const inputRef = useRef(null);
  const dropdownRef = useRef(null);
  const debounceRef = useRef(null);

  // Debounced search
  const searchStocks = useCallback(async (query) => {
    if (!query || query.length < 1) {
      setSuggestions([]);
      return;
    }

    setLoading(true);
    try {
      const response = await axios.get(`${API_URL}/api/stocks/search`, {
        params: { q: query, limit: 8 }
      });
      setSuggestions(response.data.results || []);
    } catch (error) {
      console.error('Stock search failed:', error);
      setSuggestions([]);
    } finally {
      setLoading(false);
    }
  }, []);

  // Handle input change with debounce
  const handleInputChange = (e) => {
    const newValue = e.target.value.toUpperCase();
    setInputValue(newValue);
    setSelectedIndex(-1);
    setShowDropdown(true);

    // Clear previous debounce
    if (debounceRef.current) {
      clearTimeout(debounceRef.current);
    }

    // Debounce search
    debounceRef.current = setTimeout(() => {
      searchStocks(newValue);
    }, 300);
  };

  // Handle selection
  const handleSelect = (stock) => {
    const symbol = typeof stock === 'string' ? stock : stock.symbol;
    setInputValue(symbol);
    setShowDropdown(false);
    setStockInfo(typeof stock === 'object' ? stock : null);
    onChange(symbol);
  };

  // Handle keyboard navigation
  const handleKeyDown = (e) => {
    if (!showDropdown || suggestions.length === 0) {
      if (e.key === 'Enter') {
        handleSelect(inputValue);
      }
      return;
    }

    switch (e.key) {
      case 'ArrowDown':
        e.preventDefault();
        setSelectedIndex(prev =>
          prev < suggestions.length - 1 ? prev + 1 : prev
        );
        break;
      case 'ArrowUp':
        e.preventDefault();
        setSelectedIndex(prev => prev > 0 ? prev - 1 : -1);
        break;
      case 'Enter':
        e.preventDefault();
        if (selectedIndex >= 0 && suggestions[selectedIndex]) {
          handleSelect(suggestions[selectedIndex]);
        } else {
          handleSelect(inputValue);
        }
        break;
      case 'Escape':
        setShowDropdown(false);
        break;
      default:
        break;
    }
  };

  // Close dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (event) => {
      if (
        dropdownRef.current &&
        !dropdownRef.current.contains(event.target) &&
        inputRef.current &&
        !inputRef.current.contains(event.target)
      ) {
        setShowDropdown(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  // Sync external value changes
  useEffect(() => {
    if (value !== inputValue) {
      setInputValue(value || '');
    }
  }, [value]);

  // Load initial suggestions on focus
  const handleFocus = async () => {
    setShowDropdown(true);
    if (!inputValue && suggestions.length === 0) {
      setLoading(true);
      try {
        const response = await axios.get(`${API_URL}/api/stocks/search`, {
          params: { limit: 8 }
        });
        setSuggestions(response.data.results || []);
      } catch (error) {
        console.error('Failed to load suggestions:', error);
      } finally {
        setLoading(false);
      }
    }
  };

  const inputStyles = {
    '& .MuiOutlinedInput-root': {
      backgroundColor: isDark ? 'rgba(255,255,255,0.05)' : '#fff',
      '& fieldset': {
        borderColor: isDark ? 'rgba(255,255,255,0.2)' : 'rgba(0,0,0,0.23)',
      },
      '&:hover fieldset': {
        borderColor: isDark ? 'rgba(255,255,255,0.3)' : 'rgba(0,0,0,0.5)',
      },
      '&.Mui-focused fieldset': {
        borderColor: '#2962FF',
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
    <Box sx={{ position: 'relative' }}>
      {/* Quick Pick Chips */}
      {showQuickPicks && (
        <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5, mb: 1 }}>
          {QUICK_PICKS.map((symbol) => (
            <Chip
              key={symbol}
              label={symbol}
              onClick={() => handleSelect(symbol)}
              color={inputValue === symbol ? "primary" : "default"}
              variant={inputValue === symbol ? "filled" : "outlined"}
              size="small"
              sx={{
                fontSize: '12px',
                height: '24px',
                cursor: 'pointer',
              }}
            />
          ))}
        </Box>
      )}

      {/* Search Input */}
      <TextField
        ref={inputRef}
        label={label}
        variant="outlined"
        size={size}
        fullWidth={fullWidth}
        value={inputValue}
        onChange={handleInputChange}
        onKeyDown={handleKeyDown}
        onFocus={handleFocus}
        sx={inputStyles}
        InputProps={{
          endAdornment: loading ? (
            <InputAdornment position="end">
              <CircularProgress size={18} />
            </InputAdornment>
          ) : inputValue ? (
            <InputAdornment position="end">
              <IconButton
                size="small"
                onClick={() => {
                  setInputValue('');
                  onChange('');
                  setStockInfo(null);
                }}
                sx={{ color: isDark ? 'rgba(255,255,255,0.5)' : 'rgba(0,0,0,0.5)' }}
              >
                <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
                  <path d="M19 6.41L17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12z"/>
                </svg>
              </IconButton>
            </InputAdornment>
          ) : null,
        }}
        placeholder="Search by symbol or name..."
      />

      {/* Stock Info Display */}
      {stockInfo && stockInfo.name && (
        <Box sx={{
          fontSize: '11px',
          color: isDark ? 'rgba(255,255,255,0.6)' : 'rgba(0,0,0,0.6)',
          mt: 0.5,
          px: 0.5,
        }}>
          {stockInfo.name} ({stockInfo.exchange})
        </Box>
      )}

      {/* Dropdown Suggestions */}
      {showDropdown && suggestions.length > 0 && (
        <Paper
          ref={dropdownRef}
          sx={{
            position: 'absolute',
            top: showQuickPicks ? '76px' : '44px',
            left: 0,
            right: 0,
            zIndex: 1000,
            maxHeight: '300px',
            overflow: 'auto',
            backgroundColor: isDark ? '#2d2d2d' : '#fff',
            border: `1px solid ${isDark ? 'rgba(255,255,255,0.1)' : 'rgba(0,0,0,0.1)'}`,
            boxShadow: isDark
              ? '0 4px 12px rgba(0,0,0,0.5)'
              : '0 4px 12px rgba(0,0,0,0.15)',
          }}
        >
          <List dense disablePadding>
            {suggestions.map((stock, index) => (
              <ListItem
                key={stock.symbol}
                button
                selected={index === selectedIndex}
                onClick={() => handleSelect(stock)}
                sx={{
                  '&:hover': {
                    backgroundColor: isDark ? 'rgba(255,255,255,0.1)' : 'rgba(0,0,0,0.04)',
                  },
                  '&.Mui-selected': {
                    backgroundColor: isDark ? 'rgba(41, 98, 255, 0.2)' : 'rgba(41, 98, 255, 0.08)',
                  },
                  borderBottom: `1px solid ${isDark ? 'rgba(255,255,255,0.05)' : 'rgba(0,0,0,0.05)'}`,
                }}
              >
                <ListItemText
                  primary={
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      <span style={{
                        fontWeight: 600,
                        color: isDark ? '#fff' : '#000',
                        minWidth: '60px',
                      }}>
                        {stock.symbol}
                      </span>
                      <span style={{
                        fontSize: '11px',
                        color: isDark ? 'rgba(255,255,255,0.5)' : 'rgba(0,0,0,0.5)',
                        padding: '2px 6px',
                        borderRadius: '4px',
                        backgroundColor: isDark ? 'rgba(255,255,255,0.1)' : 'rgba(0,0,0,0.05)',
                      }}>
                        {stock.exchange}
                      </span>
                    </Box>
                  }
                  secondary={
                    <span style={{
                      color: isDark ? 'rgba(255,255,255,0.6)' : 'rgba(0,0,0,0.6)',
                      fontSize: '12px',
                    }}>
                      {stock.name}
                    </span>
                  }
                  sx={{
                    '& .MuiListItemText-primary': {
                      color: isDark ? '#fff' : '#000',
                    },
                  }}
                />
              </ListItem>
            ))}
          </List>
        </Paper>
      )}
    </Box>
  );
}

export default StockSearch;
