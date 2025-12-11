import React from 'react';
import Box from '@mui/material/Box';
import IconButton from '@mui/material/IconButton';
import Slider from '@mui/material/Slider';
import Tooltip from '@mui/material/Tooltip';
import Typography from '@mui/material/Typography';

function PlaybackControls({ visualState, onStateChange, theme }) {
  const isDark = theme === 'dark';
  const { isPlaying, speed, currentBarIndex, totalBars } = visualState;

  const handlePlayPause = () => {
    onStateChange(prev => ({ ...prev, isPlaying: !prev.isPlaying }));
  };

  const handleSpeedChange = (newSpeed) => {
    onStateChange(prev => ({ ...prev, speed: newSpeed }));
  };

  const handleProgressChange = (event, newValue) => {
    onStateChange(prev => ({ ...prev, currentBarIndex: newValue }));
  };

  const handleStepForward = () => {
    onStateChange(prev => ({
      ...prev,
      currentBarIndex: Math.min(prev.currentBarIndex + 1, totalBars - 1),
    }));
  };

  const handleStepBackward = () => {
    onStateChange(prev => ({
      ...prev,
      currentBarIndex: Math.max(prev.currentBarIndex - 1, 0),
    }));
  };

  const handleJumpToStart = () => {
    onStateChange(prev => ({ ...prev, currentBarIndex: 0, isPlaying: false }));
  };

  const handleJumpToEnd = () => {
    onStateChange(prev => ({ ...prev, currentBarIndex: totalBars - 1, isPlaying: false }));
  };

  const speedOptions = [1, 2, 5, 10];

  const iconColor = isDark ? 'rgba(255,255,255,0.87)' : 'rgba(0,0,0,0.87)';
  const disabledColor = isDark ? 'rgba(255,255,255,0.3)' : 'rgba(0,0,0,0.3)';

  return (
    <Box
      sx={{
        display: 'flex',
        alignItems: 'center',
        gap: 2,
        p: 1.5,
        mt: 1,
        backgroundColor: isDark ? 'rgba(255,255,255,0.05)' : 'rgba(0,0,0,0.02)',
        borderRadius: 1,
      }}
    >
      {/* Playback Controls */}
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
        {/* Jump to Start */}
        <Tooltip title="Jump to Start">
          <IconButton size="small" onClick={handleJumpToStart}>
            <svg width="18" height="18" viewBox="0 0 24 24" fill={iconColor}>
              <path d="M6 6h2v12H6zm3.5 6l8.5 6V6z"/>
            </svg>
          </IconButton>
        </Tooltip>

        {/* Step Backward */}
        <Tooltip title="Step Backward">
          <IconButton size="small" onClick={handleStepBackward} disabled={currentBarIndex <= 0}>
            <svg width="18" height="18" viewBox="0 0 24 24" fill={currentBarIndex <= 0 ? disabledColor : iconColor}>
              <path d="M11 18V6l-8.5 6 8.5 6zm.5-6l8.5 6V6l-8.5 6z"/>
            </svg>
          </IconButton>
        </Tooltip>

        {/* Play/Pause */}
        <Tooltip title={isPlaying ? "Pause" : "Play"}>
          <IconButton
            onClick={handlePlayPause}
            sx={{
              backgroundColor: '#2962FF',
              color: '#fff',
              '&:hover': { backgroundColor: '#1e4bd8' },
              width: 36,
              height: 36,
            }}
          >
            {isPlaying ? (
              <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor">
                <path d="M6 19h4V5H6v14zm8-14v14h4V5h-4z"/>
              </svg>
            ) : (
              <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor">
                <path d="M8 5v14l11-7z"/>
              </svg>
            )}
          </IconButton>
        </Tooltip>

        {/* Step Forward */}
        <Tooltip title="Step Forward">
          <IconButton size="small" onClick={handleStepForward} disabled={currentBarIndex >= totalBars - 1}>
            <svg width="18" height="18" viewBox="0 0 24 24" fill={currentBarIndex >= totalBars - 1 ? disabledColor : iconColor}>
              <path d="M4 18l8.5-6L4 6v12zm9-12v12l8.5-6L13 6z"/>
            </svg>
          </IconButton>
        </Tooltip>

        {/* Jump to End */}
        <Tooltip title="Jump to End">
          <IconButton size="small" onClick={handleJumpToEnd}>
            <svg width="18" height="18" viewBox="0 0 24 24" fill={iconColor}>
              <path d="M6 18l8.5-6L6 6v12zM16 6v12h2V6h-2z"/>
            </svg>
          </IconButton>
        </Tooltip>
      </Box>

      {/* Progress Slider */}
      <Box sx={{ flex: 1, display: 'flex', alignItems: 'center', gap: 2 }}>
        <Typography
          variant="caption"
          sx={{ color: isDark ? 'rgba(255,255,255,0.7)' : 'rgba(0,0,0,0.7)', minWidth: 60 }}
        >
          Bar {currentBarIndex + 1}/{totalBars}
        </Typography>
        <Slider
          value={currentBarIndex}
          onChange={handleProgressChange}
          min={0}
          max={Math.max(0, totalBars - 1)}
          sx={{
            color: '#2962FF',
            '& .MuiSlider-thumb': {
              width: 14,
              height: 14,
            },
          }}
        />
      </Box>

      {/* Speed Controls */}
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
        <Typography
          variant="caption"
          sx={{ color: isDark ? 'rgba(255,255,255,0.5)' : 'rgba(0,0,0,0.5)' }}
        >
          Speed:
        </Typography>
        {speedOptions.map((s) => (
          <Tooltip key={s} title={`${s}x speed`}>
            <IconButton
              size="small"
              onClick={() => handleSpeedChange(s)}
              sx={{
                fontSize: '12px',
                fontWeight: speed === s ? 600 : 400,
                color: speed === s ? '#2962FF' : (isDark ? 'rgba(255,255,255,0.7)' : 'rgba(0,0,0,0.7)'),
                backgroundColor: speed === s ? 'rgba(41, 98, 255, 0.1)' : 'transparent',
                minWidth: 32,
                '&:hover': {
                  backgroundColor: speed === s ? 'rgba(41, 98, 255, 0.2)' : 'rgba(0,0,0,0.05)',
                },
              }}
            >
              {s}x
            </IconButton>
          </Tooltip>
        ))}
      </Box>
    </Box>
  );
}

export default PlaybackControls;
