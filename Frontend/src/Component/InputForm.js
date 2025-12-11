import React from "react";
import TextField from "@mui/material/TextField";
import Box from "@mui/material/Box";
import Slider from "@mui/material/Slider";
import Typography from "@mui/material/Typography";
import FormControl from "@mui/material/FormControl";
import InputLabel from "@mui/material/InputLabel";
import Select from "@mui/material/Select";
import MenuItem from "@mui/material/MenuItem";

function InputForm({
  inputStock,
  setInputStock,
  firstRetracement,
  setFirstRetracement,
  secondRetracement,
  setSecondRetracement,
  touchCount,
  setTouchCount,
  levelRange,
  setLevelRange,
  timeframe,
  setTimeframe
}) {
  const handleChange = (e) => {
    setInputStock(e.target.value.toUpperCase());
  };

  return (
    <Box
      component='form'
      sx={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        "& > :not(style)": { m: 1, width: "25ch" },
      }}
      noValidate
      autoComplete='off'
      onSubmit={(e) => e.preventDefault()}
    >
      <TextField
        id='stock-input'
        label='Stock Symbol'
        variant='outlined'
        value={inputStock}
        onChange={handleChange}
      />

      <FormControl sx={{ minWidth: 200 }}>
        <InputLabel id="timeframe-label">Timeframe</InputLabel>
        <Select
          labelId="timeframe-label"
          id="timeframe-select"
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

      <Box sx={{ width: 300, mt: 2 }}>
        <Typography id="first-retracement-slider" gutterBottom>
          First Retracement: {firstRetracement}%
        </Typography>
        <Slider
          aria-label="First Retracement"
          value={firstRetracement}
          onChange={(e, val) => setFirstRetracement(val)}
          valueLabelDisplay="auto"
          step={1}
          min={1}
          max={50}
        />
      </Box>

      <Box sx={{ width: 300 }}>
        <Typography id="second-retracement-slider" gutterBottom>
          Second Retracement: {secondRetracement}%
        </Typography>
        <Slider
          aria-label="Second Retracement"
          value={secondRetracement}
          onChange={(e, val) => setSecondRetracement(val)}
          valueLabelDisplay="auto"
          step={1}
          min={1}
          max={50}
        />
      </Box>

      <Box sx={{ width: 300 }}>
        <Typography id="touch-count-slider" gutterBottom>
          Touch Count: {touchCount}
        </Typography>
        <Slider
          aria-label="Touch Count"
          value={touchCount}
          onChange={(e, val) => setTouchCount(val)}
          valueLabelDisplay="auto"
          step={1}
          min={1}
          max={10}
        />
      </Box>

      <Box sx={{ width: 300 }}>
        <Typography id="level-range-slider" gutterBottom>
          Level Range: {(levelRange * 100).toFixed(1)}%
        </Typography>
        <Slider
          aria-label="Level Range"
          value={levelRange}
          onChange={(e, val) => setLevelRange(val)}
          valueLabelDisplay="auto"
          step={0.005}
          min={0.001}
          max={0.1}
        />
      </Box>
    </Box>
  );
}

export default InputForm;
