import React, { useState } from 'react';
import {
  Box,
  Paper,
  Typography,
  TextField,
  Button,
  Card,
  CardContent,
  Grid,
  CircularProgress,
  Alert,
  Chip,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  InputAdornment
} from '@mui/material';
import SearchIcon from '@mui/icons-material/Search';
import TrendingUpIcon from '@mui/icons-material/TrendingUp';
import TrendingDownIcon from '@mui/icons-material/TrendingDown';
import axios from 'axios';

const API_URL = process.env.REACT_APP_API_URL || 'http://127.0.0.1:5000';

const PredictionPanel = ({ hasModel }) => {
  const [symbol, setSymbol] = useState('AAPL');
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const getPrediction = async () => {
    if (!symbol) return;

    setLoading(true);
    setError(null);

    try {
      const response = await axios.get(`${API_URL}/api/ml/predictions`, {
        params: { symbol: symbol.toUpperCase() }
      });
      setPrediction(response.data);
    } catch (err) {
      setError(err.response?.data?.error || 'Failed to get prediction');
      setPrediction(null);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter') {
      getPrediction();
    }
  };

  const formatReturn = (value) => {
    const percent = (value * 100).toFixed(3);
    return `${percent > 0 ? '+' : ''}${percent}%`;
  };

  const getReturnColor = (value) => {
    if (value > 0.01) return 'success';
    if (value > 0) return 'info';
    if (value > -0.01) return 'warning';
    return 'error';
  };

  if (!hasModel) {
    return (
      <Paper sx={{ p: 3, textAlign: 'center' }}>
        <Typography color="text.secondary">
          No trained model available. Please run the training pipeline first.
        </Typography>
      </Paper>
    );
  }

  return (
    <Box>
      <Typography variant="h6" gutterBottom>
        Stock Prediction
      </Typography>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
        Get ML predictions for any stock using the trained ensemble model.
      </Typography>

      {/* Search Input */}
      <Paper sx={{ p: 2, mb: 3 }}>
        <Box sx={{ display: 'flex', gap: 2 }}>
          <TextField
            label="Stock Symbol"
            value={symbol}
            onChange={(e) => setSymbol(e.target.value.toUpperCase())}
            onKeyPress={handleKeyPress}
            placeholder="e.g., AAPL, MSFT, GOOGL"
            size="small"
            sx={{ flex: 1, maxWidth: 300 }}
            InputProps={{
              startAdornment: (
                <InputAdornment position="start">
                  <SearchIcon />
                </InputAdornment>
              )
            }}
          />
          <Button
            variant="contained"
            onClick={getPrediction}
            disabled={loading || !symbol}
          >
            {loading ? <CircularProgress size={24} /> : 'Get Prediction'}
          </Button>
        </Box>
      </Paper>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError(null)}>
          {error}
        </Alert>
      )}

      {/* Prediction Results */}
      {prediction && (
        <Grid container spacing={3}>
          {/* Summary Card */}
          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h5" gutterBottom>
                  {prediction.symbol}
                </Typography>
                <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                  Prediction Date: {prediction.date}
                </Typography>

                <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 2 }}>
                  <Typography variant="h4" fontWeight="bold">
                    {formatReturn(prediction.predictions?.ensemble || 0)}
                  </Typography>
                  {prediction.predictions?.ensemble > 0 ? (
                    <TrendingUpIcon color="success" fontSize="large" />
                  ) : (
                    <TrendingDownIcon color="error" fontSize="large" />
                  )}
                </Box>

                <Chip
                  label={`${prediction.horizon_days}-Day Prediction`}
                  color="primary"
                  size="small"
                />
              </CardContent>
            </Card>
          </Grid>

          {/* Model Breakdown */}
          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Model Breakdown
                </Typography>
                <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                  Predictions from individual models in the ensemble
                </Typography>

                <TableContainer>
                  <Table size="small">
                    <TableHead>
                      <TableRow>
                        <TableCell>Model</TableCell>
                        <TableCell align="right">Prediction</TableCell>
                        <TableCell align="right">Signal</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {prediction.predictions && Object.entries(prediction.predictions).map(([model, value]) => (
                        <TableRow key={model}>
                          <TableCell sx={{ textTransform: 'capitalize' }}>
                            {model.replace('_', ' ')}
                          </TableCell>
                          <TableCell align="right">
                            {formatReturn(value)}
                          </TableCell>
                          <TableCell align="right">
                            <Chip
                              label={value > 0 ? 'BUY' : 'SELL'}
                              color={getReturnColor(value)}
                              size="small"
                              variant="outlined"
                            />
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </TableContainer>
              </CardContent>
            </Card>
          </Grid>

          {/* Disclaimer */}
          <Grid item xs={12}>
            <Alert severity="warning">
              <strong>Disclaimer:</strong> These predictions are generated by machine learning models
              and should not be considered financial advice. Past performance does not guarantee future
              results. Always do your own research before making investment decisions.
            </Alert>
          </Grid>
        </Grid>
      )}
    </Box>
  );
};

export default PredictionPanel;
