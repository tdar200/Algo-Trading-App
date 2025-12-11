import React, { useState, useEffect, useCallback } from 'react';
import {
  Box,
  Paper,
  Typography,
  Button,
  LinearProgress,
  Grid,
  Card,
  CardContent,
  Alert,
  Chip,
  Divider,
  CircularProgress
} from '@mui/material';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import AutoGraphIcon from '@mui/icons-material/AutoGraph';
import TrendingUpIcon from '@mui/icons-material/TrendingUp';
import axios from 'axios';
import io from 'socket.io-client';
import PatternViewer from './PatternViewer';
import PredictionPanel from './PredictionPanel';
import FeatureImportance from './FeatureImportance';

const API_URL = process.env.REACT_APP_API_URL || 'http://127.0.0.1:5000';

const MLDashboard = () => {
  const [status, setStatus] = useState(null);
  const [isRunning, setIsRunning] = useState(false);
  const [hasModel, setHasModel] = useState(false);
  const [error, setError] = useState(null);
  const [patterns, setPatterns] = useState([]);
  const [features, setFeatures] = useState([]);
  const [socket, setSocket] = useState(null);
  const [activeTab, setActiveTab] = useState('overview');

  // Check initial status
  useEffect(() => {
    checkStatus();
    loadPatterns();
    loadFeatureImportance();
  }, []);

  // Setup WebSocket connection
  useEffect(() => {
    const newSocket = io(API_URL);

    newSocket.on('connect', () => {
      console.log('Connected to ML WebSocket');
      newSocket.emit('subscribe_ml_progress');
    });

    newSocket.on('ml_progress', (data) => {
      setStatus(data);
      setIsRunning(data.stage !== 'complete' && data.stage !== 'error');
    });

    newSocket.on('ml_complete', (data) => {
      setStatus({ stage: 'complete', progress: 100, message: 'Training complete!' });
      setIsRunning(false);
      setHasModel(true);
      loadPatterns();
      loadFeatureImportance();
    });

    newSocket.on('ml_error', (data) => {
      setError(data.message);
      setIsRunning(false);
    });

    setSocket(newSocket);

    return () => {
      newSocket.disconnect();
    };
  }, []);

  const checkStatus = async () => {
    try {
      const response = await axios.get(`${API_URL}/api/ml/status`);
      setIsRunning(response.data.is_running);
      setStatus(response.data.status);
      setHasModel(response.data.has_trained_model);
    } catch (err) {
      console.error('Failed to check ML status:', err);
    }
  };

  const loadPatterns = async () => {
    try {
      const response = await axios.get(`${API_URL}/api/ml/patterns`);
      setPatterns(response.data.patterns || []);
    } catch (err) {
      console.log('No patterns available yet');
    }
  };

  const loadFeatureImportance = async () => {
    try {
      const response = await axios.get(`${API_URL}/api/ml/feature-importance?top_n=20`);
      setFeatures(response.data.features || []);
    } catch (err) {
      console.log('No feature importance available yet');
    }
  };

  const startTraining = async () => {
    setError(null);
    setIsRunning(true);
    setStatus({ stage: 'initializing', progress: 0, message: 'Starting...' });

    try {
      await axios.post(`${API_URL}/api/ml/train`, {
        max_stocks: null // Train on all S&P 500 stocks
      });
    } catch (err) {
      setError(err.response?.data?.error || 'Failed to start training');
      setIsRunning(false);
    }
  };

  const getStageColor = (stage) => {
    switch (stage) {
      case 'loading': return 'info';
      case 'features': return 'primary';
      case 'training': return 'warning';
      case 'discovery': return 'secondary';
      case 'complete': return 'success';
      case 'error': return 'error';
      default: return 'default';
    }
  };

  return (
    <Box sx={{ p: 3, maxWidth: 1400, margin: '0 auto' }}>
      <Typography variant="h4" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
        <AutoGraphIcon /> ML Pattern Discovery
      </Typography>

      <Typography variant="body1" color="text.secondary" sx={{ mb: 3 }}>
        Discover universal trading patterns across S&P 500 stocks using GPU-accelerated machine learning.
      </Typography>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError(null)}>
          {error}
        </Alert>
      )}

      {/* Training Status Card */}
      <Paper sx={{ p: 3, mb: 3 }}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
          <Box>
            <Typography variant="h6">Training Pipeline</Typography>
            <Typography variant="body2" color="text.secondary">
              {hasModel ? 'Model trained and ready' : 'No trained model available'}
            </Typography>
          </Box>
          <Button
            variant="contained"
            color="primary"
            startIcon={isRunning ? <CircularProgress size={20} color="inherit" /> : <PlayArrowIcon />}
            onClick={startTraining}
            disabled={isRunning}
          >
            {isRunning ? 'Training...' : 'Start Training'}
          </Button>
        </Box>

        {status && (
          <Box>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 1 }}>
              <Chip
                label={status.stage?.toUpperCase() || 'IDLE'}
                color={getStageColor(status.stage)}
                size="small"
              />
              <Typography variant="body2">{status.message}</Typography>
            </Box>
            <LinearProgress
              variant="determinate"
              value={status.progress || 0}
              sx={{ height: 8, borderRadius: 4 }}
            />
            <Typography variant="caption" color="text.secondary" sx={{ mt: 0.5, display: 'block' }}>
              {status.progress || 0}% complete
            </Typography>
          </Box>
        )}
      </Paper>

      {/* Results Tabs */}
      <Box sx={{ display: 'flex', gap: 1, mb: 2 }}>
        {['overview', 'patterns', 'features', 'predict'].map((tab) => (
          <Chip
            key={tab}
            label={tab.charAt(0).toUpperCase() + tab.slice(1)}
            onClick={() => setActiveTab(tab)}
            color={activeTab === tab ? 'primary' : 'default'}
            variant={activeTab === tab ? 'filled' : 'outlined'}
          />
        ))}
      </Box>

      {/* Tab Content */}
      {activeTab === 'overview' && (
        <Grid container spacing={3}>
          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  <TrendingUpIcon sx={{ mr: 1, verticalAlign: 'middle' }} />
                  Quick Stats
                </Typography>
                <Divider sx={{ mb: 2 }} />
                <Typography variant="body2" color="text.secondary">
                  Model Status: {hasModel ? 'Ready' : 'Not Trained'}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Patterns Discovered: {patterns.length}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Top Features: {features.length}
                </Typography>
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>About This System</Typography>
                <Divider sx={{ mb: 2 }} />
                <Typography variant="body2" color="text.secondary" paragraph>
                  This ML system analyzes all ~500 S&P 500 stocks using 5 years of historical data
                  to discover universal trading patterns.
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Features include: Technical indicators, Statistical measures, Cross-sectional analysis,
                  and Fundamental data.
                </Typography>
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      )}

      {activeTab === 'patterns' && (
        <PatternViewer patterns={patterns} />
      )}

      {activeTab === 'features' && (
        <FeatureImportance features={features} />
      )}

      {activeTab === 'predict' && (
        <PredictionPanel hasModel={hasModel} />
      )}
    </Box>
  );
};

export default MLDashboard;
