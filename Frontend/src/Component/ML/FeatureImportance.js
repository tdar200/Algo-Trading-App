import React from 'react';
import {
  Box,
  Paper,
  Typography,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  LinearProgress,
  Chip,
  Tooltip
} from '@mui/material';
import TrendingUpIcon from '@mui/icons-material/TrendingUp';
import TrendingDownIcon from '@mui/icons-material/TrendingDown';

const FeatureImportance = ({ features }) => {
  if (!features || features.length === 0) {
    return (
      <Paper sx={{ p: 3, textAlign: 'center' }}>
        <Typography color="text.secondary">
          No feature importance data available. Run the training pipeline to see feature rankings.
        </Typography>
      </Paper>
    );
  }

  // Get max importance for normalization
  const maxImportance = Math.max(...features.map(f => f.average || f.importance || 0));

  const formatNumber = (value) => {
    if (value === undefined || value === null) return '-';
    return value.toFixed(4);
  };

  const getFeatureCategory = (featureName) => {
    if (featureName.includes('rsi') || featureName.includes('macd') || featureName.includes('sma') || featureName.includes('ema')) {
      return { label: 'Technical', color: 'primary' };
    }
    if (featureName.includes('return') || featureName.includes('vol') || featureName.includes('zscore')) {
      return { label: 'Statistical', color: 'secondary' };
    }
    if (featureName.includes('sector') || featureName.includes('beta') || featureName.includes('rel_')) {
      return { label: 'Cross-sectional', color: 'info' };
    }
    if (featureName.includes('pe_') || featureName.includes('roe') || featureName.includes('margin')) {
      return { label: 'Fundamental', color: 'success' };
    }
    return { label: 'Other', color: 'default' };
  };

  return (
    <Box>
      <Typography variant="h6" gutterBottom>
        Feature Importance Rankings
      </Typography>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
        Top features contributing to model predictions, ranked by average importance across all base models.
      </Typography>

      <TableContainer component={Paper}>
        <Table size="small">
          <TableHead>
            <TableRow>
              <TableCell sx={{ fontWeight: 'bold' }}>Rank</TableCell>
              <TableCell sx={{ fontWeight: 'bold' }}>Feature</TableCell>
              <TableCell sx={{ fontWeight: 'bold' }}>Category</TableCell>
              <TableCell sx={{ fontWeight: 'bold' }} align="right">Importance</TableCell>
              <TableCell sx={{ fontWeight: 'bold', width: 200 }}>Relative</TableCell>
              <TableCell sx={{ fontWeight: 'bold' }} align="center">Direction</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {features.map((feature, index) => {
              const importance = feature.average || feature.importance || 0;
              const normalizedImportance = (importance / maxImportance) * 100;
              const category = getFeatureCategory(feature.feature);
              const meanShap = feature.mean_shap;

              return (
                <TableRow
                  key={feature.feature}
                  sx={{
                    bgcolor: index < 3 ? 'action.hover' : 'inherit',
                    '&:hover': { bgcolor: 'action.selected' }
                  }}
                >
                  <TableCell>
                    <Chip
                      label={index + 1}
                      size="small"
                      color={index < 3 ? 'primary' : 'default'}
                      variant={index < 3 ? 'filled' : 'outlined'}
                    />
                  </TableCell>
                  <TableCell>
                    <Tooltip title={feature.feature} placement="right">
                      <Typography
                        variant="body2"
                        sx={{
                          fontFamily: 'monospace',
                          maxWidth: 200,
                          overflow: 'hidden',
                          textOverflow: 'ellipsis',
                          whiteSpace: 'nowrap'
                        }}
                      >
                        {feature.feature}
                      </Typography>
                    </Tooltip>
                  </TableCell>
                  <TableCell>
                    <Chip
                      label={category.label}
                      size="small"
                      color={category.color}
                      variant="outlined"
                    />
                  </TableCell>
                  <TableCell align="right">
                    <Typography variant="body2" fontWeight="bold">
                      {formatNumber(importance)}
                    </Typography>
                  </TableCell>
                  <TableCell>
                    <LinearProgress
                      variant="determinate"
                      value={normalizedImportance}
                      sx={{
                        height: 8,
                        borderRadius: 4,
                        bgcolor: 'grey.200',
                        '& .MuiLinearProgress-bar': {
                          borderRadius: 4,
                          bgcolor: index < 3 ? 'primary.main' : 'grey.500'
                        }
                      }}
                    />
                  </TableCell>
                  <TableCell align="center">
                    {meanShap !== undefined && (
                      meanShap > 0 ? (
                        <Tooltip title="Higher values increase prediction">
                          <TrendingUpIcon color="success" fontSize="small" />
                        </Tooltip>
                      ) : (
                        <Tooltip title="Higher values decrease prediction">
                          <TrendingDownIcon color="error" fontSize="small" />
                        </Tooltip>
                      )
                    )}
                  </TableCell>
                </TableRow>
              );
            })}
          </TableBody>
        </Table>
      </TableContainer>

      {/* Legend */}
      <Box sx={{ mt: 2, display: 'flex', gap: 2, flexWrap: 'wrap' }}>
        <Typography variant="caption" color="text.secondary">Categories:</Typography>
        {['Technical', 'Statistical', 'Cross-sectional', 'Fundamental'].map((cat) => {
          const category = getFeatureCategory(cat.toLowerCase());
          return (
            <Chip
              key={cat}
              label={cat}
              size="small"
              color={category.color}
              variant="outlined"
            />
          );
        })}
      </Box>
    </Box>
  );
};

export default FeatureImportance;
