import React from 'react';
import {
  Box,
  Paper,
  Typography,
  Card,
  CardContent,
  Grid,
  Chip,
  LinearProgress,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Divider
} from '@mui/material';
import RuleIcon from '@mui/icons-material/Rule';
import TrendingUpIcon from '@mui/icons-material/TrendingUp';
import TrendingDownIcon from '@mui/icons-material/TrendingDown';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';

const PatternViewer = ({ patterns }) => {
  if (!patterns || patterns.length === 0) {
    return (
      <Paper sx={{ p: 3, textAlign: 'center' }}>
        <Typography color="text.secondary">
          No patterns discovered yet. Run the training pipeline to discover universal patterns.
        </Typography>
      </Paper>
    );
  }

  const getReturnColor = (avgReturn) => {
    if (avgReturn > 0.02) return 'success';
    if (avgReturn > 0) return 'info';
    if (avgReturn > -0.02) return 'warning';
    return 'error';
  };

  const formatPercent = (value) => {
    return `${(value * 100).toFixed(2)}%`;
  };

  return (
    <Box>
      <Typography variant="h6" gutterBottom>
        Discovered Universal Patterns ({patterns.length})
      </Typography>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
        These patterns have been identified as consistently predictive across multiple S&P 500 stocks.
      </Typography>

      <Grid container spacing={3}>
        {patterns.map((pattern, index) => (
          <Grid item xs={12} md={6} key={pattern.pattern_id || index}>
            <Card
              sx={{
                height: '100%',
                borderLeft: 4,
                borderColor: pattern.avg_return > 0 ? 'success.main' : 'error.main'
              }}
            >
              <CardContent>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', mb: 2 }}>
                  <Box>
                    <Typography variant="h6" component="div">
                      Pattern #{index + 1}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      {pattern.description}
                    </Typography>
                  </Box>
                  <Chip
                    icon={pattern.avg_return > 0 ? <TrendingUpIcon /> : <TrendingDownIcon />}
                    label={formatPercent(pattern.avg_return)}
                    color={getReturnColor(pattern.avg_return)}
                    size="small"
                  />
                </Box>

                <Grid container spacing={2} sx={{ mb: 2 }}>
                  <Grid item xs={4}>
                    <Typography variant="caption" color="text.secondary">Win Rate</Typography>
                    <Typography variant="body1" fontWeight="bold">
                      {formatPercent(pattern.win_rate)}
                    </Typography>
                  </Grid>
                  <Grid item xs={4}>
                    <Typography variant="caption" color="text.secondary">Samples</Typography>
                    <Typography variant="body1" fontWeight="bold">
                      {pattern.n_samples?.toLocaleString()}
                    </Typography>
                  </Grid>
                  <Grid item xs={4}>
                    <Typography variant="caption" color="text.secondary">Confidence</Typography>
                    <Typography variant="body1" fontWeight="bold">
                      {formatPercent(pattern.confidence)}
                    </Typography>
                  </Grid>
                </Grid>

                <Box sx={{ mb: 2 }}>
                  <Typography variant="caption" color="text.secondary">Confidence Level</Typography>
                  <LinearProgress
                    variant="determinate"
                    value={(pattern.confidence || 0) * 100}
                    sx={{ height: 6, borderRadius: 3 }}
                    color={pattern.confidence > 0.7 ? 'success' : pattern.confidence > 0.4 ? 'warning' : 'error'}
                  />
                </Box>

                <Divider sx={{ my: 2 }} />

                <Typography variant="subtitle2" sx={{ mb: 1, display: 'flex', alignItems: 'center', gap: 1 }}>
                  <RuleIcon fontSize="small" /> Rules
                </Typography>

                <List dense disablePadding>
                  {(pattern.rules || []).slice(0, 5).map((rule, ruleIndex) => (
                    <ListItem key={ruleIndex} disableGutters sx={{ py: 0.5 }}>
                      <ListItemIcon sx={{ minWidth: 28 }}>
                        <CheckCircleIcon fontSize="small" color="success" />
                      </ListItemIcon>
                      <ListItemText
                        primary={rule}
                        primaryTypographyProps={{
                          variant: 'body2',
                          sx: { fontFamily: 'monospace', fontSize: '0.75rem' }
                        }}
                      />
                    </ListItem>
                  ))}
                  {pattern.rules && pattern.rules.length > 5 && (
                    <ListItem disableGutters sx={{ py: 0.5 }}>
                      <ListItemText
                        primary={`... and ${pattern.rules.length - 5} more rules`}
                        primaryTypographyProps={{
                          variant: 'caption',
                          color: 'text.secondary'
                        }}
                      />
                    </ListItem>
                  )}
                </List>
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>
    </Box>
  );
};

export default PatternViewer;
