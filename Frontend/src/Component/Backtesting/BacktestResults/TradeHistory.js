import React, { useState } from 'react';
import Box from '@mui/material/Box';
import Table from '@mui/material/Table';
import TableBody from '@mui/material/TableBody';
import TableCell from '@mui/material/TableCell';
import TableContainer from '@mui/material/TableContainer';
import TableHead from '@mui/material/TableHead';
import TableRow from '@mui/material/TableRow';
import TableSortLabel from '@mui/material/TableSortLabel';
import Chip from '@mui/material/Chip';

function TradeHistory({ trades, theme }) {
  const isDark = theme === 'dark';
  const [orderBy, setOrderBy] = useState('entry_date');
  const [order, setOrder] = useState('desc');

  const handleSort = (property) => {
    const isAsc = orderBy === property && order === 'asc';
    setOrder(isAsc ? 'desc' : 'asc');
    setOrderBy(property);
  };

  const sortedTrades = React.useMemo(() => {
    if (!trades || trades.length === 0) return [];

    return [...trades].sort((a, b) => {
      let aValue = a[orderBy];
      let bValue = b[orderBy];

      if (orderBy === 'entry_date' || orderBy === 'exit_date') {
        aValue = new Date(aValue || 0).getTime();
        bValue = new Date(bValue || 0).getTime();
      }

      if (order === 'asc') {
        return aValue < bValue ? -1 : aValue > bValue ? 1 : 0;
      } else {
        return aValue > bValue ? -1 : aValue < bValue ? 1 : 0;
      }
    });
  }, [trades, orderBy, order]);

  const cellStyle = {
    color: isDark ? 'rgba(255,255,255,0.87)' : 'inherit',
    borderColor: isDark ? 'rgba(255,255,255,0.1)' : 'rgba(0,0,0,0.1)',
  };

  const headerStyle = {
    ...cellStyle,
    fontWeight: 600,
    backgroundColor: isDark ? 'rgba(255,255,255,0.05)' : 'rgba(0,0,0,0.02)',
  };

  if (!trades || trades.length === 0) {
    return (
      <Box
        sx={{
          textAlign: 'center',
          py: 4,
          color: isDark ? 'rgba(255,255,255,0.5)' : 'rgba(0,0,0,0.5)',
        }}
      >
        No trades executed during this backtest
      </Box>
    );
  }

  return (
    <TableContainer sx={{ maxHeight: 400 }}>
      <Table stickyHeader size="small">
        <TableHead>
          <TableRow>
            <TableCell sx={headerStyle}>#</TableCell>
            <TableCell sx={headerStyle}>
              <TableSortLabel
                active={orderBy === 'type'}
                direction={orderBy === 'type' ? order : 'asc'}
                onClick={() => handleSort('type')}
              >
                Type
              </TableSortLabel>
            </TableCell>
            <TableCell sx={headerStyle}>
              <TableSortLabel
                active={orderBy === 'entry_date'}
                direction={orderBy === 'entry_date' ? order : 'asc'}
                onClick={() => handleSort('entry_date')}
              >
                Entry Date
              </TableSortLabel>
            </TableCell>
            <TableCell sx={headerStyle} align="right">Entry Price</TableCell>
            <TableCell sx={headerStyle}>
              <TableSortLabel
                active={orderBy === 'exit_date'}
                direction={orderBy === 'exit_date' ? order : 'asc'}
                onClick={() => handleSort('exit_date')}
              >
                Exit Date
              </TableSortLabel>
            </TableCell>
            <TableCell sx={headerStyle} align="right">Exit Price</TableCell>
            <TableCell sx={headerStyle} align="right">Size</TableCell>
            <TableCell sx={headerStyle}>
              <TableSortLabel
                active={orderBy === 'pnl'}
                direction={orderBy === 'pnl' ? order : 'asc'}
                onClick={() => handleSort('pnl')}
              >
                P&L
              </TableSortLabel>
            </TableCell>
            <TableCell sx={headerStyle}>
              <TableSortLabel
                active={orderBy === 'pnl_percent'}
                direction={orderBy === 'pnl_percent' ? order : 'asc'}
                onClick={() => handleSort('pnl_percent')}
              >
                P&L %
              </TableSortLabel>
            </TableCell>
            <TableCell sx={headerStyle}>Exit Reason</TableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {sortedTrades.map((trade, index) => {
            const isProfitable = trade.pnl >= 0;
            const pnlColor = isProfitable ? '#4caf50' : '#f44336';

            return (
              <TableRow
                key={trade.id || index}
                hover
                sx={{
                  '&:hover': {
                    backgroundColor: isDark ? 'rgba(255,255,255,0.05)' : 'rgba(0,0,0,0.02)',
                  },
                }}
              >
                <TableCell sx={cellStyle}>{trade.id || index + 1}</TableCell>
                <TableCell sx={cellStyle}>
                  <Chip
                    label={trade.type?.toUpperCase() || 'LONG'}
                    size="small"
                    sx={{
                      backgroundColor: trade.type === 'short' ? 'rgba(244,67,54,0.1)' : 'rgba(76,175,80,0.1)',
                      color: trade.type === 'short' ? '#f44336' : '#4caf50',
                      fontWeight: 500,
                      fontSize: '11px',
                    }}
                  />
                </TableCell>
                <TableCell sx={cellStyle}>
                  {trade.entry_date ? new Date(trade.entry_date).toLocaleDateString() : '-'}
                </TableCell>
                <TableCell sx={cellStyle} align="right">
                  ${trade.entry_price?.toFixed(2) || '-'}
                </TableCell>
                <TableCell sx={cellStyle}>
                  {trade.exit_date ? new Date(trade.exit_date).toLocaleDateString() : '-'}
                </TableCell>
                <TableCell sx={cellStyle} align="right">
                  ${trade.exit_price?.toFixed(2) || '-'}
                </TableCell>
                <TableCell sx={cellStyle} align="right">
                  {trade.size?.toFixed(0) || '-'}
                </TableCell>
                <TableCell sx={{ ...cellStyle, color: pnlColor, fontWeight: 500 }}>
                  {isProfitable ? '+' : ''}{trade.pnl?.toFixed(2) || '0.00'}
                </TableCell>
                <TableCell sx={{ ...cellStyle, color: pnlColor }}>
                  {isProfitable ? '+' : ''}{trade.pnl_percent?.toFixed(2) || '0.00'}%
                </TableCell>
                <TableCell sx={cellStyle}>
                  <Chip
                    label={trade.exit_reason?.replace('_', ' ').toUpperCase() || 'SIGNAL'}
                    size="small"
                    variant="outlined"
                    sx={{
                      fontSize: '10px',
                      height: '22px',
                      borderColor: isDark ? 'rgba(255,255,255,0.2)' : 'rgba(0,0,0,0.2)',
                      color: isDark ? 'rgba(255,255,255,0.7)' : 'inherit',
                    }}
                  />
                </TableCell>
              </TableRow>
            );
          })}
        </TableBody>
      </Table>
    </TableContainer>
  );
}

export default TradeHistory;
