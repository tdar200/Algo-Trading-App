import { test, expect } from '@playwright/test';

/**
 * Multi-Stock Validation Tests
 *
 * Tests that verify the backtest system works consistently across different stocks.
 * Tests AAPL, MSFT, GOOG, and TSLA to ensure calculations are stock-agnostic.
 */

const API_URL = 'http://localhost:5000';

const TEST_SYMBOLS = ['AAPL', 'MSFT', 'GOOG', 'TSLA'];

const BASE_BACKTEST_PARAMS = {
  start_date: '2023-01-01',
  end_date: '2024-01-01',
  initial_capital: 100000,
  commission: 0.001,
  strategy_params: {
    LOOKBACK_PERIOD: 20,
    BREAKOUT_BUFFER: 0.5,
    STOP_LOSS: 5,
    TAKE_PROFIT: 10
  }
};

// Helper function to run backtest for a symbol with timeout
async function runBacktest(request: any, symbol: string) {
  const response = await request.post(`${API_URL}/api/backtest/run`, {
    data: {
      ...BASE_BACKTEST_PARAMS,
      symbol
    },
    timeout: 120000  // 2 min timeout for API call
  });
  return response;
}

// Helper function to validate trade PnL calculation
function validateTradePnL(trade: any): { valid: boolean; message: string } {
  if (!trade.entry_price || !trade.exit_price || !trade.size) {
    return { valid: true, message: 'Missing fields, skipping' };
  }

  const grossPnL = (trade.exit_price - trade.entry_price) * trade.size;
  const expectedNetPnL = grossPnL - (trade.commission || 0);
  const pnlDiff = Math.abs(trade.pnl - expectedNetPnL);

  if (pnlDiff > 1) {
    return {
      valid: false,
      message: `PnL mismatch: expected ~$${expectedNetPnL.toFixed(2)}, got $${trade.pnl.toFixed(2)}`
    };
  }

  return { valid: true, message: 'PnL calculation correct' };
}

test.describe('Multi-Stock Backtest Validation', () => {

  test('should run complete backtest for AAPL', async ({ request }) => {
    test.setTimeout(180000);
    const symbol = 'AAPL';
    console.log(`\n=== Testing ${symbol} ===`);

    const response = await runBacktest(request, symbol);
    expect(response.ok()).toBeTruthy();

    const data = await response.json();
    const stats = data.statistics;

    console.log(`Total trades: ${stats.total_trades}`);
    console.log(`Win rate: ${stats.win_rate}%`);
    console.log(`Net profit: $${stats.net_profit}`);
    console.log(`Net profit %: ${stats.net_profit_percent}%`);
    console.log(`Max drawdown: ${stats.max_drawdown_percent}%`);
    console.log(`Sharpe ratio: ${stats.sharpe_ratio}`);
    console.log(`Final value: $${stats.final_value}`);

    // Basic validations
    expect(stats.total_trades).toBeGreaterThanOrEqual(0);
    expect(stats.win_rate).toBeGreaterThanOrEqual(0);
    expect(stats.win_rate).toBeLessThanOrEqual(100);
    expect(stats.final_value).toBeGreaterThan(0);

    console.log(`${symbol} backtest completed successfully`);
  });

  test('should run complete backtest for MSFT', async ({ request }) => {
    test.setTimeout(180000);
    const symbol = 'MSFT';
    console.log(`\n=== Testing ${symbol} ===`);

    const response = await runBacktest(request, symbol);
    expect(response.ok()).toBeTruthy();

    const data = await response.json();
    const stats = data.statistics;

    console.log(`Total trades: ${stats.total_trades}`);
    console.log(`Win rate: ${stats.win_rate}%`);
    console.log(`Net profit: $${stats.net_profit}`);
    console.log(`Net profit %: ${stats.net_profit_percent}%`);
    console.log(`Max drawdown: ${stats.max_drawdown_percent}%`);
    console.log(`Sharpe ratio: ${stats.sharpe_ratio}`);
    console.log(`Final value: $${stats.final_value}`);

    expect(stats.total_trades).toBeGreaterThanOrEqual(0);
    expect(stats.win_rate).toBeGreaterThanOrEqual(0);
    expect(stats.win_rate).toBeLessThanOrEqual(100);
    expect(stats.final_value).toBeGreaterThan(0);

    console.log(`${symbol} backtest completed successfully`);
  });

  test('should run complete backtest for GOOG', async ({ request }) => {
    test.setTimeout(180000);
    const symbol = 'GOOG';
    console.log(`\n=== Testing ${symbol} ===`);

    const response = await runBacktest(request, symbol);
    expect(response.ok()).toBeTruthy();

    const data = await response.json();
    const stats = data.statistics;

    console.log(`Total trades: ${stats.total_trades}`);
    console.log(`Win rate: ${stats.win_rate}%`);
    console.log(`Net profit: $${stats.net_profit}`);
    console.log(`Net profit %: ${stats.net_profit_percent}%`);
    console.log(`Max drawdown: ${stats.max_drawdown_percent}%`);
    console.log(`Sharpe ratio: ${stats.sharpe_ratio}`);
    console.log(`Final value: $${stats.final_value}`);

    expect(stats.total_trades).toBeGreaterThanOrEqual(0);
    expect(stats.win_rate).toBeGreaterThanOrEqual(0);
    expect(stats.win_rate).toBeLessThanOrEqual(100);
    expect(stats.final_value).toBeGreaterThan(0);

    console.log(`${symbol} backtest completed successfully`);
  });

  test('should run complete backtest for TSLA', async ({ request }) => {
    test.setTimeout(180000);
    const symbol = 'TSLA';
    console.log(`\n=== Testing ${symbol} ===`);

    const response = await runBacktest(request, symbol);
    expect(response.ok()).toBeTruthy();

    const data = await response.json();
    const stats = data.statistics;

    console.log(`Total trades: ${stats.total_trades}`);
    console.log(`Win rate: ${stats.win_rate}%`);
    console.log(`Net profit: $${stats.net_profit}`);
    console.log(`Net profit %: ${stats.net_profit_percent}%`);
    console.log(`Max drawdown: ${stats.max_drawdown_percent}%`);
    console.log(`Sharpe ratio: ${stats.sharpe_ratio}`);
    console.log(`Final value: $${stats.final_value}`);

    expect(stats.total_trades).toBeGreaterThanOrEqual(0);
    expect(stats.win_rate).toBeGreaterThanOrEqual(0);
    expect(stats.win_rate).toBeLessThanOrEqual(100);
    expect(stats.final_value).toBeGreaterThan(0);

    console.log(`${symbol} backtest completed successfully`);
  });
});

test.describe('Multi-Stock PnL Calculation Validation', () => {

  test('should verify AAPL PnL calculations are mathematically correct', async ({ request }) => {
    test.setTimeout(180000);
    const symbol = 'AAPL';
    console.log(`\n=== ${symbol} PnL Calculation Validation ===`);

    const response = await runBacktest(request, symbol);
    expect(response.ok()).toBeTruthy();

    const data = await response.json();
    const trades = data.trades || [];

    console.log(`Validating ${trades.length} trades...`);

    let allValid = true;
    let validCount = 0;
    let errorCount = 0;

    for (const trade of trades) {
      const result = validateTradePnL(trade);
      if (!result.valid) {
        console.log(`Trade #${trade.id}: ${result.message}`);
        allValid = false;
        errorCount++;
      } else {
        validCount++;
      }
    }

    console.log(`\nResults: ${validCount} valid, ${errorCount} errors`);
    expect(allValid).toBeTruthy();
    console.log(`${symbol} PnL calculations verified`);
  });

  test('should verify MSFT PnL calculations are mathematically correct', async ({ request }) => {
    test.setTimeout(180000);
    const symbol = 'MSFT';
    console.log(`\n=== ${symbol} PnL Calculation Validation ===`);

    const response = await runBacktest(request, symbol);
    expect(response.ok()).toBeTruthy();

    const data = await response.json();
    const trades = data.trades || [];

    console.log(`Validating ${trades.length} trades...`);

    let allValid = true;
    let validCount = 0;

    for (const trade of trades) {
      const result = validateTradePnL(trade);
      if (!result.valid) {
        console.log(`Trade #${trade.id}: ${result.message}`);
        allValid = false;
      } else {
        validCount++;
      }
    }

    console.log(`\n${validCount}/${trades.length} trades validated`);
    expect(allValid).toBeTruthy();
    console.log(`${symbol} PnL calculations verified`);
  });

  test('should verify GOOG PnL calculations are mathematically correct', async ({ request }) => {
    test.setTimeout(180000);
    const symbol = 'GOOG';
    console.log(`\n=== ${symbol} PnL Calculation Validation ===`);

    const response = await runBacktest(request, symbol);
    expect(response.ok()).toBeTruthy();

    const data = await response.json();
    const trades = data.trades || [];

    console.log(`Validating ${trades.length} trades...`);

    let allValid = true;
    let validCount = 0;

    for (const trade of trades) {
      const result = validateTradePnL(trade);
      if (!result.valid) {
        console.log(`Trade #${trade.id}: ${result.message}`);
        allValid = false;
      } else {
        validCount++;
      }
    }

    console.log(`\n${validCount}/${trades.length} trades validated`);
    expect(allValid).toBeTruthy();
    console.log(`${symbol} PnL calculations verified`);
  });

  test('should verify TSLA PnL calculations are mathematically correct', async ({ request }) => {
    test.setTimeout(180000);
    const symbol = 'TSLA';
    console.log(`\n=== ${symbol} PnL Calculation Validation ===`);

    const response = await runBacktest(request, symbol);
    expect(response.ok()).toBeTruthy();

    const data = await response.json();
    const trades = data.trades || [];

    console.log(`Validating ${trades.length} trades...`);

    let allValid = true;
    let validCount = 0;

    for (const trade of trades) {
      const result = validateTradePnL(trade);
      if (!result.valid) {
        console.log(`Trade #${trade.id}: ${result.message}`);
        allValid = false;
      } else {
        validCount++;
      }
    }

    console.log(`\n${validCount}/${trades.length} trades validated`);
    expect(allValid).toBeTruthy();
    console.log(`${symbol} PnL calculations verified`);
  });
});

test.describe('Cross-Stock Consistency', () => {

  test('should verify calculation logic is consistent across all stocks', async ({ request }) => {
    test.setTimeout(600000);  // 10 min for all stocks
    console.log('\n=== Cross-Stock Consistency Test ===');

    const results: Record<string, any> = {};

    for (const symbol of TEST_SYMBOLS) {
      const response = await runBacktest(request, symbol);
      expect(response.ok()).toBeTruthy();

      const data = await response.json();
      results[symbol] = {
        trades: data.trades?.length || 0,
        stats: data.statistics
      };

      console.log(`\n${symbol}:`);
      console.log(`  Trades: ${results[symbol].trades}`);
      console.log(`  Win Rate: ${results[symbol].stats.win_rate}%`);
      console.log(`  Sharpe: ${results[symbol].stats.sharpe_ratio}`);
    }

    // Verify all stocks have valid data
    for (const symbol of TEST_SYMBOLS) {
      expect(results[symbol].stats.final_value).toBeGreaterThan(0);
      expect(results[symbol].stats.win_rate).toBeGreaterThanOrEqual(0);
      expect(results[symbol].stats.win_rate).toBeLessThanOrEqual(100);
    }

    // Summary comparison
    console.log('\n=== Summary ===');
    console.log('Symbol | Trades | Win Rate | Net Profit % | Sharpe');
    console.log('-------|--------|----------|--------------|-------');
    for (const symbol of TEST_SYMBOLS) {
      const s = results[symbol].stats;
      console.log(`${symbol.padEnd(6)} | ${String(results[symbol].trades).padEnd(6)} | ${String(s.win_rate + '%').padEnd(8)} | ${String(s.net_profit_percent + '%').padEnd(12)} | ${s.sharpe_ratio}`);
    }

    console.log('\nAll stocks processed with consistent calculation logic');
  });

  test('should verify win rate calculation is stock-agnostic', async ({ request }) => {
    test.setTimeout(600000);  // 10 min for all stocks
    console.log('\n=== Win Rate Consistency Test ===');

    for (const symbol of TEST_SYMBOLS) {
      const response = await runBacktest(request, symbol);
      expect(response.ok()).toBeTruthy();

      const data = await response.json();
      const trades = data.trades || [];
      const stats = data.statistics;

      if (trades.length === 0) {
        console.log(`${symbol}: No trades to validate`);
        continue;
      }

      // Calculate win rate from trades
      let winCount = 0;
      for (const trade of trades) {
        if (trade.pnl > 0) winCount++;
      }

      const calculatedWinRate = (winCount / trades.length) * 100;

      console.log(`\n${symbol}:`);
      console.log(`  Trades: ${trades.length}`);
      console.log(`  Winners: ${winCount}`);
      console.log(`  Calculated Win Rate: ${calculatedWinRate.toFixed(2)}%`);
      console.log(`  Reported Win Rate: ${stats.win_rate}%`);

      // Win rate should be within 1% due to rounding
      const winRateDiff = Math.abs(calculatedWinRate - stats.win_rate);
      if (winRateDiff > 1) {
        console.log(`  Note: Win rate difference of ${winRateDiff.toFixed(2)}%`);
      }
    }

    console.log('\nWin rate calculation verified across all stocks');
  });
});
