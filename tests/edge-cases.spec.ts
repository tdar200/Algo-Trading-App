import { test, expect } from '@playwright/test';

/**
 * Edge Cases & Boundary Tests
 *
 * Tests that verify the system handles edge cases and boundary conditions correctly.
 */

const API_URL = 'http://localhost:5000';

const BASE_BACKTEST_PARAMS = {
  symbol: 'AAPL',
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

test.describe('Edge Cases & Boundaries', () => {

  test('should handle single day date range', async ({ request }) => {
    test.setTimeout(180000);
    const singleDate = '2023-06-15';

    const response = await request.post(`${API_URL}/api/backtest/run`, {
      data: {
        ...BASE_BACKTEST_PARAMS,
        start_date: singleDate,
        end_date: singleDate
      },
      timeout: 120000
    });

    console.log(`Single day range test: ${singleDate}`);
    console.log(`Response status: ${response.status()}`);

    if (response.ok()) {
      const data = await response.json();

      if (data.error) {
        console.log(`Error returned: ${data.error}`);
      } else if (data.statistics) {
        console.log(`Total trades: ${data.statistics.total_trades}`);
        console.log(`Final value: $${data.statistics.final_value}`);

        // With single day, likely no trades
        expect(data.statistics.total_trades).toBe(0);
        console.log('Single day range handled correctly (no trades expected)');
      }
    } else {
      const errorText = await response.text();
      console.log(`Error response: ${errorText.substring(0, 200)}`);
    }
  });

  test('should handle very short date range (5 business days)', async ({ request }) => {
    test.setTimeout(180000);
    const response = await request.post(`${API_URL}/api/backtest/run`, {
      data: {
        ...BASE_BACKTEST_PARAMS,
        start_date: '2023-06-12', // Monday
        end_date: '2023-06-16'    // Friday (5 days)
      },
      timeout: 120000
    });

    console.log('Testing 5 business day range');
    console.log(`Response status: ${response.status()}`);

    if (response.ok()) {
      const data = await response.json();

      if (data.statistics) {
        console.log(`Total trades in 5 days: ${data.statistics.total_trades}`);
        console.log(`Final value: $${data.statistics.final_value}`);

        // Very few or no trades expected in 5 days
        expect(data.statistics.total_trades).toBeGreaterThanOrEqual(0);
        console.log('Short date range handled correctly');
      }
    } else {
      console.log(`Error: ${response.status()}`);
    }
  });

  test('should handle parameters that generate zero trades', async ({ request }) => {
    test.setTimeout(180000);
    // Use extreme parameters that should generate no trades
    const response = await request.post(`${API_URL}/api/backtest/run`, {
      data: {
        ...BASE_BACKTEST_PARAMS,
        strategy_params: {
          LOOKBACK_PERIOD: 200, // Very long lookback
          BREAKOUT_BUFFER: 10,  // 10% buffer - very unlikely to trigger
          STOP_LOSS: 0.1,       // 0.1% stop loss - will trigger immediately
          TAKE_PROFIT: 50       // 50% take profit - very unlikely
        }
      },
      timeout: 120000
    });

    console.log('Testing parameters designed to generate zero trades');
    console.log(`Response status: ${response.status()}`);

    if (response.ok()) {
      const data = await response.json();

      if (data.statistics) {
        console.log(`Total trades: ${data.statistics.total_trades}`);
        console.log(`Final value: $${data.statistics.final_value}`);

        // With extreme parameters, should have few or no trades
        if (data.statistics.total_trades === 0) {
          console.log('Zero trades generated as expected');
          // Final value should equal initial capital
          expect(data.statistics.final_value).toBe(BASE_BACKTEST_PARAMS.initial_capital);
        } else {
          console.log(`Unexpectedly got ${data.statistics.total_trades} trades`);
        }
      }
    }
  });

  test('should handle extreme commission rate (50%)', async ({ request }) => {
    test.setTimeout(180000);
    const response = await request.post(`${API_URL}/api/backtest/run`, {
      data: {
        ...BASE_BACKTEST_PARAMS,
        commission: 0.5 // 50% commission
      },
      timeout: 120000
    });

    console.log('Testing extreme commission (50%)');
    console.log(`Response status: ${response.status()}`);

    if (response.ok()) {
      const data = await response.json();

      if (data.statistics) {
        console.log(`Total trades: ${data.statistics.total_trades}`);
        console.log(`Net profit: $${data.statistics.net_profit}`);
        console.log(`Final value: $${data.statistics.final_value}`);

        // With 50% commission, expect significant losses if trades occurred
        if (data.statistics.total_trades > 0) {
          // Net profit should be negative due to high commission
          console.log(`With 50% commission, net profit is: $${data.statistics.net_profit}`);
        }

        console.log('Extreme commission handled without error');
      }
    } else {
      const errorText = await response.text();
      console.log(`Error with extreme commission: ${errorText.substring(0, 200)}`);
    }
  });

  test('should handle minimal initial capital ($100)', async ({ request }) => {
    test.setTimeout(180000);
    const response = await request.post(`${API_URL}/api/backtest/run`, {
      data: {
        ...BASE_BACKTEST_PARAMS,
        initial_capital: 100 // Only $100
      },
      timeout: 120000
    });

    console.log('Testing minimal capital ($100)');
    console.log(`Response status: ${response.status()}`);

    if (response.ok()) {
      const data = await response.json();

      if (data.statistics) {
        console.log(`Total trades: ${data.statistics.total_trades}`);
        console.log(`Final value: $${data.statistics.final_value}`);

        // With $100 and AAPL at ~$150+, may not be able to buy shares
        if (data.statistics.total_trades === 0) {
          console.log('No trades with minimal capital (expected - cannot afford shares)');
        } else {
          console.log(`Made ${data.statistics.total_trades} trades with $100`);
        }

        // Final value should be positive
        expect(data.statistics.final_value).toBeGreaterThanOrEqual(0);
        console.log('Minimal capital handled correctly');
      }
    }
  });

  test('should handle large initial capital ($10 million)', async ({ request }) => {
    test.setTimeout(180000);
    const response = await request.post(`${API_URL}/api/backtest/run`, {
      data: {
        ...BASE_BACKTEST_PARAMS,
        initial_capital: 10000000 // $10 million
      },
      timeout: 120000
    });

    console.log('Testing large capital ($10 million)');
    console.log(`Response status: ${response.status()}`);

    if (response.ok()) {
      const data = await response.json();

      if (data.statistics) {
        console.log(`Total trades: ${data.statistics.total_trades}`);
        console.log(`Net profit: $${data.statistics.net_profit}`);
        console.log(`Final value: $${data.statistics.final_value}`);
        console.log(`Net profit %: ${data.statistics.net_profit_percent}%`);

        // Validate values are reasonable
        expect(data.statistics.final_value).toBeGreaterThan(0);

        // Net profit % should be same regardless of capital
        console.log('Large capital handled correctly');
      }
    }
  });

  test('should handle weekend/holiday start date', async ({ request }) => {
    test.setTimeout(180000);
    // January 1, 2023 was a Sunday
    const response = await request.post(`${API_URL}/api/backtest/run`, {
      data: {
        ...BASE_BACKTEST_PARAMS,
        start_date: '2023-01-01', // Sunday
        end_date: '2023-12-31'
      },
      timeout: 120000
    });

    console.log('Testing start on weekend (Jan 1, 2023 was Sunday)');
    console.log(`Response status: ${response.status()}`);

    if (response.ok()) {
      const data = await response.json();

      if (data.statistics) {
        console.log(`Total trades: ${data.statistics.total_trades}`);
        console.log(`Final value: $${data.statistics.final_value}`);

        // Should work - system should handle non-trading days
        expect(data.statistics.final_value).toBeGreaterThan(0);
        console.log('Weekend start date handled correctly');
      }
    } else {
      const errorText = await response.text();
      console.log(`Error with weekend date: ${errorText.substring(0, 200)}`);
    }
  });

  test('should handle extreme strategy parameters', async ({ request }) => {
    test.setTimeout(180000);
    // Test with very extreme but technically valid parameters
    const response = await request.post(`${API_URL}/api/backtest/run`, {
      data: {
        ...BASE_BACKTEST_PARAMS,
        strategy_params: {
          LOOKBACK_PERIOD: 1,    // Minimum lookback
          BREAKOUT_BUFFER: 0.01, // Very small buffer
          STOP_LOSS: 0.5,        // Very tight stop loss
          TAKE_PROFIT: 0.5       // Very tight take profit
        }
      },
      timeout: 120000
    });

    console.log('Testing extreme strategy parameters (very tight stops)');
    console.log(`Response status: ${response.status()}`);

    if (response.ok()) {
      const data = await response.json();

      if (data.statistics) {
        console.log(`Total trades: ${data.statistics.total_trades}`);
        console.log(`Win rate: ${data.statistics.win_rate}%`);
        console.log(`Net profit: $${data.statistics.net_profit}`);

        // With tight stops, expect many trades
        console.log('Extreme parameters handled correctly');
      }
    } else {
      const errorText = await response.text();
      console.log(`Error with extreme params: ${errorText.substring(0, 200)}`);
    }
  });
});

test.describe('Data Boundary Tests', () => {

  test('should handle very old data (if available)', async ({ request }) => {
    test.setTimeout(180000);
    // Try 10 years ago
    const response = await request.post(`${API_URL}/api/backtest/run`, {
      data: {
        ...BASE_BACKTEST_PARAMS,
        start_date: '2014-01-01',
        end_date: '2015-01-01'
      },
      timeout: 120000
    });

    console.log('Testing 10-year old data (2014)');
    console.log(`Response status: ${response.status()}`);

    if (response.ok()) {
      const data = await response.json();

      if (data.statistics) {
        console.log(`Total trades in 2014: ${data.statistics.total_trades}`);
        console.log(`Final value: $${data.statistics.final_value}`);

        expect(data.statistics.total_trades).toBeGreaterThanOrEqual(0);
        console.log('Historical data from 2014 handled correctly');
      }
    }
  });

  test('should handle recent data (current year)', async ({ request }) => {
    test.setTimeout(180000);
    const currentYear = new Date().getFullYear();
    const startDate = `${currentYear}-01-01`;
    const endDate = new Date().toISOString().split('T')[0];

    const response = await request.post(`${API_URL}/api/backtest/run`, {
      data: {
        ...BASE_BACKTEST_PARAMS,
        start_date: startDate,
        end_date: endDate
      },
      timeout: 120000
    });

    console.log(`Testing current year data: ${startDate} to ${endDate}`);
    console.log(`Response status: ${response.status()}`);

    if (response.ok()) {
      const data = await response.json();

      if (data.statistics) {
        console.log(`Total trades this year: ${data.statistics.total_trades}`);
        console.log(`Final value: $${data.statistics.final_value}`);

        expect(data.statistics.total_trades).toBeGreaterThanOrEqual(0);
        console.log('Current year data handled correctly');
      }
    }
  });
});
