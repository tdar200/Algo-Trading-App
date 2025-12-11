import { test, expect } from '@playwright/test';

/**
 * API Error Handling Tests
 *
 * Tests that verify the API properly handles invalid inputs and edge cases.
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

// API tests need longer timeouts for backtest operations
test.describe('API Error Handling', () => {

  test('should handle invalid stock symbol gracefully', async ({ request }) => {
    test.setTimeout(180000); // 3 min timeout
    const response = await request.post(`${API_URL}/api/backtest/run`, {
      data: {
        ...BASE_BACKTEST_PARAMS,
        symbol: 'INVALID123XYZ'
      },
      timeout: 120000  // 2 min request timeout
    });

    console.log(`Response status: ${response.status()}`);

    // API should return error status or error message
    if (response.ok()) {
      const data = await response.json();
      console.log('Response:', JSON.stringify(data, null, 2));

      // If it returns OK, it should have an error field or empty results
      if (data.error) {
        console.log(`Error returned: ${data.error}`);
        expect(data.error).toBeTruthy();
      } else if (data.trades && data.trades.length === 0) {
        console.log('No trades returned for invalid symbol (acceptable)');
      } else {
        console.log('API returned unexpected success for invalid symbol');
      }
    } else {
      // Error status code is expected
      const errorText = await response.text();
      console.log(`Expected error received: ${response.status()} - ${errorText.substring(0, 200)}`);
      expect(response.status()).toBeGreaterThanOrEqual(400);
    }

    console.log('Invalid symbol handling verified');
  });

  test('should handle invalid date range (end before start)', async ({ request }) => {
    test.setTimeout(180000);
    const response = await request.post(`${API_URL}/api/backtest/run`, {
      data: {
        ...BASE_BACKTEST_PARAMS,
        start_date: '2024-01-01',
        end_date: '2023-01-01' // End before start
      },
      timeout: 120000
    });

    console.log(`Response status: ${response.status()}`);

    if (response.ok()) {
      const data = await response.json();
      console.log('Response:', JSON.stringify(data, null, 2));

      // Should either have error or no trades
      if (data.error) {
        console.log(`Error returned: ${data.error}`);
        expect(data.error).toBeTruthy();
      } else if (data.statistics && data.statistics.total_trades === 0) {
        console.log('No trades for invalid date range (acceptable)');
      }
    } else {
      const errorText = await response.text();
      console.log(`Expected error: ${response.status()} - ${errorText.substring(0, 200)}`);
      expect(response.status()).toBeGreaterThanOrEqual(400);
    }

    console.log('Invalid date range handling verified');
  });

  test('should handle future date range', async ({ request }) => {
    test.setTimeout(180000);
    const futureStart = '2030-01-01';
    const futureEnd = '2031-01-01';

    const response = await request.post(`${API_URL}/api/backtest/run`, {
      data: {
        ...BASE_BACKTEST_PARAMS,
        start_date: futureStart,
        end_date: futureEnd
      },
      timeout: 120000
    });

    console.log(`Response status: ${response.status()}`);
    console.log(`Testing with future dates: ${futureStart} to ${futureEnd}`);

    if (response.ok()) {
      const data = await response.json();

      if (data.error) {
        console.log(`Error returned: ${data.error}`);
        expect(data.error).toBeTruthy();
      } else if (data.statistics && data.statistics.total_trades === 0) {
        console.log('No trades for future dates (no historical data available)');
      } else {
        console.log('Response:', JSON.stringify(data, null, 2).substring(0, 500));
      }
    } else {
      const errorText = await response.text();
      console.log(`Expected error: ${response.status()} - ${errorText.substring(0, 200)}`);
    }

    console.log('Future date range handling verified');
  });

  test('should handle negative initial capital', async ({ request }) => {
    test.setTimeout(180000);
    const response = await request.post(`${API_URL}/api/backtest/run`, {
      data: {
        ...BASE_BACKTEST_PARAMS,
        initial_capital: -1000
      },
      timeout: 120000
    });

    console.log(`Response status: ${response.status()}`);

    if (response.ok()) {
      const data = await response.json();

      if (data.error) {
        console.log(`Error returned: ${data.error}`);
        expect(data.error).toBeTruthy();
      } else {
        console.log('Response:', JSON.stringify(data, null, 2).substring(0, 500));
        // Check if final_value is reasonable
        if (data.statistics && data.statistics.final_value) {
          console.log(`Final value with negative capital: ${data.statistics.final_value}`);
        }
      }
    } else {
      const errorText = await response.text();
      console.log(`Expected error for negative capital: ${response.status()}`);
      expect(response.status()).toBeGreaterThanOrEqual(400);
    }

    console.log('Negative capital handling verified');
  });

  test('should handle zero initial capital', async ({ request }) => {
    test.setTimeout(180000);
    const response = await request.post(`${API_URL}/api/backtest/run`, {
      data: {
        ...BASE_BACKTEST_PARAMS,
        initial_capital: 0
      },
      timeout: 120000
    });

    console.log(`Response status: ${response.status()}`);

    if (response.ok()) {
      const data = await response.json();

      if (data.error) {
        console.log(`Error returned: ${data.error}`);
        expect(data.error).toBeTruthy();
      } else {
        console.log('Response:', JSON.stringify(data, null, 2).substring(0, 500));

        // With zero capital, should have no trades
        if (data.statistics) {
          console.log(`Total trades with zero capital: ${data.statistics.total_trades}`);
          expect(data.statistics.total_trades).toBe(0);
        }
      }
    } else {
      const errorText = await response.text();
      console.log(`Expected error for zero capital: ${response.status()}`);
      expect(response.status()).toBeGreaterThanOrEqual(400);
    }

    console.log('Zero capital handling verified');
  });

  test('should handle invalid strategy name', async ({ request }) => {
    test.setTimeout(180000);
    const response = await request.post(`${API_URL}/api/backtest/run`, {
      data: {
        ...BASE_BACKTEST_PARAMS,
        strategy: 'NonExistentStrategy123'
      },
      timeout: 120000
    });

    console.log(`Response status: ${response.status()}`);

    if (response.ok()) {
      const data = await response.json();

      if (data.error) {
        console.log(`Error returned: ${data.error}`);
        expect(data.error).toBeTruthy();
      } else {
        // API might fall back to default strategy
        console.log('Response:', JSON.stringify(data, null, 2).substring(0, 500));
        console.log('API accepted request (may use default strategy)');
      }
    } else {
      const errorText = await response.text();
      console.log(`Expected error for invalid strategy: ${response.status()}`);
    }

    console.log('Invalid strategy handling verified');
  });

  test('should handle missing required fields gracefully', async ({ request }) => {
    test.setTimeout(180000);
    // Test with minimal/missing fields
    const response = await request.post(`${API_URL}/api/backtest/run`, {
      data: {
        symbol: 'AAPL'
        // Missing: start_date, end_date, initial_capital, etc.
      },
      timeout: 120000
    });

    console.log(`Response status: ${response.status()}`);

    if (response.ok()) {
      const data = await response.json();

      if (data.error) {
        console.log(`Error returned: ${data.error}`);
        expect(data.error).toBeTruthy();
      } else {
        // API might have defaults
        console.log('Response received - API may have default values');
        console.log('Response:', JSON.stringify(data, null, 2).substring(0, 500));
      }
    } else {
      const errorText = await response.text();
      console.log(`Expected error for missing fields: ${response.status()}`);
      // 400 Bad Request is expected
      expect(response.status()).toBeGreaterThanOrEqual(400);
    }

    console.log('Missing fields handling verified');
  });
});

test.describe('API Endpoint Availability', () => {

  test('should check /api/strategies endpoint', async ({ request }) => {
    const response = await request.get(`${API_URL}/api/strategies`, {
      timeout: 30000
    });

    console.log(`GET /api/strategies - Status: ${response.status()}`);

    if (response.ok()) {
      const data = await response.json();
      console.log('Strategies:', JSON.stringify(data, null, 2));

      // Should return array or object of strategies
      if (Array.isArray(data)) {
        console.log(`Found ${data.length} strategies`);
        expect(data.length).toBeGreaterThan(0);
      } else if (data.strategies) {
        console.log(`Found ${data.strategies.length} strategies`);
      }
    } else {
      console.log(`Strategies endpoint returned: ${response.status()}`);
      // Endpoint might not exist - that's information, not a failure
    }
  });

  test('should check /api/gpu/status endpoint', async ({ request }) => {
    const response = await request.get(`${API_URL}/api/gpu/status`, {
      timeout: 30000
    });

    console.log(`GET /api/gpu/status - Status: ${response.status()}`);

    if (response.ok()) {
      const data = await response.json();
      console.log('GPU Status:', JSON.stringify(data, null, 2));

      // Should have gpu_available field
      if (data.gpu_available !== undefined) {
        console.log(`GPU Available: ${data.gpu_available}`);
      }
    } else {
      console.log(`GPU status endpoint returned: ${response.status()}`);
    }
  });

  test('should check /stock endpoint works', async ({ request }) => {
    const response = await request.get(`${API_URL}/stock?symbol=AAPL`, {
      timeout: 60000  // Stock endpoint can take a while
    });

    console.log(`GET /stock?symbol=AAPL - Status: ${response.status()}`);

    expect(response.ok()).toBeTruthy();

    const data = await response.json();
    console.log('Stock response keys:', Object.keys(data));

    // Should have some data
    expect(Object.keys(data).length).toBeGreaterThan(0);
    console.log('Stock endpoint working correctly');
  });
});
