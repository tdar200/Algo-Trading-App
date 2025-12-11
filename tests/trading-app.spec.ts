import { test, expect } from '@playwright/test';

test.describe('Trading App', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    // Wait for app to load
    await page.waitForLoadState('domcontentloaded');
    await page.waitForTimeout(2000); // Wait for React to render
  });

  test('should load the app and display stock input', async ({ page }) => {
    // Check that the stock input exists and has default value
    const stockInput = page.locator('input#stock-input');
    await expect(stockInput).toBeVisible({ timeout: 10000 });
    await expect(stockInput).toHaveValue('AMZN');
  });

  test('should display securities list with 5 stocks', async ({ page }) => {
    const stockList = ['AMZN', 'GOOG', 'IBM', 'TSLA', 'MSFT'];

    for (const stock of stockList) {
      await expect(page.getByRole('button', { name: stock })).toBeVisible({ timeout: 10000 });
    }
  });

  test('should display chart when data is loaded', async ({ page }) => {
    // Wait for chart canvas to appear (lightweight-charts uses canvas)
    await expect(page.locator('canvas').first()).toBeVisible({ timeout: 30000 });
  });

  test('should change stock when clicking on securities list', async ({ page }) => {
    // Wait for initial load
    await page.locator('canvas').first().waitFor({ timeout: 30000 });

    // Click on TSLA in the list
    await page.getByRole('button', { name: 'TSLA' }).click();

    // Check that input value changed
    const stockInput = page.locator('input#stock-input');
    await expect(stockInput).toHaveValue('TSLA', { timeout: 5000 });
  });

  test('should update chart when typing new stock symbol', async ({ page }) => {
    // Wait for initial chart to load
    await expect(page.locator('canvas').first()).toBeVisible({ timeout: 60000 });

    const stockInput = page.locator('input#stock-input');
    await stockInput.clear();
    await stockInput.fill('MSFT');

    // Wait for loading to complete and new chart to appear
    await page.waitForTimeout(5000);
    await expect(page.locator('canvas').first()).toBeVisible({ timeout: 60000 });
  });

  test('should convert stock symbol to uppercase', async ({ page }) => {
    const stockInput = page.locator('input#stock-input');
    await stockInput.clear();
    await stockInput.fill('aapl');

    // Should be converted to uppercase
    await expect(stockInput).toHaveValue('AAPL');
  });
});

test.describe('Chart Functionality', () => {
  test('should display candlestick chart with canvas', async ({ page }) => {
    await page.goto('/');

    // Wait for chart to load (lightweight-charts renders to canvas)
    const canvas = page.locator('canvas').first();
    await expect(canvas).toBeVisible({ timeout: 30000 });

    // Check canvas has rendered (has non-zero dimensions)
    const box = await canvas.boundingBox();
    expect(box?.width).toBeGreaterThan(0);
    expect(box?.height).toBeGreaterThan(0);
  });
});

test.describe('API Integration', () => {
  test('should fetch stock data from backend API', async ({ request }) => {
    // Direct API test with percentage-based parameters
    const response = await request.get('http://localhost:5000/stock?symbol=AMZN&first_retracement=5&second_retracement=5&level_range=0.001&touch_count=1');
    expect(response.ok()).toBeTruthy();

    const data = await response.json();
    expect(data).toHaveProperty('stock_data');
    expect(data).toHaveProperty('trades');
    expect(data).toHaveProperty('levels');
    expect(data).toHaveProperty('resistance_levels');
    expect(data).toHaveProperty('support_levels');
    expect(data).toHaveProperty('resistance_clusters');
    expect(data).toHaveProperty('support_clusters');

    // Check stock_data has entries
    expect(Object.keys(data.stock_data).length).toBeGreaterThan(0);
  });

  test('should return error for invalid symbol via API', async ({ request }) => {
    const response = await request.get('http://localhost:5000/stock?symbol=INVALID123');
    expect(response.status()).toBe(400);

    const data = await response.json();
    expect(data).toHaveProperty('error');
  });

  test('should return resistance levels as array', async ({ request }) => {
    const response = await request.get('http://localhost:5000/stock?symbol=AMZN');
    expect(response.ok()).toBeTruthy();

    const data = await response.json();
    expect(Array.isArray(data.resistance_levels)).toBe(true);
  });

  test('should return support levels as array', async ({ request }) => {
    const response = await request.get('http://localhost:5000/stock?symbol=AMZN');
    expect(response.ok()).toBeTruthy();

    const data = await response.json();
    expect(Array.isArray(data.support_levels)).toBe(true);
  });

  test('should return trades array', async ({ request }) => {
    const response = await request.get('http://localhost:5000/stock?symbol=AMZN');
    expect(response.ok()).toBeTruthy();

    const data = await response.json();
    expect(Array.isArray(data.trades)).toBe(true);
    // Trades should have type, price, and date if present
    if (data.trades.length > 0) {
      expect(data.trades[0]).toHaveProperty('type');
      expect(data.trades[0]).toHaveProperty('price');
      expect(data.trades[0]).toHaveProperty('date');
    }
  });
});

test.describe('Error Handling', () => {
  test('should display error for invalid stock symbol', async ({ page }) => {
    await page.goto('/');

    const stockInput = page.locator('input#stock-input');
    await stockInput.clear();
    await stockInput.fill('XXXXXX');

    // Wait for error alert to appear
    await expect(page.locator('[role="alert"]')).toBeVisible({ timeout: 15000 });
  });

  test('should show loading or chart after changing stock', async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('domcontentloaded');
    await page.waitForTimeout(2000); // Wait for React to render

    // Wait for initial chart
    await expect(page.locator('canvas').first()).toBeVisible({ timeout: 60000 });

    // Clear and type new symbol
    const stockInput = page.locator('input#stock-input');
    await stockInput.clear();
    await stockInput.fill('GOOG');

    // Wait for loading to complete and new chart to appear
    await page.waitForTimeout(5000);
    await expect(page.locator('canvas').first()).toBeVisible({ timeout: 60000 });
  });
});

test.describe('Retracement Parameters Tests', () => {
  test.describe('API Parameter Tests', () => {
    test('should accept first_retracement parameter', async ({ request }) => {
      const response = await request.get('http://localhost:5000/stock?symbol=AMZN&first_retracement=10');
      expect(response.ok()).toBeTruthy();

      const data = await response.json();
      expect(data).toHaveProperty('resistance_levels');
      expect(data).toHaveProperty('support_levels');
    });

    test('should accept second_retracement parameter', async ({ request }) => {
      const response = await request.get('http://localhost:5000/stock?symbol=AMZN&second_retracement=10');
      expect(response.ok()).toBeTruthy();

      const data = await response.json();
      expect(data).toHaveProperty('resistance_levels');
      expect(data).toHaveProperty('support_levels');
    });

    test('should accept level_range parameter', async ({ request }) => {
      const response = await request.get('http://localhost:5000/stock?symbol=AMZN&level_range=0.005');
      expect(response.ok()).toBeTruthy();

      const data = await response.json();
      expect(data).toHaveProperty('resistance_clusters');
      expect(data).toHaveProperty('support_clusters');
    });

    test('should accept touch_count parameter', async ({ request }) => {
      const response = await request.get('http://localhost:5000/stock?symbol=AMZN&touch_count=2');
      expect(response.ok()).toBeTruthy();

      const data = await response.json();
      expect(data).toHaveProperty('resistance_levels');
      expect(data).toHaveProperty('support_levels');
    });
  });

  test.describe('Retracement Behavior', () => {
    test('different first_retracement values should produce valid levels', async ({ request }) => {
      const response5 = await request.get('http://localhost:5000/stock?symbol=AMZN&first_retracement=5&second_retracement=5');
      const response20 = await request.get('http://localhost:5000/stock?symbol=AMZN&first_retracement=20&second_retracement=5');

      expect(response5.ok()).toBeTruthy();
      expect(response20.ok()).toBeTruthy();

      const data5 = await response5.json();
      const data20 = await response20.json();

      // Both should return valid array of levels
      expect(Array.isArray(data5.resistance_levels)).toBe(true);
      expect(Array.isArray(data5.support_levels)).toBe(true);
      expect(Array.isArray(data20.resistance_levels)).toBe(true);
      expect(Array.isArray(data20.support_levels)).toBe(true);

      console.log(`5% first retracement: ${data5.resistance_levels.length} resistance, ${data5.support_levels.length} support`);
      console.log(`20% first retracement: ${data20.resistance_levels.length} resistance, ${data20.support_levels.length} support`);
    });

    test('different second_retracement values should produce valid levels', async ({ request }) => {
      const response5 = await request.get('http://localhost:5000/stock?symbol=AMZN&first_retracement=5&second_retracement=5');
      const response20 = await request.get('http://localhost:5000/stock?symbol=AMZN&first_retracement=5&second_retracement=20');

      expect(response5.ok()).toBeTruthy();
      expect(response20.ok()).toBeTruthy();

      const data5 = await response5.json();
      const data20 = await response20.json();

      // Both should return valid array of levels
      expect(Array.isArray(data5.resistance_levels)).toBe(true);
      expect(Array.isArray(data5.support_levels)).toBe(true);
      expect(Array.isArray(data20.resistance_levels)).toBe(true);
      expect(Array.isArray(data20.support_levels)).toBe(true);
    });
  });

  test.describe('Touch Count Behavior', () => {
    test('higher touch_count should result in fewer or equal levels', async ({ request }) => {
      const response1 = await request.get('http://localhost:5000/stock?symbol=AMZN&touch_count=1');
      const response3 = await request.get('http://localhost:5000/stock?symbol=AMZN&touch_count=3');

      const data1 = await response1.json();
      const data3 = await response3.json();

      // Higher touch requirement should result in fewer or equal levels
      expect(data3.resistance_levels.length).toBeLessThanOrEqual(data1.resistance_levels.length);
      expect(data3.support_levels.length).toBeLessThanOrEqual(data1.support_levels.length);

      console.log(`touch_count=1: ${data1.resistance_levels.length} resistance, ${data1.support_levels.length} support`);
      console.log(`touch_count=3: ${data3.resistance_levels.length} resistance, ${data3.support_levels.length} support`);
    });
  });

  test.describe('Level Clustering', () => {
    test('should return cluster information', async ({ request }) => {
      const response = await request.get('http://localhost:5000/stock?symbol=AMZN&level_range=0.005');
      const data = await response.json();

      expect(data).toHaveProperty('resistance_clusters');
      expect(data).toHaveProperty('support_clusters');

      // Clusters should be objects
      expect(typeof data.resistance_clusters).toBe('object');
      expect(typeof data.support_clusters).toBe('object');

      console.log('Resistance clusters:', data.resistance_clusters);
      console.log('Support clusters:', data.support_clusters);
    });

    test('wider level_range should result in more clustering', async ({ request }) => {
      const responseNarrow = await request.get('http://localhost:5000/stock?symbol=AMZN&level_range=0.001');
      const responseWide = await request.get('http://localhost:5000/stock?symbol=AMZN&level_range=0.01');

      const dataNarrow = await responseNarrow.json();
      const dataWide = await responseWide.json();

      // Wider range should result in fewer or equal cluster keys (more grouping)
      const narrowClusterCount = Object.keys(dataNarrow.resistance_clusters).length;
      const wideClusterCount = Object.keys(dataWide.resistance_clusters).length;

      expect(wideClusterCount).toBeLessThanOrEqual(narrowClusterCount);

      console.log(`level_range=0.001: ${narrowClusterCount} clusters`);
      console.log(`level_range=0.01: ${wideClusterCount} clusters`);
    });
  });

  test.describe('UI Slider Tests', () => {
    test('should display First Retracement slider', async ({ page }) => {
      await page.goto('/');
      await page.waitForLoadState('domcontentloaded');
    await page.waitForTimeout(2000); // Wait for React to render

      await expect(page.getByText(/First Retracement:/)).toBeVisible({ timeout: 10000 });
    });

    test('should display Second Retracement slider', async ({ page }) => {
      await page.goto('/');
      await page.waitForLoadState('domcontentloaded');
    await page.waitForTimeout(2000); // Wait for React to render

      await expect(page.getByText(/Second Retracement:/)).toBeVisible({ timeout: 10000 });
    });

    test('should display Touch Count slider', async ({ page }) => {
      await page.goto('/');
      await page.waitForLoadState('domcontentloaded');
    await page.waitForTimeout(2000); // Wait for React to render

      await expect(page.getByText(/Touch Count:/)).toBeVisible({ timeout: 10000 });
    });

    test('should display Level Range slider', async ({ page }) => {
      await page.goto('/');
      await page.waitForLoadState('domcontentloaded');
    await page.waitForTimeout(2000); // Wait for React to render

      await expect(page.getByText(/Level Range:/)).toBeVisible({ timeout: 10000 });
    });

    test('all sliders should be visible and interactive', async ({ page }) => {
      await page.goto('/');
      await page.waitForLoadState('domcontentloaded');
    await page.waitForTimeout(2000); // Wait for React to render

      // Check that MUI sliders are present (4 sliders total)
      const sliders = page.locator('.MuiSlider-root');
      await expect(sliders).toHaveCount(4, { timeout: 10000 });
    });
  });
});
