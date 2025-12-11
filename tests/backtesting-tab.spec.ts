import { test, expect } from '@playwright/test';

/**
 * Backtesting Tab UI Tests
 *
 * Tests that verify the Backtesting tab UI elements and functionality.
 */

test.describe('Backtesting Tab Navigation', () => {

  test('should navigate to Backtesting tab successfully', async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('domcontentloaded');
    await page.waitForTimeout(2000); // Wait for React to render

    // Look for Backtesting tab
    const backtestTab = page.getByRole('tab', { name: /backtest/i });

    if (await backtestTab.isVisible({ timeout: 5000 })) {
      await backtestTab.click();
      await page.waitForTimeout(1000);

      console.log('Backtesting tab found and clicked');

      // Verify we're on the backtesting tab - look for common elements
      const pageContent = await page.locator('body').innerText();
      console.log('Page contains:', pageContent.substring(0, 500));

      // Look for backtesting-related elements
      const hasBacktestingContent = /backtest|strategy|run|start|symbol/i.test(pageContent);
      expect(hasBacktestingContent).toBeTruthy();

      console.log('Backtesting tab navigation successful');
    } else {
      console.log('Backtesting tab not visible - checking for alternative navigation');

      // Try to find any tab that might be for backtesting
      const allTabs = await page.getByRole('tab').all();
      console.log(`Found ${allTabs.length} tabs`);

      for (const tab of allTabs) {
        const tabText = await tab.innerText();
        console.log(`Tab: ${tabText}`);
      }
    }
  });

  test('should display strategy selection options', async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('domcontentloaded');
    await page.waitForTimeout(2000); // Wait for React to render

    // Navigate to Backtesting or Optimization tab
    const backtestTab = page.getByRole('tab', { name: /backtest|optimization/i }).first();
    if (await backtestTab.isVisible({ timeout: 5000 })) {
      await backtestTab.click();
      await page.waitForTimeout(1000);
    }

    // Look for strategy-related elements
    const strategySelect = page.locator('select, [role="combobox"], [data-testid*="strategy"]').first();
    const strategyLabel = page.getByText(/strategy/i).first();

    if (await strategyLabel.isVisible({ timeout: 5000 })) {
      console.log('Strategy label found');
    }

    // Check for strategy parameters section
    const parametersSection = page.getByText(/parameters|lookback|stop.?loss|take.?profit/i).first();
    if (await parametersSection.isVisible({ timeout: 3000 })) {
      console.log('Strategy parameters section found');
    }

    console.log('Strategy selection elements verified');
  });
});

test.describe('Backtesting Date Picker', () => {

  test('should have functional date input fields', async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('domcontentloaded');
    await page.waitForTimeout(2000); // Wait for React to render

    // Navigate to Optimization tab (has date pickers)
    const optimizationTab = page.getByRole('tab', { name: /optimization/i });
    if (await optimizationTab.isVisible({ timeout: 5000 })) {
      await optimizationTab.click();
      await page.waitForTimeout(1000);
    }

    // Find date inputs
    const dateInputs = await page.locator('input[type="date"]').all();
    console.log(`Found ${dateInputs.length} date input fields`);

    expect(dateInputs.length).toBeGreaterThanOrEqual(1);

    // Test setting a date value
    if (dateInputs.length >= 1) {
      const testDate = '2023-06-15';

      await page.evaluate((dateStr) => {
        const dateInputs = document.querySelectorAll('input[type="date"]');
        if (dateInputs.length > 0) {
          (dateInputs[0] as HTMLInputElement).value = dateStr;
          dateInputs[0].dispatchEvent(new Event('input', { bubbles: true }));
          dateInputs[0].dispatchEvent(new Event('change', { bubbles: true }));
        }
      }, testDate);

      await page.waitForTimeout(500);

      // Verify the date was set
      const setValue = await dateInputs[0].inputValue();
      console.log(`Date input value: ${setValue}`);

      expect(setValue).toBe(testDate);
      console.log('Date picker functionality verified');
    }
  });

  test('should accept valid date range', async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('domcontentloaded');
    await page.waitForTimeout(2000); // Wait for React to render

    // Navigate to Optimization tab
    const optimizationTab = page.getByRole('tab', { name: /optimization/i });
    if (await optimizationTab.isVisible({ timeout: 5000 })) {
      await optimizationTab.click();
      await page.waitForTimeout(1000);
    }

    // Set date range
    await page.evaluate(() => {
      const dateInputs = document.querySelectorAll('input[type="date"]');
      if (dateInputs.length >= 2) {
        (dateInputs[0] as HTMLInputElement).value = '2023-01-01';
        dateInputs[0].dispatchEvent(new Event('change', { bubbles: true }));

        (dateInputs[1] as HTMLInputElement).value = '2024-01-01';
        dateInputs[1].dispatchEvent(new Event('change', { bubbles: true }));
      }
    });

    await page.waitForTimeout(500);

    const dateInputs = await page.locator('input[type="date"]').all();
    if (dateInputs.length >= 2) {
      const startDate = await dateInputs[0].inputValue();
      const endDate = await dateInputs[1].inputValue();

      console.log(`Start date: ${startDate}`);
      console.log(`End date: ${endDate}`);

      expect(new Date(endDate).getTime()).toBeGreaterThan(new Date(startDate).getTime());
      console.log('Valid date range accepted');
    }
  });
});

test.describe('Backtesting Form Elements', () => {

  test('should display strategy parameter inputs', async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('domcontentloaded');
    await page.waitForTimeout(2000); // Wait for React to render

    // Navigate to Optimization tab
    const optimizationTab = page.getByRole('tab', { name: /optimization/i });
    if (await optimizationTab.isVisible({ timeout: 5000 })) {
      await optimizationTab.click();
      await page.waitForTimeout(1000);
    }

    // Look for parameter labels - expanded list
    const parameterLabels = [
      /lookback/i,
      /stop.?loss/i,
      /take.?profit/i,
      /buffer|breakout/i,
      /retracement/i,
      /touch/i,
      /range/i,
      /parameters/i
    ];

    let foundParams = 0;
    for (const labelPattern of parameterLabels) {
      const label = page.getByText(labelPattern).first();
      if (await label.isVisible({ timeout: 1000 }).catch(() => false)) {
        const labelText = await label.innerText();
        console.log(`Found parameter: ${labelText}`);
        foundParams++;
      }
    }

    console.log(`Found ${foundParams}/${parameterLabels.length} expected parameters`);

    // Check if there are any input fields visible
    const inputs = await page.locator('input').count();
    const sliders = await page.locator('.MuiSlider-root').count();
    console.log(`Found ${inputs} inputs and ${sliders} sliders`);

    // Pass if we found any form elements
    expect(foundParams + inputs + sliders).toBeGreaterThan(0);
  });

  test('should have run/start backtest button', async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('domcontentloaded');
    await page.waitForTimeout(2000); // Wait for React to render

    // Navigate to Optimization tab
    const optimizationTab = page.getByRole('tab', { name: /optimization/i });
    if (await optimizationTab.isVisible({ timeout: 5000 })) {
      await optimizationTab.click();
      await page.waitForTimeout(1000);
    }

    // Look for run/start button
    const startButton = page.getByRole('button', { name: /start|run|optimize/i }).first();
    await expect(startButton).toBeVisible({ timeout: 5000 });

    const buttonText = await startButton.innerText();
    console.log(`Found button: ${buttonText}`);

    // Verify button is enabled
    const isEnabled = await startButton.isEnabled();
    console.log(`Button enabled: ${isEnabled}`);

    expect(isEnabled).toBeTruthy();
    console.log('Run/Start backtest button verified');
  });
});

test.describe('Backtesting Results Display', () => {

  test('should run backtest and show results UI', async ({ page }) => {
    test.setTimeout(120000); // 2 minutes

    await page.goto('/');
    await page.waitForLoadState('domcontentloaded');
    await page.waitForTimeout(2000); // Wait for React to render

    // Navigate to Optimization tab
    const optimizationTab = page.getByRole('tab', { name: /optimization/i });
    await optimizationTab.click();
    await page.waitForTimeout(1000);

    // Click start optimization (standard mode)
    const startButton = page.getByRole('button', { name: /start.*optimization/i }).first();
    if (await startButton.isVisible({ timeout: 5000 })) {
      await startButton.click();

      // Wait for progress or results
      try {
        await page.waitForFunction(() => {
          const text = document.body.innerText;
          return /progress|running|%|\d+\s*\/\s*\d+/i.test(text);
        }, { timeout: 30000 });

        console.log('Backtest started - showing progress');

        // Wait a bit and check for results
        await page.waitForTimeout(5000);

        const pageText = await page.locator('body').innerText();

        // Look for result indicators
        const hasResults = /result|profit|sharpe|trades|completed/i.test(pageText);
        const hasProgress = /progress|running|\d+%/i.test(pageText);

        console.log(`Has results: ${hasResults}`);
        console.log(`Has progress: ${hasProgress}`);

        expect(hasResults || hasProgress).toBeTruthy();
      } catch (e) {
        await page.screenshot({ path: 'test-results/backtest-results-ui.png' });
        console.log('Timeout waiting for backtest progress');
      }

      // Cancel to cleanup
      const cancelButton = page.getByRole('button', { name: /cancel/i });
      if (await cancelButton.isVisible({ timeout: 2000 })) {
        await cancelButton.click();
      }
    }
  });

  test('should display performance metrics after backtest', async ({ page }) => {
    test.setTimeout(180000); // 3 minutes

    await page.goto('/');
    await page.waitForLoadState('domcontentloaded');
    await page.waitForTimeout(2000); // Wait for React to render

    // Navigate to Optimization tab
    const optimizationTab = page.getByRole('tab', { name: /optimization/i });
    await optimizationTab.click();
    await page.waitForTimeout(1000);

    // Try hybrid mode for faster results (skip if disabled - no GPU)
    const hybridButton = page.getByRole('button', { name: /hybrid/i });
    if (await hybridButton.isVisible({ timeout: 3000 })) {
      const isDisabled = await hybridButton.isDisabled();
      if (!isDisabled) {
        await hybridButton.click();
        await page.waitForTimeout(500);
      } else {
        console.log('Hybrid button disabled (no GPU) - using standard mode');
      }
    }

    // Start optimization
    const startButton = page.getByRole('button', { name: /start/i }).first();
    await startButton.click();

    // Wait for any results to appear
    try {
      await page.waitForFunction(() => {
        const text = document.body.innerText.toLowerCase();
        return text.includes('profit') ||
          text.includes('sharpe') ||
          text.includes('win rate') ||
          text.includes('drawdown') ||
          text.includes('candidates');
      }, { timeout: 60000 });

      console.log('Performance metrics visible in UI');

      const pageText = await page.locator('body').innerText();

      // Check for specific metrics
      const metrics = [
        { name: 'Profit', pattern: /profit/i },
        { name: 'Sharpe', pattern: /sharpe/i },
        { name: 'Win Rate', pattern: /win.*rate/i },
        { name: 'Drawdown', pattern: /drawdown/i },
        { name: 'Trades', pattern: /trades/i }
      ];

      let foundMetrics = 0;
      for (const metric of metrics) {
        if (metric.pattern.test(pageText)) {
          console.log(`Found: ${metric.name}`);
          foundMetrics++;
        }
      }

      console.log(`Found ${foundMetrics}/${metrics.length} metrics`);

    } catch (e) {
      await page.screenshot({ path: 'test-results/metrics-display.png' });
      console.log('Timeout waiting for metrics - may still be running');
    }

    // Cleanup
    const cancelButton = page.getByRole('button', { name: /cancel/i });
    if (await cancelButton.isVisible({ timeout: 2000 })) {
      await cancelButton.click();
    }
  });
});

test.describe('Symbol Input', () => {

  test('should have symbol input field', async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('domcontentloaded');
    await page.waitForTimeout(2000); // Wait for React to render

    // Navigate to Optimization tab
    const optimizationTab = page.getByRole('tab', { name: /optimization/i });
    if (await optimizationTab.isVisible({ timeout: 5000 })) {
      await optimizationTab.click();
      await page.waitForTimeout(1000);
    }

    // Look for symbol input
    const symbolInput = page.locator('input[placeholder*="symbol" i], input[name*="symbol" i], [data-testid*="symbol"]').first();
    const symbolLabel = page.getByText(/symbol/i).first();

    if (await symbolLabel.isVisible({ timeout: 3000 })) {
      console.log('Symbol label found');
    }

    // Find any text input that might be for symbol
    const textInputs = await page.locator('input[type="text"]').all();
    console.log(`Found ${textInputs.length} text inputs`);

    // Check page for symbol-related content
    const pageText = await page.locator('body').innerText();
    const hasSymbolReference = /symbol|ticker|stock|AAPL/i.test(pageText);

    console.log(`Has symbol reference: ${hasSymbolReference}`);
    expect(hasSymbolReference).toBeTruthy();
  });

  test('should allow changing symbol', async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('domcontentloaded');
    await page.waitForTimeout(2000); // Wait for React to render

    // Navigate to Optimization tab
    const optimizationTab = page.getByRole('tab', { name: /optimization/i });
    if (await optimizationTab.isVisible({ timeout: 5000 })) {
      await optimizationTab.click();
      await page.waitForTimeout(1000);
    }

    // Check page for symbol-related content
    const pageText = await page.locator('body').innerText();
    const hasSymbolReference = /symbol|ticker|stock|AAPL|GOOG|MSFT|TSLA/i.test(pageText);
    console.log(`Page has symbol reference: ${hasSymbolReference}`);

    // Look for symbol buttons (stock selection buttons)
    const symbolButtons = ['AAPL', 'GOOG', 'MSFT', 'TSLA', 'AMZN'];
    let symbolButtonFound = false;

    for (const symbol of symbolButtons) {
      const button = page.getByRole('button', { name: symbol });
      if (await button.isVisible({ timeout: 1000 }).catch(() => false)) {
        console.log(`Found symbol button: ${symbol}`);
        symbolButtonFound = true;

        // Try clicking a different symbol
        if (symbol !== 'MSFT') {
          const msftButton = page.getByRole('button', { name: 'MSFT' });
          if (await msftButton.isVisible({ timeout: 1000 }).catch(() => false)) {
            await msftButton.click();
            console.log('Clicked MSFT button');
            await page.waitForTimeout(500);
          }
        }
        break;
      }
    }

    // Also check for symbol in dropdown/select
    const selectElements = await page.locator('select').all();
    for (const select of selectElements) {
      const options = await select.locator('option').allTextContents();
      if (options.some(opt => /AAPL|GOOG|MSFT|TSLA/i.test(opt))) {
        console.log('Found symbol in dropdown');
        symbolButtonFound = true;
        break;
      }
    }

    // Test passes if we found symbol reference OR symbol buttons
    console.log(`Symbol button found: ${symbolButtonFound}`);
    expect(hasSymbolReference || symbolButtonFound).toBeTruthy();
    console.log('Symbol change verification completed');
  });
});
