import eventlet
eventlet.monkey_patch()

from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_socketio import SocketIO, emit
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import yfinance as yf
import backtrader as bt
import os
import re
import tempfile
import logging
import uuid
import time
import threading

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Initialize SocketIO
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')

# Cache for stock data: {(symbol, period): {'data': DataFrame, 'timestamp': datetime}}
stock_cache = {}
CACHE_DURATION_MINUTES = 15  # Cache stock data for 15 minutes


def get_cached_stock_data(symbol, period):
    """Get stock data from cache or fetch from yfinance if not cached/stale."""
    cache_key = (symbol, period)
    now = datetime.now()

    # Check if we have valid cached data
    if cache_key in stock_cache:
        cached = stock_cache[cache_key]
        age = now - cached['timestamp']
        if age < timedelta(minutes=CACHE_DURATION_MINUTES):
            logger.info(f"Cache HIT for {symbol} ({period}) - age: {age.seconds}s")
            return cached['data'].copy()

    # Fetch fresh data from yfinance
    logger.info(f"Cache MISS for {symbol} ({period}) - fetching from yfinance")
    stock = yf.Ticker(symbol)
    stock_data = stock.history(period=period)

    if not stock_data.empty:
        # Store in cache
        stock_cache[cache_key] = {
            'data': stock_data.copy(),
            'timestamp': now
        }

    return stock_data


# Restrict CORS to frontend origin (configure via environment variable)
CORS(app, origins=[os.environ.get('FRONTEND_URL', 'http://localhost:3000')])


def calculate_percentage_change(price_from, price_to):
    """Calculate percentage change from price_from to price_to."""
    if price_from == 0:
        return 0
    return ((price_to - price_from) / price_from) * 100


def runBacktrader(data, first_retracement=5, second_retracement=5, level_range=0.001, touch_count=1, plot=False):
    """Run backtrader strategy on the given data."""
    cerebro = bt.Cerebro()
    cerebro.adddata(data)
    cerebro.addstrategy(
        SupResStrategy,
        FIRST_RETRACEMENT=first_retracement,
        SECOND_RETRACEMENT=second_retracement,
        RES_SUP_RANGE=level_range,
        TOUCH_COUNT=touch_count
    )

    initial_cash = int(os.environ.get('INITIAL_CASH', 100000))
    cerebro.broker.set_cash(initial_cash)
    strategies = cerebro.run()

    # Recalculate strengths with final data (age, touch counts, etc.)
    strategies[0].recalculate_all_strengths()

    # Only plot if explicitly enabled (requires display)
    if plot and os.environ.get('ENABLE_PLOT', 'False').lower() == 'true':
        cerebro.plot()

    return strategies[0]


class Bar:
    def __init__(self):
        self.Date = datetime.now()
        self.High = 0
        self.Low = 0
        self.Open = 0
        self.Close = 0
        self.Volume = 0
        self.Dividends = 0
        self.Stock_splits = 0


class SupResStrategy(bt.Strategy):
    """
    Support/Resistance Strategy using cAlgo-style sequential bar tracking.

    Detects swing highs/lows using percentage-based retracements:
    - Track startBar → middleBar → endBar pattern
    - First retracement: percentage move from start to middle
    - Second retracement: percentage pullback from middle to end
    """
    params = (
        ("FIRST_RETRACEMENT", 5),      # Min % for first swing move
        ("SECOND_RETRACEMENT", 5),     # Min % for pullback confirmation
        ("RES_SUP_RANGE", 0.001),      # % range to cluster similar levels
        ("TOUCH_COUNT", 1),            # Min touches before trading
        ("TAKE_PROFIT", 10),
        ("STOP_LOSS", 5),
        ("RISK_PERCENTAGE", 5),
        ("BREAKOUT_BUFFER", 0),        # % above resistance to trigger buy
    )

    def __init__(self):
        # Level tracking
        self.resistance_levels = []
        self.support_levels = []

        # Level clustering dictionaries {base_price: [list of touches]}
        self.resistance_clusters = {}
        self.support_clusters = {}

        # Track first touch date for each cluster (for level age calculation)
        self.resistance_first_touch = {}  # {cluster_key: date}
        self.support_first_touch = {}     # {cluster_key: date}

        # Track bounce percentages for each cluster (for bounce strength)
        self.resistance_bounces = {}  # {cluster_key: [bounce_pct, ...]}
        self.support_bounces = {}     # {cluster_key: [bounce_pct, ...]}

        # Track last touch date for each cluster (for recent respect)
        self.resistance_last_touch = {}  # {cluster_key: date}
        self.support_last_touch = {}     # {cluster_key: date}

        # Sequential bar tracking for resistance (swing highs)
        # Pattern: start (low point) → middle (high point) → end (pullback)
        self.start_high_bar = {'date': None, 'high': 0, 'low': float('inf')}
        self.middle_high_bar = {'date': None, 'high': 0, 'low': 0}

        # Sequential bar tracking for support (swing lows)
        # Pattern: start (high point) → middle (low point) → end (bounce)
        self.start_low_bar = {'date': None, 'high': 0, 'low': float('inf')}
        self.middle_low_bar = {'date': None, 'high': 0, 'low': float('inf')}

        # Order tracking
        self.order = None
        self.buyprice = None
        self.buycomm = None

        # Data for Frontend
        self.trades = []
        self.levels = []

        # Debug: Track swing patterns for visualization
        self.resistance_swings = []  # List of {start, middle, end} points
        self.support_swings = []     # List of {start, middle, end} points

        # Track broken levels
        self.broken_resistances = set()
        self.broken_supports = set()

        # Current date tracking for strength calculations
        self.current_date = None

        # Track all-time high and low for "never broken" calculation
        self.all_time_high = 0
        self.all_time_low = float('inf')

    def log(self, txt, dt=None):
        """Logging function for this strategy"""
        dt = dt or self.datas[0].datetime.date(0)
        print(f'{dt.isoformat()}, {txt}')

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status in [order.Completed]:
            dt = self.datas[0].datetime.date(0).isoformat()
            if order.isbuy():
                self.log(f'BUY EXECUTED, Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}')
                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
                self.trades.append({
                    'type': 'buy',
                    'price': order.executed.price,
                    'date': dt
                })
            elif order.issell():
                self.log(f'SELL EXECUTED, Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}')
                self.trades.append({
                    'type': 'sell',
                    'price': order.executed.price,
                    'date': dt
                })
            self.bar_executed = len(self)

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')

        self.order = None

    def notify_trade(self, trade):
        if not trade.isclosed:
            return
        self.log(f'OPERATION PROFIT, GROSS {trade.pnl:.2f}, NET {trade.pnlcomm:.2f}')

    def is_round_number(self, price):
        """Check if price is a round number (ends in 0, 00, 50, etc.)."""
        # Check for multiples of 10
        if price % 10 == 0:
            return True
        # Check for multiples of 5 (like 105, 155)
        if price % 5 == 0:
            return True
        # Check for .00 ending (whole dollars)
        if price == int(price):
            return True
        return False

    def is_level_broken(self, level, level_type):
        """Check if price has crossed through this level."""
        # A level is "broken" if price went beyond it
        # Use 0.1% tolerance for floating point precision only

        if level_type == 'resistance':
            # Resistance is broken if price ever went above it
            return self.all_time_high > level * 1.001
        else:  # support
            # Support is broken if price ever went below it
            return self.all_time_low < level * 0.999

    def calculate_level_strength(self, level, level_type, cluster_key, touch_count, bounce_pct=0):
        """
        Calculate strength score (0-100) for a support/resistance level.

        Scoring factors:
        - Never Broken (25 pts): Level has never been crossed by price
        - Touch Count (25 pts max): More touches = stronger validation
        - Bounce Strength (20 pts): Larger bounces = stronger level
        - Level Age (15 pts): Older levels more validated
        - Confluence (10 pts): Round numbers, price clusters
        - Cluster Tightness (5 pts): How close touches are to each other
        """
        score = 0

        # 1. Never Broken (25 pts)
        # This is the most important factor - unbroken levels are strong
        if not self.is_level_broken(level, level_type):
            score += 25

        # 2. Touch Count (25 pts max)
        # 1 touch = 8, 2 = 15, 3 = 21, 4+ = 25
        if touch_count >= 4:
            score += 25
        elif touch_count == 3:
            score += 21
        elif touch_count == 2:
            score += 15
        else:
            score += 8

        # 3. Bounce Strength (20 pts max)
        # How strongly price rejected from this level
        if level_type == 'resistance':
            bounces = self.resistance_bounces.get(cluster_key, [])
        else:
            bounces = self.support_bounces.get(cluster_key, [])

        if bounces:
            avg_bounce = sum(bounces) / len(bounces)
            max_bounce = max(bounces)
            # Scale: 5% = 8pts, 10% = 15pts, 15%+ = 20pts
            bounce_score = min(avg_bounce * 1.5, 12) + min(max_bounce * 0.4, 8)
            score += int(bounce_score)
        elif bounce_pct > 0:
            score += int(min(bounce_pct * 1.5, 12))

        # 4. Level Age (15 pts max)
        # Older levels that still hold are more significant
        if level_type == 'resistance':
            first_date = self.resistance_first_touch.get(cluster_key)
        else:
            first_date = self.support_first_touch.get(cluster_key)

        if first_date and self.current_date:
            age_days = (self.current_date - first_date).days
            # Scale: 30 days = 7pts, 90 days = 12pts, 180+ days = 15pts
            if age_days >= 180:
                score += 15
            elif age_days >= 90:
                score += 12
            elif age_days >= 30:
                score += 7
            else:
                score += int(age_days / 5)  # 0-6 based on days

        # 5. Confluence (10 pts max)
        # Round numbers are psychologically significant
        if level % 100 == 0:  # Multiples of 100 (e.g., 200, 300)
            score += 10
        elif level % 50 == 0:  # Multiples of 50
            score += 8
        elif level % 10 == 0:  # Multiples of 10
            score += 5
        elif level % 5 == 0:   # Multiples of 5
            score += 2

        # 6. Cluster Tightness (5 pts max)
        # How close are touches within this cluster? Tighter = stronger
        if level_type == 'resistance':
            cluster_prices = self.resistance_clusters.get(cluster_key, [])
        else:
            cluster_prices = self.support_clusters.get(cluster_key, [])

        if len(cluster_prices) >= 2:
            # Calculate % spread of prices in cluster
            min_price = min(cluster_prices)
            max_price = max(cluster_prices)
            spread_pct = (max_price - min_price) / cluster_key * 100 if cluster_key > 0 else 0

            # Tighter clusters get more points
            # 0% spread = 5pts, 0.5% = 3pts, 1%+ = 0pts
            if spread_pct <= 0.1:
                score += 5
            elif spread_pct <= 0.3:
                score += 4
            elif spread_pct <= 0.5:
                score += 2
            elif spread_pct <= 1.0:
                score += 1
            # More than 1% spread = 0 additional points
        else:
            # Single touch - give middle points
            score += 2

        return min(int(score), 100)

    def add_to_cluster(self, clusters, level, range_pct):
        """Add level to cluster and return the cluster key and count."""
        found_cluster = None
        for base in list(clusters.keys()):
            high_range = base * (1 + range_pct)
            low_range = base * (1 - range_pct)
            if low_range <= level <= high_range:
                found_cluster = base
                break

        if found_cluster:
            clusters[found_cluster].append(level)
            return found_cluster, len(clusters[found_cluster])
        else:
            clusters[level] = [level]
            return level, 1

    def calculating_top(self):
        """
        Detect resistance levels using sequential bar tracking.

        Pattern: start (lowest point) → middle (highest point) → end (pullback)

        Example with 50% first retracement and 10% second retracement:
        - Start at 143 (lowest low)
        - Price rises to 214.5 (143 × 1.50 = 50% up) → first retracement met
        - Price drops to 193.1 (214.5 × 0.90 = 10% down) → second retracement met
        - SIGNAL: Resistance at 214.5

        If price keeps going up (e.g., to 250) instead of pulling back:
        - Middle updates to 250
        - Now need 10% drop from 250 (to 225) for signal
        - Signal would be at 250 when pullback condition met
        """
        dt = self.datas[0].datetime.date(0)
        current_high = self.data.high[0]
        current_low = self.data.low[0]

        # Track if middle was updated THIS bar (can't signal on same bar as new high)
        middle_updated_this_bar = False

        # Update start bar - track the lowest point (resets the pattern)
        if current_low < self.start_high_bar['low']:
            self.start_high_bar = {'date': dt, 'high': current_high, 'low': current_low}
            self.middle_high_bar = {'date': dt, 'high': current_high, 'low': current_low}
            middle_updated_this_bar = True

        # Update middle bar - track highest high after the low
        # This continuously updates as price makes new highs
        if current_high > self.middle_high_bar['high']:
            self.middle_high_bar = {'date': dt, 'high': current_high, 'low': current_low}
            middle_updated_this_bar = True

        # Can't form signal on same bar that made the high (need actual pullback)
        if middle_updated_this_bar:
            return

        # Calculate first retracement: % move UP from start low to middle high
        first_retracement = calculate_percentage_change(
            self.start_high_bar['low'],
            self.middle_high_bar['high']
        )

        # Check if first retracement meets threshold
        if first_retracement >= self.params.FIRST_RETRACEMENT:
            # Calculate second retracement: % pullback from middle high to current low
            second_retracement = calculate_percentage_change(
                self.middle_high_bar['high'],
                current_low
            )
            # Note: second_retracement will be negative (price dropped), so we use abs
            second_retracement = abs(second_retracement)

            # Check if we have a valid swing high (resistance)
            if second_retracement >= self.params.SECOND_RETRACEMENT:
                resistance_level = round(self.middle_high_bar['high'], 2)

                # Track swing for debug visualization
                self.resistance_swings.append({
                    'start': {
                        'date': self.start_high_bar['date'].isoformat() if self.start_high_bar['date'] else None,
                        'price': self.start_high_bar['low']
                    },
                    'middle': {
                        'date': self.middle_high_bar['date'].isoformat() if self.middle_high_bar['date'] else None,
                        'price': self.middle_high_bar['high']
                    },
                    'end': {
                        'date': dt.isoformat(),
                        'price': current_low
                    }
                })

                # Add to cluster and get touch count
                cluster_key, touch_count = self.add_to_cluster(
                    self.resistance_clusters,
                    resistance_level,
                    self.params.RES_SUP_RANGE
                )

                # Track first touch date for this cluster (for level age)
                if cluster_key not in self.resistance_first_touch:
                    self.resistance_first_touch[cluster_key] = dt

                # Track last touch date (for recent respect)
                self.resistance_last_touch[cluster_key] = dt

                # Track bounce percentage (second retracement = bounce from level)
                if cluster_key not in self.resistance_bounces:
                    self.resistance_bounces[cluster_key] = []
                self.resistance_bounces[cluster_key].append(second_retracement)

                self.log(f'RESISTANCE DETECTED: {resistance_level:.2f} (first: {first_retracement:.1f}%, second: {second_retracement:.1f}%, touches: {touch_count})')

                # Only add to levels if meets touch count threshold
                if touch_count >= self.params.TOUCH_COUNT:
                    # Use the max value from cluster as the resistance level
                    max_resistance = max(self.resistance_clusters[cluster_key])

                    # Calculate strength score
                    strength = self.calculate_level_strength(
                        max_resistance,
                        'resistance',
                        cluster_key,
                        touch_count,
                        second_retracement
                    )

                    # Check if level already exists, update if so
                    existing = next((l for l in self.resistance_levels if l['price'] == max_resistance), None)
                    if existing:
                        existing['strength'] = strength
                    else:
                        self.resistance_levels.append({'price': max_resistance, 'strength': strength})

                # Reset: start from the current pullback point for next pattern
                self.start_high_bar = {'date': dt, 'high': current_high, 'low': current_low}
                self.middle_high_bar = {'date': dt, 'high': current_high, 'low': current_low}

    def calculating_bottom(self):
        """
        Detect support levels using sequential bar tracking.

        Pattern: start (highest point) → middle (lowest point) → end (bounce)

        Example with 50% first retracement and 10% second retracement:
        - Start at 200 (highest high)
        - Price drops to 100 (200 × 0.50 = 50% down) → first retracement met
        - Price bounces to 110 (100 × 1.10 = 10% up) → second retracement met
        - SIGNAL: Support at 100

        If price keeps going down (e.g., to 80) instead of bouncing:
        - Middle updates to 80
        - Now need 10% bounce from 80 (to 88) for signal
        - Signal would be at 80 when bounce condition met
        """
        dt = self.datas[0].datetime.date(0)
        current_high = self.data.high[0]
        current_low = self.data.low[0]

        # Track if middle was updated THIS bar (can't signal on same bar as new low)
        middle_updated_this_bar = False

        # Update start bar - track the highest point (resets the pattern)
        if current_high > self.start_low_bar['high']:
            self.start_low_bar = {'date': dt, 'high': current_high, 'low': current_low}
            self.middle_low_bar = {'date': dt, 'high': current_high, 'low': current_low}
            middle_updated_this_bar = True

        # Update middle bar - track lowest low after the high
        # This continuously updates as price makes new lows
        if current_low < self.middle_low_bar['low']:
            self.middle_low_bar = {'date': dt, 'high': current_high, 'low': current_low}
            middle_updated_this_bar = True

        # Can't form signal on same bar that made the low (need actual bounce)
        if middle_updated_this_bar:
            return

        # Calculate first retracement: % move DOWN from start high to middle low
        first_retracement = calculate_percentage_change(
            self.start_low_bar['high'],
            self.middle_low_bar['low']
        )
        # Note: first_retracement will be negative (price dropped), so we use abs
        first_retracement = abs(first_retracement)

        # Check if first retracement meets threshold
        if first_retracement >= self.params.FIRST_RETRACEMENT:
            # Calculate second retracement: % bounce from middle low to current high
            second_retracement = calculate_percentage_change(
                self.middle_low_bar['low'],
                current_high
            )

            # Check if we have a valid swing low (support)
            if second_retracement >= self.params.SECOND_RETRACEMENT:
                support_level = round(self.middle_low_bar['low'], 2)

                # Track swing for debug visualization
                self.support_swings.append({
                    'start': {
                        'date': self.start_low_bar['date'].isoformat() if self.start_low_bar['date'] else None,
                        'price': self.start_low_bar['high']
                    },
                    'middle': {
                        'date': self.middle_low_bar['date'].isoformat() if self.middle_low_bar['date'] else None,
                        'price': self.middle_low_bar['low']
                    },
                    'end': {
                        'date': dt.isoformat(),
                        'price': current_high
                    }
                })

                # Add to cluster and get touch count
                cluster_key, touch_count = self.add_to_cluster(
                    self.support_clusters,
                    support_level,
                    self.params.RES_SUP_RANGE
                )

                # Track first touch date for this cluster (for level age)
                if cluster_key not in self.support_first_touch:
                    self.support_first_touch[cluster_key] = dt

                # Track last touch date (for recent respect)
                self.support_last_touch[cluster_key] = dt

                # Track bounce percentage (second retracement = bounce from level)
                if cluster_key not in self.support_bounces:
                    self.support_bounces[cluster_key] = []
                self.support_bounces[cluster_key].append(second_retracement)

                self.log(f'SUPPORT DETECTED: {support_level:.2f} (first: {first_retracement:.1f}%, second: {second_retracement:.1f}%, touches: {touch_count})')

                # Only add to levels if meets touch count threshold
                if touch_count >= self.params.TOUCH_COUNT:
                    # Use the min value from cluster as the support level
                    min_support = min(self.support_clusters[cluster_key])

                    # Calculate strength score
                    strength = self.calculate_level_strength(
                        min_support,
                        'support',
                        cluster_key,
                        touch_count,
                        second_retracement
                    )

                    # Check if level already exists, update if so
                    existing = next((l for l in self.support_levels if l['price'] == min_support), None)
                    if existing:
                        existing['strength'] = strength
                    else:
                        self.support_levels.append({'price': min_support, 'strength': strength})

                # Reset: start from the current bounce point for next pattern
                self.start_low_bar = {'date': dt, 'high': current_high, 'low': current_low}
                self.middle_low_bar = {'date': dt, 'high': current_high, 'low': current_low}

    def recalculate_all_strengths(self):
        """Recalculate strength for all levels using final data."""
        # Recalculate resistance levels
        for level_data in self.resistance_levels:
            price = level_data['price']
            # Find the cluster key for this price
            cluster_key = None
            for key in self.resistance_clusters:
                if price in self.resistance_clusters[key] or abs(price - key) < price * 0.002:
                    cluster_key = key
                    break
            if cluster_key:
                touch_count = len(self.resistance_clusters.get(cluster_key, []))
                level_data['strength'] = self.calculate_level_strength(
                    price, 'resistance', cluster_key, touch_count
                )

        # Recalculate support levels
        for level_data in self.support_levels:
            price = level_data['price']
            # Find the cluster key for this price
            cluster_key = None
            for key in self.support_clusters:
                if price in self.support_clusters[key] or abs(price - key) < price * 0.002:
                    cluster_key = key
                    break
            if cluster_key:
                touch_count = len(self.support_clusters.get(cluster_key, []))
                level_data['strength'] = self.calculate_level_strength(
                    price, 'support', cluster_key, touch_count
                )

    def next(self):
        dt_obj = self.datas[0].datetime.date(0)
        dt = dt_obj.isoformat()
        current_close = self.data.close[0]
        current_high = self.data.high[0]
        current_low = self.data.low[0]

        # Track current date for strength calculations
        self.current_date = dt_obj

        # Track all-time high and low for "never broken" calculation
        self.all_time_high = max(self.all_time_high, current_high)
        self.all_time_low = min(self.all_time_low, current_low)

        # === SWING DETECTION (cAlgo style) ===
        self.calculating_top()
        self.calculating_bottom()

        # Record levels for frontend (extract price from new format)
        self.levels.append({
            'date': dt,
            'resistance': self.resistance_levels[-1]['price'] if self.resistance_levels else None,
            'support': self.support_levels[-1]['price'] if self.support_levels else None
        })

        # === TRADING LOGIC - BREAKOUT STRATEGY ===
        if self.order:
            return

        if not self.position:
            # BUY on BREAKOUT above resistance (with optional buffer)
            prev_close = self.data.close[-1] if len(self) > 1 else current_close

            # Extract prices from level dicts and sort
            resistance_prices = sorted([l['price'] for l in self.resistance_levels])
            for resistance in resistance_prices:
                # Calculate entry trigger: resistance + (buffer% of resistance)
                entry_trigger = resistance * (1 + self.params.BREAKOUT_BUFFER / 100)

                # Breakout: price crossed above entry trigger
                if prev_close <= entry_trigger < current_close and resistance not in self.broken_resistances:
                    cash = self.broker.get_cash()
                    size = int((cash * self.params.RISK_PERCENTAGE / 100) / current_close)
                    if size < 1:
                        size = 1
                    self.log(f'BUY CREATE, {current_close:.2f} (BREAKOUT above {resistance:.2f} + {self.params.BREAKOUT_BUFFER}% buffer)')
                    self.order = self.buy(size=size)
                    self.broken_resistances.add(resistance)
                    break

        else:
            # SELL when position is profitable or stop loss
            if self.position.size > 0:
                # Take profit
                if self.buyprice and current_close >= self.buyprice * (1 + self.params.TAKE_PROFIT / 100):
                    self.log(f'SELL CREATE (TAKE PROFIT), {current_close:.2f}')
                    self.order = self.close()
                # Stop loss
                elif self.buyprice and current_close <= self.buyprice * (1 - self.params.STOP_LOSS / 100):
                    self.log(f'SELL CREATE (STOP LOSS), {current_close:.2f}')
                    self.order = self.close()
                # Breakdown below support
                elif self.support_levels:
                    support_prices = [l['price'] for l in self.support_levels]
                    for support in support_prices:
                        if current_close < support and support not in self.broken_supports:
                            self.log(f'SELL CREATE, {current_close:.2f} (BREAKDOWN below {support:.2f})')
                            self.order = self.close()
                            self.broken_supports.add(support)
                            break


def validate_symbol(symbol):
    """Validate stock symbol format."""
    if not symbol:
        return False
    # Allow 1-5 uppercase letters (standard stock symbol format)
    if not re.match(r'^[A-Z]{1,5}$', symbol.upper()):
        return False
    return True


@app.route('/stock')
def get_stock_data():
    try:
        symbol = request.args.get('symbol', default='AMZN').upper()

        # Percentage-based parameters
        first_retracement = float(request.args.get('first_retracement', default=5))
        second_retracement = float(request.args.get('second_retracement', default=5))
        level_range = float(request.args.get('level_range', default=0.001))
        touch_count = int(request.args.get('touch_count', default=1))

        # Timeframe parameter (valid yfinance periods)
        valid_periods = ['1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'max']
        period = request.args.get('period', default='2y')
        if period not in valid_periods:
            period = '2y'

        # Validate symbol format
        if not validate_symbol(symbol):
            return jsonify({"error": "Invalid stock symbol format"}), 400

        # Get stock data from cache or fetch fresh
        stock_data = get_cached_stock_data(symbol, period)

        if stock_data.empty:
            return jsonify({"error": f"No data found for symbol: {symbol}"}), 404

        # Prepare data for Backtrader (Keep index as datetime)
        bt_dataframe = stock_data.copy()

        # Prepare data for JSON response (Convert index to string)
        stock_data.index = stock_data.index.strftime("%Y-%m-%d")
        stock_dict = stock_data.to_dict(orient="index")

        strat = None
        try:
            # Use PandasData feed directly
            data = bt.feeds.PandasData(
                dataname=bt_dataframe,
                open='Open',
                high='High',
                low='Low',
                close='Close',
                volume='Volume',
                openinterest=None
            )

            strat = runBacktrader(
                data,
                first_retracement=first_retracement,
                second_retracement=second_retracement,
                level_range=level_range,
                touch_count=touch_count
            )
        except Exception as e:
            logger.error(f"Backtrader error: {str(e)}")
            # Continue even if strategy fails, returning just stock data
            pass

        # Calculate absolute resistance/support from detected signals
        # Absolute Resistance = highest detected resistance level
        # Absolute Support = lowest detected support level
        absolute_resistance = None
        absolute_support = None

        # Debug logging
        if strat is not None:
            logger.info(f"DEBUG: all_time_high = {strat.all_time_high}, all_time_low = {strat.all_time_low}")
            for level in strat.resistance_levels:
                is_broken = strat.all_time_high > level['price'] * 1.001
                logger.info(f"DEBUG: Resistance {level['price']} - broken={is_broken} (ATH > {level['price'] * 1.001})")
            for level in strat.support_levels:
                is_broken = strat.all_time_low < level['price'] * 0.999
                logger.info(f"DEBUG: Support {level['price']} - broken={is_broken} (ATL < {level['price'] * 0.999})")

        if strat is not None and strat.resistance_levels:
            # Find the highest resistance level by price
            max_resistance = max(strat.resistance_levels, key=lambda x: x['price'])
            # Find when this level was first detected from swings
            resistance_date = None
            for swing in strat.resistance_swings:
                if swing.get('middle') and abs(swing['middle']['price'] - max_resistance['price']) < max_resistance['price'] * 0.01:
                    resistance_date = swing['middle'].get('date')
                    break
            # Absolute resistance gets +25 bonus since it's the highest detected level
            # (It may appear "broken" by all_time_high but it's still the strongest signal)
            abs_res_strength = min(100, max_resistance.get('strength', 50) + 25)
            absolute_resistance = {
                "price": max_resistance['price'],
                "strength": abs_res_strength,
                "date": resistance_date
            }

        if strat is not None and strat.support_levels:
            # Find the lowest support level by price
            min_support = min(strat.support_levels, key=lambda x: x['price'])
            # Find when this level was first detected from swings
            support_date = None
            for swing in strat.support_swings:
                if swing.get('middle') and abs(swing['middle']['price'] - min_support['price']) < min_support['price'] * 0.01:
                    support_date = swing['middle'].get('date')
                    break
            # Absolute support gets +25 bonus since it's the lowest detected level
            # (It may appear "broken" by all_time_low but it's still the strongest signal)
            abs_sup_strength = min(100, min_support.get('strength', 50) + 25)
            absolute_support = {
                "price": min_support['price'],
                "strength": abs_sup_strength,
                "date": support_date
            }

        # Return strategy data (levels are already filtered by touch_count in strategy)
        response = {
            "stock_data": stock_dict,
            "trades": strat.trades if strat is not None else [],
            "levels": strat.levels if strat is not None else [],
            "resistance_levels": strat.resistance_levels if strat is not None else [],
            "support_levels": strat.support_levels if strat is not None else [],
            "resistance_clusters": {str(k): len(v) for k, v in strat.resistance_clusters.items()} if strat is not None else {},
            "support_clusters": {str(k): len(v) for k, v in strat.support_clusters.items()} if strat is not None else {},
            "resistance_swings": strat.resistance_swings if strat is not None else [],
            "support_swings": strat.support_swings if strat is not None else [],
            "absolute_resistance": absolute_resistance,
            "absolute_support": absolute_support
        }

        return jsonify(response)

    except Exception as e:
        logger.error(f"Error fetching stock data: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": "Failed to fetch stock data"}), 500


@app.route('/api/backtest/run', methods=['POST'])
def run_backtest():
    """Run a backtest with the given configuration."""
    try:
        data = request.get_json()

        if not data:
            return jsonify({"error": "No configuration provided"}), 400

        # Extract configuration
        symbol = data.get('symbol', 'AAPL').upper()
        start_date = data.get('start_date')
        end_date = data.get('end_date')
        initial_capital = float(data.get('initial_capital', 100000))
        commission = float(data.get('commission', 0.001))  # 0.1% default
        strategy_id = data.get('strategy', 'SupResStrategy')
        strategy_params = data.get('strategy_params', {})

        # Validate symbol
        if not validate_symbol(symbol):
            return jsonify({"error": "Invalid stock symbol format"}), 400

        # Fetch stock data with date range
        stock = yf.Ticker(symbol)

        if start_date and end_date:
            stock_data = stock.history(start=start_date, end=end_date)
        else:
            stock_data = stock.history(period='2y')

        if stock_data.empty:
            return jsonify({"error": f"No data found for symbol: {symbol}"}), 404

        # Prepare data for Backtrader
        bt_dataframe = stock_data.copy()

        # Create Backtrader cerebro
        cerebro = bt.Cerebro()

        # Add data feed
        bt_data = bt.feeds.PandasData(
            dataname=bt_dataframe,
            open='Open',
            high='High',
            low='Low',
            close='Close',
            volume='Volume',
            openinterest=None
        )
        cerebro.adddata(bt_data)

        # Add strategy with parameters
        cerebro.addstrategy(
            BacktestStrategy,
            FIRST_RETRACEMENT=strategy_params.get('FIRST_RETRACEMENT', 5),
            SECOND_RETRACEMENT=strategy_params.get('SECOND_RETRACEMENT', 5),
            RES_SUP_RANGE=strategy_params.get('RES_SUP_RANGE', 0.001),
            TOUCH_COUNT=strategy_params.get('TOUCH_COUNT', 1),
            TAKE_PROFIT=strategy_params.get('TAKE_PROFIT', 10),
            STOP_LOSS=strategy_params.get('STOP_LOSS', 5),
            BREAKOUT_BUFFER=strategy_params.get('BREAKOUT_BUFFER', 0),
        )

        # Set broker settings
        cerebro.broker.set_cash(initial_capital)
        cerebro.broker.setcommission(commission=commission)

        # Add analyzers
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', riskfreerate=0.0)

        # Run backtest
        results = cerebro.run()
        strat = results[0]

        # Get analyzer results
        trade_analysis = strat.analyzers.trades.get_analysis()
        drawdown_analysis = strat.analyzers.drawdown.get_analysis()
        sharpe_analysis = strat.analyzers.sharpe.get_analysis()

        # Calculate statistics
        final_value = cerebro.broker.getvalue()
        net_profit = final_value - initial_capital
        net_profit_percent = (net_profit / initial_capital) * 100

        # Extract trade stats
        total_trades = trade_analysis.get('total', {}).get('total', 0)
        won_trades = trade_analysis.get('won', {}).get('total', 0)
        lost_trades = trade_analysis.get('lost', {}).get('total', 0)

        gross_profit = trade_analysis.get('won', {}).get('pnl', {}).get('total', 0)
        gross_loss = abs(trade_analysis.get('lost', {}).get('pnl', {}).get('total', 0))

        win_rate = (won_trades / total_trades * 100) if total_trades > 0 else 0
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float('inf')

        avg_win = (gross_profit / won_trades) if won_trades > 0 else 0
        avg_loss = (gross_loss / lost_trades) if lost_trades > 0 else 0

        largest_win = trade_analysis.get('won', {}).get('pnl', {}).get('max', 0)
        largest_loss = abs(trade_analysis.get('lost', {}).get('pnl', {}).get('max', 0))

        max_drawdown = drawdown_analysis.get('max', {}).get('drawdown', 0)
        max_drawdown_value = drawdown_analysis.get('max', {}).get('moneydown', 0)

        sharpe_ratio = sharpe_analysis.get('sharperatio') or 0

        statistics = {
            'net_profit': round(net_profit, 2),
            'net_profit_percent': round(net_profit_percent, 2),
            'gross_profit': round(gross_profit, 2),
            'gross_loss': round(gross_loss, 2),
            'profit_factor': round(profit_factor, 2) if profit_factor != float('inf') else 'Infinity',
            'total_trades': total_trades,
            'winning_trades': won_trades,
            'losing_trades': lost_trades,
            'win_rate': round(win_rate, 2),
            'average_win': round(avg_win, 2),
            'average_loss': round(avg_loss, 2),
            'largest_win': round(largest_win, 2),
            'largest_loss': round(largest_loss, 2),
            'max_drawdown': round(max_drawdown_value, 2),
            'max_drawdown_percent': round(max_drawdown, 2),
            'sharpe_ratio': round(sharpe_ratio, 2) if sharpe_ratio else 0,
            'initial_capital': initial_capital,
            'final_value': round(final_value, 2),
        }

        # Prepare stock data for chart
        stock_data.index = stock_data.index.strftime("%Y-%m-%d")
        stock_data_list = [
            {
                'date': date,
                'open': row['Open'],
                'high': row['High'],
                'low': row['Low'],
                'close': row['Close'],
                'volume': row['Volume'],
            }
            for date, row in stock_data.iterrows()
        ]

        # Calculate enhanced risk metrics
        risk_metrics = calculate_risk_metrics(
            strat.equity_curve,
            strat.detailed_trades,
            initial_capital
        )

        response = {
            'status': 'completed',
            'statistics': statistics,
            'risk_metrics': risk_metrics,
            'trades': strat.detailed_trades,
            'equity_curve': strat.equity_curve,
            'stock_data': stock_data_list,
        }

        return jsonify(response)

    except Exception as e:
        logger.error(f"Backtest error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/api/strategies', methods=['GET'])
def get_strategies():
    """Return available strategies and their parameters."""
    strategies = [
        {
            'id': 'SupResStrategy',
            'name': 'Support/Resistance Breakout',
            'description': 'Trades breakouts above resistance and breakdowns below support levels',
            'parameters': [
                {'name': 'FIRST_RETRACEMENT', 'type': 'float', 'default': 5, 'min': 1, 'max': 50, 'label': 'First Retracement %'},
                {'name': 'SECOND_RETRACEMENT', 'type': 'float', 'default': 5, 'min': 1, 'max': 50, 'label': 'Second Retracement %'},
                {'name': 'RES_SUP_RANGE', 'type': 'float', 'default': 0.001, 'min': 0.0001, 'max': 0.1, 'label': 'Level Range'},
                {'name': 'TOUCH_COUNT', 'type': 'int', 'default': 1, 'min': 1, 'max': 10, 'label': 'Touch Count'},
                {'name': 'BREAKOUT_BUFFER', 'type': 'float', 'default': 0, 'min': 0, 'max': 20, 'label': 'Breakout Buffer %'},
                {'name': 'TAKE_PROFIT', 'type': 'float', 'default': 10, 'min': 1, 'max': 100, 'label': 'Take Profit %'},
                {'name': 'STOP_LOSS', 'type': 'float', 'default': 5, 'min': 1, 'max': 50, 'label': 'Stop Loss %'},
            ]
        }
    ]
    return jsonify({'strategies': strategies})


# ============================================
# STOCK SEARCH API
# ============================================

# Popular stocks cache for quick suggestions
POPULAR_STOCKS = [
    {"symbol": "AAPL", "name": "Apple Inc.", "exchange": "NASDAQ"},
    {"symbol": "MSFT", "name": "Microsoft Corporation", "exchange": "NASDAQ"},
    {"symbol": "GOOGL", "name": "Alphabet Inc.", "exchange": "NASDAQ"},
    {"symbol": "GOOG", "name": "Alphabet Inc. Class C", "exchange": "NASDAQ"},
    {"symbol": "AMZN", "name": "Amazon.com Inc.", "exchange": "NASDAQ"},
    {"symbol": "NVDA", "name": "NVIDIA Corporation", "exchange": "NASDAQ"},
    {"symbol": "META", "name": "Meta Platforms Inc.", "exchange": "NASDAQ"},
    {"symbol": "TSLA", "name": "Tesla Inc.", "exchange": "NASDAQ"},
    {"symbol": "BRK-B", "name": "Berkshire Hathaway Inc.", "exchange": "NYSE"},
    {"symbol": "JPM", "name": "JPMorgan Chase & Co.", "exchange": "NYSE"},
    {"symbol": "V", "name": "Visa Inc.", "exchange": "NYSE"},
    {"symbol": "JNJ", "name": "Johnson & Johnson", "exchange": "NYSE"},
    {"symbol": "WMT", "name": "Walmart Inc.", "exchange": "NYSE"},
    {"symbol": "PG", "name": "Procter & Gamble Co.", "exchange": "NYSE"},
    {"symbol": "MA", "name": "Mastercard Inc.", "exchange": "NYSE"},
    {"symbol": "UNH", "name": "UnitedHealth Group Inc.", "exchange": "NYSE"},
    {"symbol": "HD", "name": "Home Depot Inc.", "exchange": "NYSE"},
    {"symbol": "DIS", "name": "Walt Disney Co.", "exchange": "NYSE"},
    {"symbol": "BAC", "name": "Bank of America Corp.", "exchange": "NYSE"},
    {"symbol": "XOM", "name": "Exxon Mobil Corporation", "exchange": "NYSE"},
    {"symbol": "NFLX", "name": "Netflix Inc.", "exchange": "NASDAQ"},
    {"symbol": "ADBE", "name": "Adobe Inc.", "exchange": "NASDAQ"},
    {"symbol": "CRM", "name": "Salesforce Inc.", "exchange": "NYSE"},
    {"symbol": "AMD", "name": "Advanced Micro Devices", "exchange": "NASDAQ"},
    {"symbol": "INTC", "name": "Intel Corporation", "exchange": "NASDAQ"},
    {"symbol": "CSCO", "name": "Cisco Systems Inc.", "exchange": "NASDAQ"},
    {"symbol": "ORCL", "name": "Oracle Corporation", "exchange": "NYSE"},
    {"symbol": "IBM", "name": "International Business Machines", "exchange": "NYSE"},
    {"symbol": "QCOM", "name": "Qualcomm Inc.", "exchange": "NASDAQ"},
    {"symbol": "AVGO", "name": "Broadcom Inc.", "exchange": "NASDAQ"},
    {"symbol": "TXN", "name": "Texas Instruments Inc.", "exchange": "NASDAQ"},
    {"symbol": "NOW", "name": "ServiceNow Inc.", "exchange": "NYSE"},
    {"symbol": "PYPL", "name": "PayPal Holdings Inc.", "exchange": "NASDAQ"},
    {"symbol": "SQ", "name": "Block Inc.", "exchange": "NYSE"},
    {"symbol": "SHOP", "name": "Shopify Inc.", "exchange": "NYSE"},
    {"symbol": "UBER", "name": "Uber Technologies Inc.", "exchange": "NYSE"},
    {"symbol": "ABNB", "name": "Airbnb Inc.", "exchange": "NASDAQ"},
    {"symbol": "COIN", "name": "Coinbase Global Inc.", "exchange": "NASDAQ"},
    {"symbol": "PLTR", "name": "Palantir Technologies", "exchange": "NYSE"},
    {"symbol": "SNOW", "name": "Snowflake Inc.", "exchange": "NYSE"},
    {"symbol": "SPY", "name": "SPDR S&P 500 ETF Trust", "exchange": "NYSE"},
    {"symbol": "QQQ", "name": "Invesco QQQ Trust", "exchange": "NASDAQ"},
    {"symbol": "IWM", "name": "iShares Russell 2000 ETF", "exchange": "NYSE"},
    {"symbol": "DIA", "name": "SPDR Dow Jones Industrial Average", "exchange": "NYSE"},
    {"symbol": "VTI", "name": "Vanguard Total Stock Market ETF", "exchange": "NYSE"},
    {"symbol": "GLD", "name": "SPDR Gold Shares", "exchange": "NYSE"},
    {"symbol": "SLV", "name": "iShares Silver Trust", "exchange": "NYSE"},
    {"symbol": "USO", "name": "United States Oil Fund", "exchange": "NYSE"},
    {"symbol": "BA", "name": "Boeing Company", "exchange": "NYSE"},
    {"symbol": "GS", "name": "Goldman Sachs Group Inc.", "exchange": "NYSE"},
]

# Cache for stock info lookups
stock_info_cache = {}
STOCK_INFO_CACHE_DURATION = 86400  # 24 hours in seconds


@app.route('/api/stocks/search', methods=['GET'])
def search_stocks():
    """Search for stocks by symbol or name."""
    query = request.args.get('q', '').strip().upper()
    limit = min(int(request.args.get('limit', 10)), 50)

    if not query:
        # Return popular stocks when no query
        return jsonify({'results': POPULAR_STOCKS[:limit]})

    results = []

    # First, search in popular stocks (fast)
    for stock in POPULAR_STOCKS:
        if query in stock['symbol'] or query.lower() in stock['name'].lower():
            results.append(stock)
            if len(results) >= limit:
                break

    # If we don't have enough results, try yfinance lookup
    if len(results) < limit and len(query) >= 1:
        try:
            # Check if the exact symbol exists
            if query not in [r['symbol'] for r in results]:
                ticker = yf.Ticker(query)
                info = ticker.info
                if info and info.get('symbol'):
                    results.insert(0, {
                        'symbol': info.get('symbol', query),
                        'name': info.get('shortName') or info.get('longName') or query,
                        'exchange': info.get('exchange', 'Unknown'),
                        'type': info.get('quoteType', 'EQUITY'),
                    })
        except Exception as e:
            logger.debug(f"yfinance lookup failed for {query}: {e}")

    return jsonify({'results': results[:limit]})


@app.route('/api/stocks/validate', methods=['GET'])
def validate_stock():
    """Validate a stock symbol and return basic info."""
    symbol = request.args.get('symbol', '').strip().upper()

    if not symbol:
        return jsonify({'valid': False, 'error': 'No symbol provided'}), 400

    # Check cache first
    cache_key = f"validate_{symbol}"
    now = time.time()
    if cache_key in stock_info_cache:
        cached = stock_info_cache[cache_key]
        if now - cached['timestamp'] < STOCK_INFO_CACHE_DURATION:
            return jsonify(cached['data'])

    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period='5d')

        if hist.empty:
            return jsonify({'valid': False, 'error': f'No data found for {symbol}'})

        info = ticker.info or {}
        result = {
            'valid': True,
            'symbol': symbol,
            'name': info.get('shortName') or info.get('longName') or symbol,
            'exchange': info.get('exchange', 'Unknown'),
            'type': info.get('quoteType', 'EQUITY'),
            'currency': info.get('currency', 'USD'),
            'currentPrice': info.get('currentPrice') or info.get('regularMarketPrice') or float(hist['Close'].iloc[-1]),
        }

        # Cache the result
        stock_info_cache[cache_key] = {'data': result, 'timestamp': now}

        return jsonify(result)

    except Exception as e:
        logger.error(f"Error validating stock {symbol}: {e}")
        return jsonify({'valid': False, 'error': str(e)})


# ============================================
# ENHANCED RISK METRICS
# ============================================

def calculate_risk_metrics(equity_curve, trades, initial_capital, risk_free_rate=0.02):
    """
    Calculate comprehensive risk metrics for a backtest.

    Returns:
        dict: Dictionary containing all risk metrics
    """
    if not equity_curve or len(equity_curve) < 2:
        return get_empty_risk_metrics()

    # Extract equity values
    equity_values = [e.get('equity', e.get('balance', initial_capital)) for e in equity_curve]
    dates = [e.get('date') for e in equity_curve]

    # Calculate daily returns
    returns = []
    for i in range(1, len(equity_values)):
        if equity_values[i-1] > 0:
            daily_return = (equity_values[i] - equity_values[i-1]) / equity_values[i-1]
            returns.append(daily_return)

    if not returns:
        return get_empty_risk_metrics()

    returns = np.array(returns)

    # Basic statistics
    total_return = (equity_values[-1] - initial_capital) / initial_capital
    trading_days = len(returns)
    annualization_factor = 252  # Trading days per year

    # Mean and standard deviation (annualized)
    mean_return = np.mean(returns) * annualization_factor
    std_return = np.std(returns, ddof=1) * np.sqrt(annualization_factor) if len(returns) > 1 else 0

    # Sharpe Ratio (annualized)
    sharpe_ratio = (mean_return - risk_free_rate) / std_return if std_return > 0 else 0

    # Sortino Ratio (uses downside deviation)
    negative_returns = returns[returns < 0]
    downside_std = np.std(negative_returns, ddof=1) * np.sqrt(annualization_factor) if len(negative_returns) > 1 else 0
    sortino_ratio = (mean_return - risk_free_rate) / downside_std if downside_std > 0 else 0

    # Maximum Drawdown
    peak = equity_values[0]
    max_drawdown = 0
    max_drawdown_duration = 0
    current_drawdown_start = 0
    drawdown_periods = []

    for i, value in enumerate(equity_values):
        if value > peak:
            if current_drawdown_start > 0:
                drawdown_periods.append(i - current_drawdown_start)
            peak = value
            current_drawdown_start = i
        drawdown = (peak - value) / peak
        if drawdown > max_drawdown:
            max_drawdown = drawdown

    max_drawdown_percent = max_drawdown * 100

    # Calmar Ratio (annualized return / max drawdown)
    annualized_return = ((equity_values[-1] / initial_capital) ** (annualization_factor / trading_days) - 1) if trading_days > 0 else 0
    calmar_ratio = annualized_return / max_drawdown if max_drawdown > 0 else 0

    # Value at Risk (VaR) - 95% confidence
    var_95 = np.percentile(returns, 5) * initial_capital if len(returns) > 0 else 0
    var_99 = np.percentile(returns, 1) * initial_capital if len(returns) > 0 else 0

    # Conditional VaR (CVaR) / Expected Shortfall
    var_95_threshold = np.percentile(returns, 5)
    cvar_95 = np.mean(returns[returns <= var_95_threshold]) * initial_capital if len(returns[returns <= var_95_threshold]) > 0 else 0

    # Volatility metrics
    daily_volatility = np.std(returns, ddof=1) if len(returns) > 1 else 0
    annualized_volatility = daily_volatility * np.sqrt(annualization_factor)

    # Win/Loss analysis from trades
    if trades:
        winning_trades = [t for t in trades if t.get('pnl', 0) > 0]
        losing_trades = [t for t in trades if t.get('pnl', 0) <= 0]

        win_rate = len(winning_trades) / len(trades) * 100 if trades else 0

        avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
        avg_loss = abs(np.mean([t['pnl'] for t in losing_trades])) if losing_trades else 0

        # Profit factor
        gross_profit = sum(t['pnl'] for t in winning_trades)
        gross_loss = abs(sum(t['pnl'] for t in losing_trades))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # Expected value per trade
        expected_value = np.mean([t.get('pnl', 0) for t in trades]) if trades else 0

        # Risk-Reward Ratio
        risk_reward = avg_win / avg_loss if avg_loss > 0 else 0

        # Consecutive wins/losses
        max_consecutive_wins = 0
        max_consecutive_losses = 0
        current_wins = 0
        current_losses = 0

        for trade in trades:
            if trade.get('pnl', 0) > 0:
                current_wins += 1
                current_losses = 0
                max_consecutive_wins = max(max_consecutive_wins, current_wins)
            else:
                current_losses += 1
                current_wins = 0
                max_consecutive_losses = max(max_consecutive_losses, current_losses)
    else:
        win_rate = 0
        avg_win = 0
        avg_loss = 0
        profit_factor = 0
        expected_value = 0
        risk_reward = 0
        max_consecutive_wins = 0
        max_consecutive_losses = 0

    # Recovery factor
    recovery_factor = total_return / max_drawdown if max_drawdown > 0 else 0

    # Position sizing recommendations
    kelly_criterion = 0
    if win_rate > 0 and risk_reward > 0:
        win_prob = win_rate / 100
        kelly_criterion = (win_prob * risk_reward - (1 - win_prob)) / risk_reward if risk_reward > 0 else 0
        kelly_criterion = max(0, min(kelly_criterion, 1))  # Clamp between 0 and 1

    # Half-Kelly (more conservative)
    half_kelly = kelly_criterion / 2

    # Optimal position size based on VaR
    var_position_size = 0.02 / abs(var_95 / initial_capital) if var_95 != 0 else 0.05  # Target 2% VaR
    var_position_size = max(0.01, min(var_position_size, 0.25))  # Clamp between 1% and 25%

    return {
        # Return metrics
        'total_return': round(total_return * 100, 2),
        'annualized_return': round(annualized_return * 100, 2),

        # Risk-adjusted returns
        'sharpe_ratio': round(sharpe_ratio, 2),
        'sortino_ratio': round(sortino_ratio, 2),
        'calmar_ratio': round(calmar_ratio, 2),

        # Volatility metrics
        'daily_volatility': round(daily_volatility * 100, 2),
        'annualized_volatility': round(annualized_volatility * 100, 2),

        # Drawdown metrics
        'max_drawdown_percent': round(max_drawdown_percent, 2),
        'recovery_factor': round(recovery_factor, 2),

        # VaR metrics
        'var_95': round(abs(var_95), 2),
        'var_99': round(abs(var_99), 2),
        'cvar_95': round(abs(cvar_95), 2),

        # Trade metrics
        'win_rate': round(win_rate, 2),
        'profit_factor': round(profit_factor, 2) if profit_factor != float('inf') else 'Infinity',
        'risk_reward_ratio': round(risk_reward, 2),
        'expected_value': round(expected_value, 2),
        'max_consecutive_wins': max_consecutive_wins,
        'max_consecutive_losses': max_consecutive_losses,

        # Position sizing recommendations
        'kelly_criterion': round(kelly_criterion * 100, 2),
        'half_kelly': round(half_kelly * 100, 2),
        'var_based_position': round(var_position_size * 100, 2),

        # Metadata
        'trading_days': trading_days,
    }


def get_empty_risk_metrics():
    """Return empty risk metrics structure."""
    return {
        'total_return': 0,
        'annualized_return': 0,
        'sharpe_ratio': 0,
        'sortino_ratio': 0,
        'calmar_ratio': 0,
        'daily_volatility': 0,
        'annualized_volatility': 0,
        'max_drawdown_percent': 0,
        'recovery_factor': 0,
        'var_95': 0,
        'var_99': 0,
        'cvar_95': 0,
        'win_rate': 0,
        'profit_factor': 0,
        'risk_reward_ratio': 0,
        'expected_value': 0,
        'max_consecutive_wins': 0,
        'max_consecutive_losses': 0,
        'kelly_criterion': 0,
        'half_kelly': 0,
        'var_based_position': 5,
        'trading_days': 0,
    }


class BacktestStrategy(bt.Strategy):
    """
    Extended Support/Resistance Strategy with detailed tracking for backtesting.
    Tracks equity curve, detailed trades, and comprehensive statistics.
    """
    params = (
        ("FIRST_RETRACEMENT", 5),
        ("SECOND_RETRACEMENT", 5),
        ("RES_SUP_RANGE", 0.001),
        ("TOUCH_COUNT", 1),
        ("TAKE_PROFIT", 10),
        ("STOP_LOSS", 5),
        ("RISK_PERCENTAGE", 5),
        ("BREAKOUT_BUFFER", 0),  # % above resistance to trigger buy (0 = buy at breakout)
    )

    def __init__(self):
        # Level tracking
        self.resistance_levels = []
        self.support_levels = []
        self.resistance_clusters = {}
        self.support_clusters = {}

        # Sequential bar tracking for resistance
        self.start_high_bar = {'date': None, 'high': 0, 'low': float('inf')}
        self.middle_high_bar = {'date': None, 'high': 0, 'low': 0}

        # Sequential bar tracking for support
        self.start_low_bar = {'date': None, 'high': 0, 'low': float('inf')}
        self.middle_low_bar = {'date': None, 'high': 0, 'low': float('inf')}

        # Order tracking
        self.order = None
        self.buyprice = None
        self.buydate = None
        self.buycomm = None

        # Track broken levels
        self.broken_resistances = set()
        self.broken_supports = set()

        # Backtest specific tracking
        self.equity_curve = []
        self.detailed_trades = []
        self.current_trade_id = 0
        self.open_trade = None

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status in [order.Completed]:
            dt = self.datas[0].datetime.date(0).isoformat()
            if order.isbuy():
                self.buyprice = order.executed.price
                self.buydate = dt
                self.buycomm = order.executed.comm
                self.current_trade_id += 1
                self.open_trade = {
                    'id': self.current_trade_id,
                    'type': 'long',
                    'entry_date': dt,
                    'entry_price': order.executed.price,
                    'size': order.executed.size,
                    'commission': order.executed.comm,
                }
            elif order.issell():
                if self.open_trade:
                    pnl = (order.executed.price - self.open_trade['entry_price']) * self.open_trade['size']
                    pnl -= self.open_trade['commission'] + order.executed.comm
                    pnl_percent = ((order.executed.price - self.open_trade['entry_price']) / self.open_trade['entry_price']) * 100

                    # Determine exit reason
                    exit_reason = 'signal'
                    if self.buyprice:
                        if order.executed.price >= self.buyprice * (1 + self.params.TAKE_PROFIT / 100):
                            exit_reason = 'take_profit'
                        elif order.executed.price <= self.buyprice * (1 - self.params.STOP_LOSS / 100):
                            exit_reason = 'stop_loss'

                    self.detailed_trades.append({
                        'id': self.open_trade['id'],
                        'type': self.open_trade['type'],
                        'entry_date': self.open_trade['entry_date'],
                        'entry_price': round(self.open_trade['entry_price'], 2),
                        'exit_date': dt,
                        'exit_price': round(order.executed.price, 2),
                        'size': int(self.open_trade['size']),
                        'pnl': round(pnl, 2),
                        'pnl_percent': round(pnl_percent, 2),
                        'commission': round(self.open_trade['commission'] + order.executed.comm, 2),
                        'exit_reason': exit_reason,
                    })
                    self.open_trade = None

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            pass

        self.order = None

    def add_to_cluster(self, clusters, level, range_pct):
        """Add level to cluster and return the cluster key and count."""
        found_cluster = None
        for base in list(clusters.keys()):
            high_range = base * (1 + range_pct)
            low_range = base * (1 - range_pct)
            if low_range <= level <= high_range:
                found_cluster = base
                break

        if found_cluster:
            clusters[found_cluster].append(level)
            return found_cluster, len(clusters[found_cluster])
        else:
            clusters[level] = [level]
            return level, 1

    def calculating_top(self):
        """Detect resistance levels using sequential bar tracking."""
        dt = self.datas[0].datetime.date(0)
        current_high = self.data.high[0]
        current_low = self.data.low[0]

        middle_updated_this_bar = False

        if current_low < self.start_high_bar['low']:
            self.start_high_bar = {'date': dt, 'high': current_high, 'low': current_low}
            self.middle_high_bar = {'date': dt, 'high': current_high, 'low': current_low}
            middle_updated_this_bar = True

        if current_high > self.middle_high_bar['high']:
            self.middle_high_bar = {'date': dt, 'high': current_high, 'low': current_low}
            middle_updated_this_bar = True

        if middle_updated_this_bar:
            return

        first_retracement = calculate_percentage_change(
            self.start_high_bar['low'],
            self.middle_high_bar['high']
        )

        if first_retracement >= self.params.FIRST_RETRACEMENT:
            second_retracement = abs(calculate_percentage_change(
                self.middle_high_bar['high'],
                current_low
            ))

            if second_retracement >= self.params.SECOND_RETRACEMENT:
                resistance_level = round(self.middle_high_bar['high'], 2)

                cluster_key, touch_count = self.add_to_cluster(
                    self.resistance_clusters,
                    resistance_level,
                    self.params.RES_SUP_RANGE
                )

                if touch_count >= self.params.TOUCH_COUNT:
                    max_resistance = max(self.resistance_clusters[cluster_key])
                    if max_resistance not in [l['price'] for l in self.resistance_levels]:
                        self.resistance_levels.append({'price': max_resistance})

                self.start_high_bar = {'date': dt, 'high': current_high, 'low': current_low}
                self.middle_high_bar = {'date': dt, 'high': current_high, 'low': current_low}

    def calculating_bottom(self):
        """Detect support levels using sequential bar tracking."""
        dt = self.datas[0].datetime.date(0)
        current_high = self.data.high[0]
        current_low = self.data.low[0]

        middle_updated_this_bar = False

        if current_high > self.start_low_bar['high']:
            self.start_low_bar = {'date': dt, 'high': current_high, 'low': current_low}
            self.middle_low_bar = {'date': dt, 'high': current_high, 'low': current_low}
            middle_updated_this_bar = True

        if current_low < self.middle_low_bar['low']:
            self.middle_low_bar = {'date': dt, 'high': current_high, 'low': current_low}
            middle_updated_this_bar = True

        if middle_updated_this_bar:
            return

        first_retracement = abs(calculate_percentage_change(
            self.start_low_bar['high'],
            self.middle_low_bar['low']
        ))

        if first_retracement >= self.params.FIRST_RETRACEMENT:
            second_retracement = calculate_percentage_change(
                self.middle_low_bar['low'],
                current_high
            )

            if second_retracement >= self.params.SECOND_RETRACEMENT:
                support_level = round(self.middle_low_bar['low'], 2)

                cluster_key, touch_count = self.add_to_cluster(
                    self.support_clusters,
                    support_level,
                    self.params.RES_SUP_RANGE
                )

                if touch_count >= self.params.TOUCH_COUNT:
                    min_support = min(self.support_clusters[cluster_key])
                    if min_support not in [l['price'] for l in self.support_levels]:
                        self.support_levels.append({'price': min_support})

                self.start_low_bar = {'date': dt, 'high': current_high, 'low': current_low}
                self.middle_low_bar = {'date': dt, 'high': current_high, 'low': current_low}

    def next(self):
        dt = self.datas[0].datetime.date(0).isoformat()
        current_close = self.data.close[0]

        # Record equity curve
        self.equity_curve.append({
            'date': dt,
            'balance': round(self.broker.get_cash(), 2),
            'equity': round(self.broker.get_value(), 2),
        })

        # Swing detection
        self.calculating_top()
        self.calculating_bottom()

        # Trading logic
        if self.order:
            return

        if not self.position:
            prev_close = self.data.close[-1] if len(self) > 1 else current_close

            resistance_prices = sorted([l['price'] for l in self.resistance_levels])
            for resistance in resistance_prices:
                # Calculate entry price with buffer: resistance + (buffer% of resistance)
                entry_trigger = resistance * (1 + self.params.BREAKOUT_BUFFER / 100)

                if prev_close <= entry_trigger < current_close and resistance not in self.broken_resistances:
                    cash = self.broker.get_cash()
                    size = int((cash * self.params.RISK_PERCENTAGE / 100) / current_close)
                    if size < 1:
                        size = 1
                    self.order = self.buy(size=size)
                    self.broken_resistances.add(resistance)
                    break
        else:
            if self.position.size > 0:
                if self.buyprice and current_close >= self.buyprice * (1 + self.params.TAKE_PROFIT / 100):
                    self.order = self.close()
                elif self.buyprice and current_close <= self.buyprice * (1 - self.params.STOP_LOSS / 100):
                    self.order = self.close()
                elif self.support_levels:
                    support_prices = [l['price'] for l in self.support_levels]
                    for support in support_prices:
                        if current_close < support and support not in self.broken_supports:
                            self.order = self.close()
                            self.broken_supports.add(support)
                            break

    def stop(self):
        """Called when backtest ends - close any open positions."""
        if self.open_trade:
            dt = self.datas[0].datetime.date(0).isoformat()
            current_price = self.data.close[0]
            pnl = (current_price - self.open_trade['entry_price']) * self.open_trade['size']
            pnl -= self.open_trade['commission']
            pnl_percent = ((current_price - self.open_trade['entry_price']) / self.open_trade['entry_price']) * 100

            self.detailed_trades.append({
                'id': self.open_trade['id'],
                'type': self.open_trade['type'],
                'entry_date': self.open_trade['entry_date'],
                'entry_price': round(self.open_trade['entry_price'], 2),
                'exit_date': dt,
                'exit_price': round(current_price, 2),
                'size': int(self.open_trade['size']),
                'pnl': round(pnl, 2),
                'pnl_percent': round(pnl_percent, 2),
                'commission': round(self.open_trade['commission'], 2),
                'exit_reason': 'end_of_data',
            })


# Visual Backtesting State Storage
visual_backtests = {}


@socketio.on('connect')
def handle_connect():
    logger.info(f"Client connected: {request.sid}")
    emit('connected', {'status': 'connected'})


@socketio.on('disconnect')
def handle_disconnect():
    logger.info(f"Client disconnected: {request.sid}")
    # Clean up any running backtests for this client
    to_remove = [bid for bid, data in visual_backtests.items() if data.get('sid') == request.sid]
    for bid in to_remove:
        visual_backtests[bid]['is_running'] = False
        del visual_backtests[bid]


@socketio.on('start_visual_backtest')
def handle_start_visual_backtest(config):
    """Initialize a visual backtest session."""
    try:
        backtest_id = str(uuid.uuid4())
        symbol = config.get('symbol', 'AAPL').upper()
        start_date = config.get('start_date')
        end_date = config.get('end_date')

        # Fetch stock data
        stock = yf.Ticker(symbol)
        if start_date and end_date:
            stock_data = stock.history(start=start_date, end=end_date)
        else:
            stock_data = stock.history(period='2y')

        if stock_data.empty:
            emit('error', {'message': f'No data found for symbol: {symbol}'})
            return

        # Convert to list of candles
        stock_data.index = stock_data.index.strftime("%Y-%m-%d")
        candles = [
            {
                'date': date,
                'open': float(row['Open']),
                'high': float(row['High']),
                'low': float(row['Low']),
                'close': float(row['Close']),
                'volume': int(row['Volume']),
            }
            for date, row in stock_data.iterrows()
        ]

        # Store backtest state
        visual_backtests[backtest_id] = {
            'sid': request.sid,
            'config': config,
            'candles': candles,
            'current_index': 0,
            'is_playing': False,
            'speed': 1,
            'is_running': True,
            'equity_curve': [],
            'trades': [],
            'open_trade': None,
            'balance': float(config.get('initial_capital', 100000)),
            'equity': float(config.get('initial_capital', 100000)),
            'resistance_levels': [],
            'support_levels': [],
            'resistance_clusters': {},
            'support_clusters': {},
            'broken_resistances': set(),
            'broken_supports': set(),
            'strategy_params': config.get('strategy_params', {}),
        }

        emit('backtest_initialized', {
            'backtest_id': backtest_id,
            'total_bars': len(candles),
            'status': 'ready',
        })

    except Exception as e:
        logger.error(f"Error starting visual backtest: {str(e)}")
        emit('error', {'message': str(e)})


@socketio.on('control')
def handle_control(data):
    """Handle playback control commands."""
    backtest_id = data.get('backtest_id')
    action = data.get('action')

    if backtest_id not in visual_backtests:
        emit('error', {'message': 'Backtest not found'})
        return

    bt_state = visual_backtests[backtest_id]

    if action == 'play':
        bt_state['is_playing'] = True
        # Start playback in a background thread
        threading.Thread(target=run_visual_playback, args=(backtest_id, request.sid)).start()

    elif action == 'pause':
        bt_state['is_playing'] = False

    elif action == 'step':
        bt_state['is_playing'] = False
        process_next_bar(backtest_id, request.sid)

    elif action == 'set_speed':
        bt_state['speed'] = data.get('value', 1)

    elif action == 'jump_to':
        index = data.get('index', 0)
        bt_state['current_index'] = max(0, min(index, len(bt_state['candles']) - 1))
        emit('bar_update', get_bar_update(backtest_id))


def run_visual_playback(backtest_id, sid):
    """Run visual backtest playback in a background thread."""
    while backtest_id in visual_backtests and visual_backtests[backtest_id]['is_running']:
        bt_state = visual_backtests[backtest_id]

        if not bt_state['is_playing']:
            time.sleep(0.1)
            continue

        if bt_state['current_index'] >= len(bt_state['candles']):
            # Backtest complete
            socketio.emit('backtest_complete', {
                'statistics': calculate_visual_statistics(backtest_id),
                'trades': bt_state['trades'],
                'equity_curve': bt_state['equity_curve'],
            }, room=sid)
            bt_state['is_playing'] = False
            break

        process_next_bar(backtest_id, sid)

        # Delay based on speed
        delay = 0.5 / bt_state['speed']
        time.sleep(delay)


def process_next_bar(backtest_id, sid):
    """Process the next bar in visual backtest."""
    if backtest_id not in visual_backtests:
        return

    bt_state = visual_backtests[backtest_id]
    candles = bt_state['candles']
    index = bt_state['current_index']

    if index >= len(candles):
        return

    candle = candles[index]
    params = bt_state['strategy_params']

    # Simple swing detection for visual mode
    if index >= 2:
        detect_visual_levels(bt_state, candles, index, params)

    # Check for trade signals
    check_visual_trades(bt_state, candle, params)

    # Update equity curve
    bt_state['equity_curve'].append({
        'date': candle['date'],
        'balance': round(bt_state['balance'], 2),
        'equity': round(bt_state['equity'], 2),
    })

    bt_state['current_index'] = index + 1

    # Emit update
    socketio.emit('bar_update', get_bar_update(backtest_id), room=sid)


def detect_visual_levels(bt_state, candles, index, params):
    """Simplified level detection for visual mode."""
    first_ret = params.get('FIRST_RETRACEMENT', 5)
    second_ret = params.get('SECOND_RETRACEMENT', 5)
    range_pct = params.get('RES_SUP_RANGE', 0.001)

    # Look back for swing highs (resistance)
    if index >= 3:
        c1, c2, c3 = candles[index-2], candles[index-1], candles[index]

        # Check for swing high
        if c2['high'] > c1['high'] and c2['high'] > c3['high']:
            move_up = ((c2['high'] - c1['low']) / c1['low']) * 100 if c1['low'] > 0 else 0
            pullback = ((c2['high'] - c3['low']) / c2['high']) * 100 if c2['high'] > 0 else 0

            if move_up >= first_ret and pullback >= second_ret:
                level = round(c2['high'], 2)
                add_visual_level(bt_state, 'resistance', level, range_pct)

        # Check for swing low
        if c2['low'] < c1['low'] and c2['low'] < c3['low']:
            move_down = ((c1['high'] - c2['low']) / c1['high']) * 100 if c1['high'] > 0 else 0
            bounce = ((c3['high'] - c2['low']) / c2['low']) * 100 if c2['low'] > 0 else 0

            if move_down >= first_ret and bounce >= second_ret:
                level = round(c2['low'], 2)
                add_visual_level(bt_state, 'support', level, range_pct)


def add_visual_level(bt_state, level_type, price, range_pct):
    """Add a level to visual backtest state."""
    clusters = bt_state[f'{level_type}_clusters']
    levels = bt_state[f'{level_type}_levels']

    # Find or create cluster
    found_cluster = None
    for base in list(clusters.keys()):
        if base * (1 - range_pct) <= price <= base * (1 + range_pct):
            found_cluster = base
            break

    if found_cluster:
        clusters[found_cluster].append(price)
    else:
        clusters[price] = [price]
        found_cluster = price

    # Add to levels if not already present
    cluster_price = max(clusters[found_cluster]) if level_type == 'resistance' else min(clusters[found_cluster])
    if cluster_price not in [l['price'] for l in levels]:
        levels.append({'price': cluster_price})


def check_visual_trades(bt_state, candle, params):
    """Check for trade signals in visual mode."""
    take_profit = params.get('TAKE_PROFIT', 10)
    stop_loss = params.get('STOP_LOSS', 5)
    risk_pct = params.get('RISK_PERCENTAGE', 5)
    breakout_buffer = params.get('BREAKOUT_BUFFER', 0)

    current_price = candle['close']

    # If we have an open trade, check for exit
    if bt_state['open_trade']:
        trade = bt_state['open_trade']
        entry_price = trade['entry_price']

        pnl_pct = ((current_price - entry_price) / entry_price) * 100

        exit_reason = None
        if pnl_pct >= take_profit:
            exit_reason = 'take_profit'
        elif pnl_pct <= -stop_loss:
            exit_reason = 'stop_loss'

        if exit_reason:
            pnl = (current_price - entry_price) * trade['size']
            bt_state['balance'] += pnl
            bt_state['equity'] = bt_state['balance']

            bt_state['trades'].append({
                'id': trade['id'],
                'type': 'long',
                'entry_date': trade['entry_date'],
                'entry_price': round(entry_price, 2),
                'exit_date': candle['date'],
                'exit_price': round(current_price, 2),
                'size': trade['size'],
                'pnl': round(pnl, 2),
                'pnl_percent': round(pnl_pct, 2),
                'exit_reason': exit_reason,
            })
            bt_state['open_trade'] = None

    else:
        # Check for entry signals (breakout above resistance + buffer)
        for level in bt_state['resistance_levels']:
            resistance = level['price']
            # Calculate entry trigger: resistance + (buffer% of resistance)
            entry_trigger = resistance * (1 + breakout_buffer / 100)
            if resistance not in bt_state['broken_resistances']:
                if current_price > entry_trigger:
                    size = int((bt_state['balance'] * risk_pct / 100) / current_price)
                    if size >= 1:
                        bt_state['open_trade'] = {
                            'id': len(bt_state['trades']) + 1,
                            'entry_date': candle['date'],
                            'entry_price': current_price,
                            'size': size,
                        }
                        bt_state['broken_resistances'].add(resistance)
                        break

    # Update equity if we have an open trade
    if bt_state['open_trade']:
        unrealized = (current_price - bt_state['open_trade']['entry_price']) * bt_state['open_trade']['size']
        bt_state['equity'] = bt_state['balance'] + unrealized


def get_bar_update(backtest_id):
    """Get current bar update data."""
    bt_state = visual_backtests[backtest_id]
    index = bt_state['current_index']
    candles = bt_state['candles']

    current_candle = candles[index - 1] if index > 0 else candles[0]

    return {
        'bar_index': index,
        'total_bars': len(candles),
        'candle': current_candle,
        'balance': round(bt_state['balance'], 2),
        'equity': round(bt_state['equity'], 2),
        'open_trade': bt_state['open_trade'],
        'resistance_levels': bt_state['resistance_levels'],
        'support_levels': bt_state['support_levels'],
        'new_trades': bt_state['trades'][-1:] if bt_state['trades'] else [],
    }


def calculate_visual_statistics(backtest_id):
    """Calculate statistics for visual backtest."""
    bt_state = visual_backtests[backtest_id]
    trades = bt_state['trades']
    initial_capital = float(bt_state['config'].get('initial_capital', 100000))

    if not trades:
        return {
            'net_profit': 0,
            'net_profit_percent': 0,
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0,
        }

    winning = [t for t in trades if t['pnl'] > 0]
    losing = [t for t in trades if t['pnl'] <= 0]

    gross_profit = sum(t['pnl'] for t in winning)
    gross_loss = abs(sum(t['pnl'] for t in losing))
    net_profit = gross_profit - gross_loss

    return {
        'net_profit': round(net_profit, 2),
        'net_profit_percent': round((net_profit / initial_capital) * 100, 2),
        'gross_profit': round(gross_profit, 2),
        'gross_loss': round(gross_loss, 2),
        'profit_factor': round(gross_profit / gross_loss, 2) if gross_loss > 0 else 'Infinity',
        'total_trades': len(trades),
        'winning_trades': len(winning),
        'losing_trades': len(losing),
        'win_rate': round((len(winning) / len(trades)) * 100, 2) if trades else 0,
        'average_win': round(gross_profit / len(winning), 2) if winning else 0,
        'average_loss': round(gross_loss / len(losing), 2) if losing else 0,
        'largest_win': round(max(t['pnl'] for t in winning), 2) if winning else 0,
        'largest_loss': round(abs(min(t['pnl'] for t in losing)), 2) if losing else 0,
    }


# ============================================
# OPTIMIZATION HANDLERS
# ============================================

# Store active optimizations
active_optimizations = {}


def generate_param_combinations(param_ranges):
    """Generate all parameter combinations from ranges."""
    import itertools

    params = {}
    for key, range_config in param_ranges.items():
        start = range_config['start']
        end = range_config['end']
        step = range_config['step']

        values = []
        current = start
        while current <= end:
            values.append(round(current, 2))
            current += step
        params[key] = values

    # Generate all combinations
    keys = list(params.keys())
    value_lists = [params[k] for k in keys]

    combinations = []
    for combo in itertools.product(*value_lists):
        combinations.append(dict(zip(keys, combo)))

    return combinations


def run_single_backtest(symbol, start_date, end_date, initial_capital, commission, strategy_params):
    """Run a single backtest and return statistics."""
    try:
        stock = yf.Ticker(symbol)
        if start_date and end_date:
            stock_data = stock.history(start=start_date, end=end_date)
        else:
            stock_data = stock.history(period='2y')

        if stock_data.empty:
            return None

        bt_dataframe = stock_data.copy()

        cerebro = bt.Cerebro()
        bt_data = bt.feeds.PandasData(
            dataname=bt_dataframe,
            open='Open',
            high='High',
            low='Low',
            close='Close',
            volume='Volume',
            openinterest=None
        )
        cerebro.adddata(bt_data)

        # Map frontend param names to strategy param names
        param_mapping = {
            'firstRetracement': 'FIRST_RETRACEMENT',
            'secondRetracement': 'SECOND_RETRACEMENT',
            'touchCount': 'TOUCH_COUNT',
            'breakoutBuffer': 'BREAKOUT_BUFFER',
            'takeProfit': 'TAKE_PROFIT',
            'stopLoss': 'STOP_LOSS',
        }

        mapped_params = {}
        for frontend_key, value in strategy_params.items():
            backend_key = param_mapping.get(frontend_key, frontend_key)
            mapped_params[backend_key] = value

        cerebro.addstrategy(
            BacktestStrategy,
            FIRST_RETRACEMENT=mapped_params.get('FIRST_RETRACEMENT', 5),
            SECOND_RETRACEMENT=mapped_params.get('SECOND_RETRACEMENT', 5),
            RES_SUP_RANGE=mapped_params.get('RES_SUP_RANGE', 0.001),
            TOUCH_COUNT=mapped_params.get('TOUCH_COUNT', 1),
            TAKE_PROFIT=mapped_params.get('TAKE_PROFIT', 10),
            STOP_LOSS=mapped_params.get('STOP_LOSS', 5),
            BREAKOUT_BUFFER=mapped_params.get('BREAKOUT_BUFFER', 0),
        )

        cerebro.broker.set_cash(initial_capital)
        cerebro.broker.setcommission(commission=commission)

        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', riskfreerate=0.0)

        results = cerebro.run()
        strat = results[0]

        trade_analysis = strat.analyzers.trades.get_analysis()
        drawdown_analysis = strat.analyzers.drawdown.get_analysis()
        sharpe_analysis = strat.analyzers.sharpe.get_analysis()

        final_value = cerebro.broker.getvalue()
        net_profit = final_value - initial_capital
        net_profit_percent = (net_profit / initial_capital) * 100

        total_trades = trade_analysis.get('total', {}).get('total', 0)
        won_trades = trade_analysis.get('won', {}).get('total', 0)
        lost_trades = trade_analysis.get('lost', {}).get('total', 0)

        gross_profit = trade_analysis.get('won', {}).get('pnl', {}).get('total', 0)
        gross_loss = abs(trade_analysis.get('lost', {}).get('pnl', {}).get('total', 0))

        win_rate = (won_trades / total_trades * 100) if total_trades > 0 else 0
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float('inf')

        max_drawdown = drawdown_analysis.get('max', {}).get('drawdown', 0)
        max_drawdown_value = drawdown_analysis.get('max', {}).get('moneydown', 0)
        sharpe_ratio = sharpe_analysis.get('sharperatio') or 0

        return {
            'net_profit': round(net_profit, 2),
            'net_profit_percent': round(net_profit_percent, 2),
            'gross_profit': round(gross_profit, 2),
            'gross_loss': round(gross_loss, 2),
            'profit_factor': round(profit_factor, 2) if profit_factor != float('inf') else 'Infinity',
            'total_trades': total_trades,
            'winning_trades': won_trades,
            'losing_trades': lost_trades,
            'win_rate': round(win_rate, 2),
            'max_drawdown': round(max_drawdown_value, 2),
            'max_drawdown_percent': round(max_drawdown, 2),
            'sharpe_ratio': round(sharpe_ratio, 2) if sharpe_ratio else 0,
        }
    except Exception as e:
        logger.error(f"Backtest error: {str(e)}")
        return None


@socketio.on('start_optimization')
def handle_start_optimization(config):
    """Start optimization with parameter ranges using parallel processing."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import time

    try:
        optimization_id = str(uuid.uuid4())
        sid = request.sid

        symbol = config.get('symbol', 'AAPL').upper()
        start_date = config.get('start_date')
        end_date = config.get('end_date')
        initial_capital = float(config.get('initial_capital', 100000))
        commission = float(config.get('commission', 0.001))
        param_ranges = config.get('parameter_ranges', {})
        cpu_cores = int(config.get('cpu_cores', 4))

        # Generate all combinations
        combinations = generate_param_combinations(param_ranges)
        total = len(combinations)

        if total == 0:
            emit('optimization_error', {'message': 'No parameter combinations to test'})
            return

        logger.info(f"Starting optimization with {total} combinations for {symbol} using {cpu_cores} workers")

        # Store optimization state
        active_optimizations[optimization_id] = {
            'sid': sid,
            'is_running': True,
            'total': total,
        }

        # Pre-fetch stock data once to avoid repeated API calls
        stock = yf.Ticker(symbol)
        if start_date and end_date:
            stock_data = stock.history(start=start_date, end=end_date)
        else:
            stock_data = stock.history(period='2y')

        if stock_data.empty:
            emit('optimization_error', {'message': f'No data found for {symbol}'})
            return

        # Run optimization in background thread with parallel processing
        def run_optimization():
            results = []
            best_so_far = None
            completed_count = 0
            start_time = time.time()

            def run_backtest_with_data(params):
                """Run backtest using pre-fetched data."""
                try:
                    bt_dataframe = stock_data.copy()
                    cerebro = bt.Cerebro()
                    bt_data = bt.feeds.PandasData(
                        dataname=bt_dataframe,
                        open='Open',
                        high='High',
                        low='Low',
                        close='Close',
                        volume='Volume',
                        openinterest=None
                    )
                    cerebro.adddata(bt_data)

                    # Map frontend param names to strategy param names
                    param_mapping = {
                        'firstRetracement': 'FIRST_RETRACEMENT',
                        'secondRetracement': 'SECOND_RETRACEMENT',
                        'touchCount': 'TOUCH_COUNT',
                        'breakoutBuffer': 'BREAKOUT_BUFFER',
                        'takeProfit': 'TAKE_PROFIT',
                        'stopLoss': 'STOP_LOSS',
                    }

                    mapped_params = {}
                    for frontend_key, value in params.items():
                        backend_key = param_mapping.get(frontend_key, frontend_key)
                        mapped_params[backend_key] = value

                    cerebro.addstrategy(
                        BacktestStrategy,
                        FIRST_RETRACEMENT=mapped_params.get('FIRST_RETRACEMENT', 5),
                        SECOND_RETRACEMENT=mapped_params.get('SECOND_RETRACEMENT', 5),
                        RES_SUP_RANGE=mapped_params.get('RES_SUP_RANGE', 0.001),
                        TOUCH_COUNT=mapped_params.get('TOUCH_COUNT', 1),
                        TAKE_PROFIT=mapped_params.get('TAKE_PROFIT', 10),
                        STOP_LOSS=mapped_params.get('STOP_LOSS', 5),
                        BREAKOUT_BUFFER=mapped_params.get('BREAKOUT_BUFFER', 0),
                    )

                    cerebro.broker.set_cash(initial_capital)
                    cerebro.broker.setcommission(commission=commission)

                    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
                    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
                    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', riskfreerate=0.0)

                    run_results = cerebro.run()
                    strat = run_results[0]

                    trade_analysis = strat.analyzers.trades.get_analysis()
                    drawdown_analysis = strat.analyzers.drawdown.get_analysis()
                    sharpe_analysis = strat.analyzers.sharpe.get_analysis()

                    final_value = cerebro.broker.getvalue()
                    net_profit = final_value - initial_capital
                    net_profit_percent = (net_profit / initial_capital) * 100

                    total_trades = trade_analysis.get('total', {}).get('total', 0)
                    won_trades = trade_analysis.get('won', {}).get('total', 0)
                    lost_trades = trade_analysis.get('lost', {}).get('total', 0)

                    gross_profit = trade_analysis.get('won', {}).get('pnl', {}).get('total', 0)
                    gross_loss = abs(trade_analysis.get('lost', {}).get('pnl', {}).get('total', 0))

                    win_rate = (won_trades / total_trades * 100) if total_trades > 0 else 0
                    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float('inf')

                    max_drawdown = drawdown_analysis.get('max', {}).get('drawdown', 0)
                    max_drawdown_value = drawdown_analysis.get('max', {}).get('moneydown', 0)
                    sharpe_ratio = sharpe_analysis.get('sharperatio') or 0

                    # Get trade details for advanced stats
                    trades_list = strat.trade_history if hasattr(strat, 'trade_history') else []

                    return {
                        'params': params,
                        'statistics': {
                            'net_profit': round(net_profit, 2),
                            'net_profit_percent': round(net_profit_percent, 2),
                            'gross_profit': round(gross_profit, 2),
                            'gross_loss': round(gross_loss, 2),
                            'profit_factor': round(profit_factor, 2) if profit_factor != float('inf') else 'Infinity',
                            'total_trades': total_trades,
                            'winning_trades': won_trades,
                            'losing_trades': lost_trades,
                            'win_rate': round(win_rate, 2),
                            'max_drawdown': round(max_drawdown_value, 2),
                            'max_drawdown_percent': round(max_drawdown, 2),
                            'sharpe_ratio': round(sharpe_ratio, 2) if sharpe_ratio else 0,
                            'avg_win': round(gross_profit / won_trades, 2) if won_trades > 0 else 0,
                            'avg_loss': round(gross_loss / lost_trades, 2) if lost_trades > 0 else 0,
                            'largest_win': round(trade_analysis.get('won', {}).get('pnl', {}).get('max', 0), 2),
                            'largest_loss': round(abs(trade_analysis.get('lost', {}).get('pnl', {}).get('max', 0)), 2),
                            'avg_trade_duration': trade_analysis.get('len', {}).get('average', 0),
                            'final_balance': round(final_value, 2),
                        },
                        'trades': trades_list[:50] if trades_list else [],  # Limit to 50 trades
                    }
                except Exception as e:
                    logger.error(f"Backtest error for params {params}: {str(e)}")
                    return None

            # Use ThreadPoolExecutor for parallel processing
            with ThreadPoolExecutor(max_workers=cpu_cores) as executor:
                # Submit all tasks
                future_to_params = {
                    executor.submit(run_backtest_with_data, params): params
                    for params in combinations
                }

                # Process results as they complete
                for future in as_completed(future_to_params):
                    # Check if cancelled
                    if optimization_id not in active_optimizations or not active_optimizations[optimization_id]['is_running']:
                        executor.shutdown(wait=False)
                        break

                    result = future.result()
                    completed_count += 1

                    if result:
                        results.append(result)

                        # Track best result
                        if best_so_far is None or result['statistics']['net_profit'] > best_so_far['net_profit']:
                            best_so_far = {
                                'params': result['params'],
                                'net_profit': result['statistics']['net_profit'],
                            }

                    # Send progress update
                    progress_percent = (completed_count / total) * 100
                    elapsed = time.time() - start_time
                    rate = completed_count / elapsed if elapsed > 0 else 0
                    eta = (total - completed_count) / rate if rate > 0 else 0

                    socketio.emit('optimization_progress', {
                        'current': completed_count,
                        'total': total,
                        'percent': round(progress_percent, 1),
                        'current_params': future_to_params[future],
                        'best_so_far': best_so_far,
                        'rate': round(rate, 1),
                        'eta_seconds': round(eta),
                    }, room=sid)

            # Sort results by net profit descending
            results.sort(key=lambda x: x['statistics']['net_profit'], reverse=True)

            elapsed_total = time.time() - start_time

            # Send completion event
            socketio.emit('optimization_complete', {
                'results': results,
                'total_combinations': total,
                'duration_seconds': round(elapsed_total, 1),
                'workers_used': cpu_cores,
            }, room=sid)

            # Cleanup
            if optimization_id in active_optimizations:
                del active_optimizations[optimization_id]

        # Start in background thread
        threading.Thread(target=run_optimization).start()

    except Exception as e:
        logger.error(f"Optimization error: {str(e)}")
        emit('optimization_error', {'message': str(e)})


@socketio.on('cancel_optimization')
def handle_cancel_optimization(data):
    """Cancel a running optimization."""
    optimization_id = data.get('optimization_id')
    if optimization_id in active_optimizations:
        active_optimizations[optimization_id]['is_running'] = False
        logger.info(f"Optimization {optimization_id} cancelled")


# ============================================================================
# HYBRID OPTIMIZATION (VectorBT + Backtrader)
# ============================================================================

def is_gpu_available():
    """Check if CUDA GPU is available for hybrid optimization."""
    try:
        import cupy as cp
        cp.cuda.runtime.getDeviceCount()
        return True
    except Exception:
        return False


def is_vectorbt_available():
    """Check if VectorBT is installed."""
    try:
        import vectorbt
        return True
    except ImportError:
        return False


@app.route('/api/gpu/status', methods=['GET'])
def get_gpu_status():
    """Get GPU and VectorBT availability status for frontend."""
    gpu = is_gpu_available()
    vbt = is_vectorbt_available()
    return jsonify({
        'gpu_available': gpu,
        'vectorbt_available': vbt,
        'hybrid_available': vbt,  # Hybrid works with or without GPU
        'message': 'CUDA GPU detected' if gpu else ('VectorBT available (CPU mode)' if vbt else 'VectorBT not installed')
    })


@socketio.on('start_hybrid_optimization')
def handle_hybrid_optimization(config):
    """
    Start hybrid optimization: VectorBT screening + Backtrader validation.

    Phase 1: VectorBT screens all combinations (CPU or GPU-accelerated if available)
    Phase 2: Backtrader validates top candidates
    """
    try:
        # Check VectorBT availability (works with or without GPU)
        if not is_vectorbt_available():
            emit('hybrid_optimization_error', {
                'message': 'VectorBT not installed. Please install vectorbt to use hybrid optimization.'
            })
            return

        optimization_id = str(uuid.uuid4())
        sid = request.sid

        symbol = config.get('symbol', 'AAPL').upper()
        start_date = config.get('start_date')
        end_date = config.get('end_date')
        initial_capital = float(config.get('initial_capital', 100000))
        commission = float(config.get('commission', 0.001))
        param_ranges = config.get('parameter_ranges', {})
        screening_filters = config.get('screening_filters', {
            'min_sharpe': 0,
            'max_drawdown': 100,
            'min_win_rate': 0,
            'min_trades': 0
        })
        top_n_candidates = int(config.get('top_n_candidates', 100))
        cpu_cores = int(config.get('cpu_cores', 4))

        # Store optimization state
        active_optimizations[optimization_id] = {
            'sid': sid,
            'is_running': True,
            'phase': 'screening',
        }

        logger.info(f"Starting hybrid optimization for {symbol}")

        def run_hybrid():
            try:
                logger.info(f"[HYBRID] Thread started for {symbol}")

                # Import hybrid modules
                logger.info("[HYBRID] Importing modules...")
                from backtesting.data.loader import fetch_data
                from backtesting.vectorbt.optimizer import run_screening
                from backtesting.validation.validator import validate_candidates
                from backtesting.validation.comparator import compare_all_results, filter_valid_results
                logger.info("[HYBRID] Modules imported successfully")

                # Fetch data using unified loader
                logger.info(f"[HYBRID] Fetching data for {symbol}...")
                df = fetch_data(symbol, start_date, end_date)
                logger.info(f"[HYBRID] Data fetched: {len(df)} rows")

                if df.empty:
                    socketio.emit('hybrid_optimization_error', {
                        'message': f'No data found for {symbol}'
                    }, room=sid)
                    return

                # ========================================
                # PHASE 1: VectorBT Screening
                # ========================================
                active_optimizations[optimization_id]['phase'] = 'screening'
                logger.info(f"[HYBRID] Starting Phase 1 screening...")
                logger.info(f"[HYBRID] param_ranges: {param_ranges}")

                def phase1_progress(current, total, candidates_found):
                    if optimization_id not in active_optimizations:
                        return
                    if not active_optimizations[optimization_id]['is_running']:
                        return

                    if current % 10 == 0:  # Log every 10 iterations
                        logger.info(f"[HYBRID] Screening progress: {current}/{total}, candidates: {candidates_found}")

                    socketio.emit('hybrid_screening_progress', {
                        'current': current,
                        'total': total,
                        'percent': round((current / total) * 100, 1),
                        'candidates_found': candidates_found
                    }, room=sid)

                    # Yield control to eventlet to allow WebSocket frame to be sent
                    eventlet.sleep(0)

                logger.info(f"[HYBRID] Calling run_screening...")
                screening_results = run_screening(
                    df=df,
                    param_ranges=param_ranges,
                    initial_capital=initial_capital,
                    commission=commission,
                    filters=screening_filters,
                    top_n=top_n_candidates,
                    progress_callback=phase1_progress,
                    use_parallel=False  # Must be False for eventlet - ThreadPoolExecutor blocks the event loop
                )

                candidates = screening_results.get('candidates', [])
                screening_stats = screening_results.get('stats', {})
                logger.info(f"[HYBRID] Screening complete: {len(candidates)} candidates found")
                logger.info(f"[HYBRID] Screening stats: {screening_stats}")

                if not candidates:
                    logger.info("[HYBRID] No candidates passed filters - sending complete event")
                    socketio.emit('hybrid_optimization_complete', {
                        'results': [],
                        'screening_stats': screening_stats,
                        'validation_stats': {},
                        'message': 'No candidates passed screening filters. Try relaxing the filters (lower min_trades, min_sharpe, etc.)'
                    }, room=sid)
                    return

                # Check if cancelled
                if not active_optimizations.get(optimization_id, {}).get('is_running', False):
                    return

                # ========================================
                # PHASE 2: Backtrader Validation
                # ========================================
                active_optimizations[optimization_id]['phase'] = 'validation'

                def phase2_progress(current, total):
                    if optimization_id not in active_optimizations:
                        return
                    if not active_optimizations[optimization_id]['is_running']:
                        return

                    socketio.emit('hybrid_validation_progress', {
                        'current': current,
                        'total': total,
                        'percent': round((current / total) * 100, 1)
                    }, room=sid)

                    # Yield control to eventlet to allow WebSocket frame to be sent
                    eventlet.sleep(0)

                validated = validate_candidates(
                    df=df,
                    candidates=candidates,
                    initial_capital=initial_capital,
                    commission=commission,
                    max_workers=cpu_cores,
                    progress_callback=phase2_progress
                )

                # ========================================
                # PHASE 3: Comparison & Filtering
                # ========================================
                comparison_results = compare_all_results(validated)
                valid_results = filter_valid_results(validated)

                # Format final results
                final_results = []
                for result in valid_results:
                    final_results.append({
                        'params': result['params'],
                        'statistics': result['bt_statistics'],  # Use BT stats as ground truth
                        'vbt_statistics': result['vbt_statistics'],
                        'comparison': result.get('comparison', {})
                    })

                # Send completion
                socketio.emit('hybrid_optimization_complete', {
                    'results': final_results,
                    'screening_stats': screening_stats,
                    'validation_stats': {
                        'total_validated': len(validated),
                        'passed_parity': len(valid_results),
                        'match_rate': comparison_results['summary']['match_rate']
                    },
                    'comparison_summary': comparison_results['summary']
                }, room=sid)

            except Exception as e:
                logger.error(f"Hybrid optimization error: {str(e)}")
                socketio.emit('hybrid_optimization_error', {
                    'message': str(e)
                }, room=sid)
            finally:
                if optimization_id in active_optimizations:
                    del active_optimizations[optimization_id]

        # Start in background greenlet (eventlet)
        eventlet.spawn(run_hybrid)

    except Exception as e:
        logger.error(f"Hybrid optimization setup error: {str(e)}")
        emit('hybrid_optimization_error', {'message': str(e)})


@socketio.on('cancel_hybrid_optimization')
def handle_cancel_hybrid_optimization(data):
    """Cancel a running hybrid optimization."""
    optimization_id = data.get('optimization_id')
    if optimization_id in active_optimizations:
        active_optimizations[optimization_id]['is_running'] = False
        logger.info(f"Hybrid optimization {optimization_id} cancelled")


# ==========================================
# ML Pattern Discovery Routes
# ==========================================

# Global state for ML pipeline
ml_pipeline_state = {
    'is_running': False,
    'status': None,
    'pipeline': None
}


@app.route('/api/ml/train', methods=['POST'])
def start_ml_training():
    """Start the ML pattern discovery training pipeline."""
    if ml_pipeline_state['is_running']:
        return jsonify({
            'error': 'Training already in progress',
            'status': ml_pipeline_state['status']
        }), 400

    data = request.get_json() or {}
    max_stocks = data.get('max_stocks')  # Optional: limit stocks for testing

    def run_ml_pipeline():
        try:
            from ml.pipeline import MLPipeline

            ml_pipeline_state['is_running'] = True
            ml_pipeline_state['status'] = {
                'stage': 'initializing',
                'progress': 0,
                'message': 'Starting ML pipeline...'
            }

            def progress_callback(stage, progress, message):
                ml_pipeline_state['status'] = {
                    'stage': stage,
                    'progress': progress,
                    'message': message
                }
                # Emit progress to all connected clients
                socketio.emit('ml_progress', ml_pipeline_state['status'])
                eventlet.sleep(0)  # Yield for eventlet

            pipeline = MLPipeline(years_of_data=5, target_horizon=10, use_gpu=True)
            results = pipeline.run(progress_callback=progress_callback, max_stocks=max_stocks)

            ml_pipeline_state['pipeline'] = pipeline
            ml_pipeline_state['status'] = {
                'stage': 'complete',
                'progress': 100,
                'message': 'Training complete!',
                'results': results
            }

            socketio.emit('ml_complete', {
                'status': 'success',
                'results': results
            })

        except Exception as e:
            logger.error(f"ML training error: {str(e)}")
            ml_pipeline_state['status'] = {
                'stage': 'error',
                'progress': 0,
                'message': str(e)
            }
            socketio.emit('ml_error', {'message': str(e)})
        finally:
            ml_pipeline_state['is_running'] = False

    # Run in background greenlet
    eventlet.spawn(run_ml_pipeline)

    return jsonify({
        'message': 'ML training started',
        'status': ml_pipeline_state['status']
    })


@app.route('/api/ml/status', methods=['GET'])
def get_ml_status():
    """Get current ML training status."""
    return jsonify({
        'is_running': ml_pipeline_state['is_running'],
        'status': ml_pipeline_state['status'],
        'has_trained_model': ml_pipeline_state['pipeline'] is not None
    })


@app.route('/api/ml/predictions', methods=['GET'])
def get_ml_predictions():
    """Get ML predictions for a stock."""
    symbol = request.args.get('symbol', 'AAPL').upper()

    pipeline = ml_pipeline_state['pipeline']
    if pipeline is None:
        # Try to load saved model
        try:
            from ml.pipeline import MLPipeline
            pipeline = MLPipeline()
            pipeline.load_model()
            ml_pipeline_state['pipeline'] = pipeline
        except Exception as e:
            return jsonify({
                'error': 'No trained model available. Run /api/ml/train first.',
                'details': str(e)
            }), 404

    try:
        prediction = pipeline.predict(symbol)
        return jsonify(prediction)
    except Exception as e:
        logger.error(f"Prediction error for {symbol}: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/ml/patterns', methods=['GET'])
def get_ml_patterns():
    """Get discovered universal patterns."""
    pipeline = ml_pipeline_state['pipeline']
    if pipeline is None:
        try:
            from ml.pipeline import MLPipeline
            pipeline = MLPipeline()
            pipeline.load_model()
            ml_pipeline_state['pipeline'] = pipeline
        except Exception as e:
            return jsonify({
                'error': 'No trained model available',
                'details': str(e)
            }), 404

    patterns = pipeline.get_patterns()
    return jsonify({
        'patterns': patterns,
        'count': len(patterns)
    })


@app.route('/api/ml/feature-importance', methods=['GET'])
def get_ml_feature_importance():
    """Get feature importance rankings."""
    top_n = request.args.get('top_n', 20, type=int)

    pipeline = ml_pipeline_state['pipeline']
    if pipeline is None:
        try:
            from ml.pipeline import MLPipeline
            pipeline = MLPipeline()
            pipeline.load_model()
            ml_pipeline_state['pipeline'] = pipeline
        except Exception as e:
            return jsonify({
                'error': 'No trained model available',
                'details': str(e)
            }), 404

    try:
        importance_df = pipeline.get_feature_importance(top_n=top_n)
        return jsonify({
            'features': importance_df.to_dict(orient='records'),
            'count': len(importance_df)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/ml/sp500', methods=['GET'])
def get_sp500_list():
    """Get current S&P 500 constituents."""
    try:
        from ml.data.sp500_list import get_sp500_constituents
        constituents = get_sp500_constituents()
        return jsonify({
            'constituents': constituents.to_dict(orient='records'),
            'count': len(constituents)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@socketio.on('subscribe_ml_progress')
def handle_subscribe_ml_progress():
    """Subscribe to ML training progress updates."""
    logger.info("Client subscribed to ML progress")
    if ml_pipeline_state['status']:
        emit('ml_progress', ml_pipeline_state['status'])


if __name__ == '__main__':
    debug_mode = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    socketio.run(app, debug=debug_mode, host=os.environ.get('FLASK_HOST', '127.0.0.1'), port=int(os.environ.get('FLASK_PORT', 5000)), allow_unsafe_werkzeug=True)
