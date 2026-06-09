/**
 * DataBento Dashboard - ES Futures Calendar Trading View
 * 
 * Shows a calendar of trading days where users can click on any day
 * to see active outrights and spreads sorted by volume/liquidity.
 */

import { useState, useEffect, useCallback, useMemo } from 'react';
import InstrumentChart from './InstrumentChart.jsx';

// =============================================================================
// CONSTANTS
// =============================================================================

const API_BASE_URL = 'http://localhost:8000/api/databento';

const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
const WEEKDAYS = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];

// =============================================================================
// UTILITY FUNCTIONS
// =============================================================================

const formatVolume = (volume) => {
  if (!volume) return '0';
  if (volume >= 1000000) return `${(volume / 1000000).toFixed(1)}M`;
  if (volume >= 1000) return `${(volume / 1000).toFixed(1)}K`;
  return volume.toLocaleString();
};

const formatPrice = (price) => {
  if (price === null || price === undefined) return 'N/A';
  return price.toFixed(2);
};

// =============================================================================
// INFO TOOLTIP COMPONENT
// =============================================================================

const InfoTooltip = ({ id, content, isDarkMode }) => {
  const [isActive, setIsActive] = useState(false);
  
  return (
    <div className="relative inline-flex">
      <button
        onClick={() => setIsActive(!isActive)}
        onMouseEnter={() => setIsActive(true)}
        onMouseLeave={() => setIsActive(false)}
        className={`ml-2 w-4 h-4 rounded-full border text-xs font-bold transition-all cursor-pointer flex items-center justify-center ${
          isDarkMode 
            ? 'border-gray-500 text-gray-400 hover:border-blue-400 hover:text-blue-400' 
            : 'border-gray-400 text-gray-500 hover:border-blue-500 hover:text-blue-600'
        }`}
      >
        i
      </button>
      {isActive && (
        <div className={`absolute z-50 bottom-full left-1/2 -translate-x-1/2 mb-2 w-80 p-4 rounded-lg shadow-lg border ${
          isDarkMode ? 'bg-gray-800 border-gray-600 text-gray-200' : 'bg-white border-gray-200 text-gray-800'
        }`}>
          <div className="text-sm leading-relaxed">{content}</div>
          <div className={`absolute top-full left-1/2 -translate-x-1/2 w-0 h-0 border-l-8 border-r-8 border-t-8 border-l-transparent border-r-transparent ${
            isDarkMode ? 'border-t-gray-800' : 'border-t-white'
          }`} />
        </div>
      )}
    </div>
  );
};

// =============================================================================
// CALENDAR COMPONENT
// =============================================================================

const TradingCalendar = ({ 
  tradingDates, 
  selectedDate, 
  onSelectDate, 
  currentMonth, 
  currentYear,
  onPrevMonth,
  onNextMonth,
  isDarkMode 
}) => {
  // Get the first day of the month and number of days
  const firstDayOfMonth = new Date(currentYear, currentMonth, 1).getDay();
  const daysInMonth = new Date(currentYear, currentMonth + 1, 0).getDate();
  
  // Create a map of trading dates for quick lookup
  const tradingDateMap = useMemo(() => {
    const map = {};
    tradingDates.forEach(d => {
      const dateStr = d.date;
      map[dateStr] = d;
    });
    return map;
  }, [tradingDates]);
  
  // Generate calendar grid
  const calendarDays = useMemo(() => {
    const days = [];
    
    // Add empty cells for days before the first of the month
    for (let i = 0; i < firstDayOfMonth; i++) {
      days.push({ type: 'empty', key: `empty-${i}` });
    }
    
    // Add actual days
    for (let day = 1; day <= daysInMonth; day++) {
      const dateStr = `${currentYear}-${String(currentMonth + 1).padStart(2, '0')}-${String(day).padStart(2, '0')}`;
      const tradingData = tradingDateMap[dateStr];
      
      days.push({
        type: 'day',
        key: dateStr,
        day,
        dateStr,
        isTrading: !!tradingData,
        volume: tradingData?.volume || 0,
        isSelected: selectedDate === dateStr,
        isWeekend: new Date(currentYear, currentMonth, day).getDay() === 0 || 
                   new Date(currentYear, currentMonth, day).getDay() === 6,
        isToday: new Date().toISOString().slice(0, 10) === dateStr
      });
    }
    
    return days;
  }, [currentYear, currentMonth, firstDayOfMonth, daysInMonth, tradingDateMap, selectedDate]);

  return (
    <div className={`rounded-lg shadow-md p-6 ${isDarkMode ? 'bg-gray-800' : 'bg-white'}`}>
      {/* Calendar Header */}
      <div className="flex items-center justify-between mb-6">
        <button
          onClick={onPrevMonth}
          className={`p-2 rounded-lg transition-colors ${
            isDarkMode 
              ? 'hover:bg-gray-700 text-gray-300' 
              : 'hover:bg-gray-100 text-gray-600'
          }`}
        >
          ← Prev
        </button>
        
        <h3 className={`text-xl font-bold ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>
          {MONTHS[currentMonth]} {currentYear}
        </h3>
        
        <button
          onClick={onNextMonth}
          className={`p-2 rounded-lg transition-colors ${
            isDarkMode 
              ? 'hover:bg-gray-700 text-gray-300' 
              : 'hover:bg-gray-100 text-gray-600'
          }`}
        >
          Next →
        </button>
      </div>

      {/* Weekday Headers */}
      <div className="grid grid-cols-7 gap-1 mb-2">
        {WEEKDAYS.map(day => (
          <div 
            key={day} 
            className={`text-center text-sm font-semibold py-2 ${
              isDarkMode ? 'text-gray-400' : 'text-gray-500'
            }`}
          >
            {day}
          </div>
        ))}
      </div>

      {/* Calendar Grid */}
      <div className="grid grid-cols-7 gap-1">
        {calendarDays.map(dayInfo => {
          if (dayInfo.type === 'empty') {
            return <div key={dayInfo.key} className="h-16" />;
          }

          const { day, dateStr, isTrading, volume, isSelected, isWeekend, isToday } = dayInfo;
          
          return (
            <button
              key={dayInfo.key}
              onClick={() => isTrading && onSelectDate(dateStr)}
              disabled={!isTrading}
              className={`h-16 rounded-lg text-sm transition-all relative ${
                isSelected
                  ? 'ring-2 ring-blue-500 bg-blue-600 text-white'
                  : isTrading
                    ? isDarkMode
                      ? 'bg-gray-700 hover:bg-gray-600 text-white cursor-pointer'
                      : 'bg-green-50 hover:bg-green-100 text-gray-900 cursor-pointer'
                    : isWeekend
                      ? isDarkMode
                        ? 'bg-gray-900/50 text-gray-600 cursor-not-allowed'
                        : 'bg-gray-100 text-gray-400 cursor-not-allowed'
                      : isDarkMode
                        ? 'bg-gray-800/50 text-gray-600 cursor-not-allowed'
                        : 'bg-gray-50 text-gray-400 cursor-not-allowed'
              } ${isToday ? 'ring-2 ring-yellow-400' : ''}`}
            >
              <div className="flex flex-col items-center justify-center h-full">
                <span className={`font-bold ${isSelected ? 'text-white' : ''}`}>
                  {day}
                </span>
                {isTrading && volume > 0 && (
                  <span className={`text-xs ${
                    isSelected 
                      ? 'text-blue-100' 
                      : isDarkMode 
                        ? 'text-green-400' 
                        : 'text-green-600'
                  }`}>
                    {formatVolume(volume)}
                  </span>
                )}
              </div>
              {isToday && (
                <div className="absolute top-1 right-1 w-2 h-2 rounded-full bg-yellow-400" />
              )}
            </button>
          );
        })}
      </div>

      {/* Legend */}
      <div className={`mt-4 flex items-center gap-4 text-xs ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>
        <div className="flex items-center gap-1">
          <div className={`w-3 h-3 rounded ${isDarkMode ? 'bg-gray-700' : 'bg-green-50'}`} />
          <span>Trading Day</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="w-3 h-3 rounded bg-blue-600" />
          <span>Selected</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="w-3 h-3 rounded ring-2 ring-yellow-400" />
          <span>Today</span>
        </div>
      </div>
    </div>
  );
};

// =============================================================================
// BUY/SELL VOLUME BAR COMPONENT
// =============================================================================

const BuySellBar = ({ buyVolume, sellVolume, isDarkMode }) => {
  if (!buyVolume && !sellVolume) return <span className="text-gray-400">-</span>;
  
  const total = (buyVolume || 0) + (sellVolume || 0);
  if (total === 0) return <span className="text-gray-400">-</span>;
  
  const buyPct = ((buyVolume || 0) / total) * 100;
  const sellPct = ((sellVolume || 0) / total) * 100;
  
  return (
    <div className="flex items-center gap-2 min-w-[100px]">
      <div className="flex-1 h-3 rounded-full overflow-hidden flex bg-gray-200 dark:bg-gray-600">
        <div 
          className="bg-green-500 h-full" 
          style={{ width: `${buyPct}%` }}
          title={`Buy: ${(buyVolume || 0).toLocaleString()}`}
        />
        <div 
          className="bg-red-500 h-full" 
          style={{ width: `${sellPct}%` }}
          title={`Sell: ${(sellVolume || 0).toLocaleString()}`}
        />
      </div>
      <span className={`text-xs font-mono ${buyPct > 50 ? 'text-green-500' : 'text-red-500'}`}>
        {buyPct.toFixed(0)}%
      </span>
    </div>
  );
};

// =============================================================================
// DAILY SUMMARY PANEL COMPONENT
// =============================================================================

const DailySummaryPanel = ({ summary, frontMonth, isDarkMode }) => {
  if (!summary) return null;
  
  return (
    <div className={`rounded-lg shadow-md p-6 mb-6 ${isDarkMode ? 'bg-gray-800' : 'bg-white'}`}>
      <h3 className={`text-lg font-semibold mb-4 ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>
        Daily Summary
      </h3>
      
      <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4">
        <div className={`p-3 rounded-lg ${isDarkMode ? 'bg-gray-700' : 'bg-gray-50'}`}>
          <div className={`text-xs uppercase ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>
            Total Outrights
          </div>
          <div className={`text-xl font-bold ${isDarkMode ? 'text-blue-400' : 'text-blue-600'}`}>
            {summary.outright_count}
          </div>
        </div>
        
        <div className={`p-3 rounded-lg ${isDarkMode ? 'bg-gray-700' : 'bg-gray-50'}`}>
          <div className={`text-xs uppercase ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>
            Total Spreads
          </div>
          <div className={`text-xl font-bold ${isDarkMode ? 'text-purple-400' : 'text-purple-600'}`}>
            {summary.spread_count}
          </div>
        </div>
        
        <div className={`p-3 rounded-lg ${isDarkMode ? 'bg-gray-700' : 'bg-gray-50'}`}>
          <div className={`text-xs uppercase ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>
            Outright Volume
          </div>
          <div className={`text-xl font-bold ${isDarkMode ? 'text-green-400' : 'text-green-600'}`}>
            {formatVolume(summary.total_outright_volume)}
          </div>
        </div>
        
        <div className={`p-3 rounded-lg ${isDarkMode ? 'bg-gray-700' : 'bg-gray-50'}`}>
          <div className={`text-xs uppercase ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>
            Spread Volume
          </div>
          <div className={`text-xl font-bold ${isDarkMode ? 'text-green-400' : 'text-green-600'}`}>
            {formatVolume(summary.total_spread_volume)}
          </div>
        </div>
        
        <div className={`p-3 rounded-lg ${isDarkMode ? 'bg-gray-700' : 'bg-gray-50'}`}>
          <div className={`text-xs uppercase ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>
            Total Trades
          </div>
          <div className={`text-xl font-bold ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>
            {summary.total_trades?.toLocaleString() || '-'}
          </div>
          {!summary.has_trade_data && (
            <div className={`text-xs ${isDarkMode ? 'text-gray-500' : 'text-gray-400'}`}>
              (Dec 2025+)
            </div>
          )}
        </div>
        
        <div className={`p-3 rounded-lg ${isDarkMode ? 'bg-gray-700' : 'bg-gray-50'}`}>
          <div className={`text-xs uppercase ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>
            Large Trades
          </div>
          <div className={`text-xl font-bold ${isDarkMode ? 'text-yellow-400' : 'text-yellow-600'}`}>
            {summary.total_large_trades || '-'}
          </div>
        </div>
      </div>
      
      {frontMonth && (
        <div className={`mt-4 p-4 rounded-lg border ${
          isDarkMode ? 'bg-blue-900/20 border-blue-800' : 'bg-blue-50 border-blue-200'
        }`}>
          <div className={`text-sm font-semibold ${isDarkMode ? 'text-blue-400' : 'text-blue-700'}`}>
            Front Month: {frontMonth.contract}
          </div>
          <div className="grid grid-cols-2 md:grid-cols-5 gap-4 mt-2">
            <div>
              <span className={`text-xs ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>Open: </span>
              <span className="font-mono">{formatPrice(frontMonth.open)}</span>
            </div>
            <div>
              <span className={`text-xs ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>High: </span>
              <span className="font-mono text-green-500">{formatPrice(frontMonth.high)}</span>
            </div>
            <div>
              <span className={`text-xs ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>Low: </span>
              <span className="font-mono text-red-500">{formatPrice(frontMonth.low)}</span>
            </div>
            <div>
              <span className={`text-xs ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>Close: </span>
              <span className="font-mono font-bold">{formatPrice(frontMonth.close)}</span>
            </div>
            <div>
              <span className={`text-xs ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>Range: </span>
              <span className="font-mono">{frontMonth.daily_range?.toFixed(2) || '-'}</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

// =============================================================================
// FORWARD-LOOKING DATA PANEL
// =============================================================================

const ForwardDataPanel = ({ forwardData, selectedDate, loading, error, isDarkMode }) => {
  if (!selectedDate) return null;
  
  if (loading) {
    return (
      <div className={`rounded-lg shadow-md p-6 mb-6 ${isDarkMode ? 'bg-gray-800' : 'bg-white'}`}>
        <div className="flex items-center gap-3">
          <div className={`animate-spin rounded-full h-5 w-5 border-b-2 ${isDarkMode ? 'border-cyan-400' : 'border-cyan-600'}`} />
          <span className={isDarkMode ? 'text-gray-300' : 'text-gray-600'}>Loading forward-looking data...</span>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className={`rounded-lg shadow-md p-4 mb-6 border ${isDarkMode ? 'bg-red-900/20 border-red-800' : 'bg-red-50 border-red-200'}`}>
        <div className={`text-sm ${isDarkMode ? 'text-red-300' : 'text-red-700'}`}>
          Forward data unavailable: {error}
        </div>
      </div>
    );
  }

  if (!forwardData) return null;

  const { rates, volatility, events, regime, rate_stability } = forwardData;

  const formatPct = (val) => {
    if (val === null || val === undefined) return '-';
    return `${(val * 100).toFixed(2)}%`;
  };

  const formatBps = (val) => {
    if (val === null || val === undefined) return '-';
    return `${(val * 10000).toFixed(1)} bps`;
  };

  // Determine overall context color for the panel
  const hasEvent = events?.on_date?.length > 0;
  const isPreEvent = events?.is_pre_event;
  const isPostEvent = events?.is_post_event;
  const rateUnstable = rate_stability && !rate_stability.is_stable;

  const contextBorder = hasEvent || isPreEvent
    ? isDarkMode ? 'border-yellow-600' : 'border-yellow-400'
    : isDarkMode ? 'border-gray-700' : 'border-gray-200';

  return (
    <div className={`rounded-lg shadow-md p-6 mb-6 border ${contextBorder} ${isDarkMode ? 'bg-gray-800' : 'bg-white'}`}>
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <h3 className={`text-lg font-semibold ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>
          Forward-Looking Context
        </h3>
        <div className="flex items-center gap-2">
          {rates?.rate_source === 'spot_fallback' && (
            <span className={`text-xs px-2 py-1 rounded ${isDarkMode ? 'bg-yellow-900/50 text-yellow-300' : 'bg-yellow-100 text-yellow-700'}`}>
              Spot Rates (no ZQ yet)
            </span>
          )}
          {rates?.rate_source === 'forward' && (
            <span className={`text-xs px-2 py-1 rounded ${isDarkMode ? 'bg-green-900/50 text-green-300' : 'bg-green-100 text-green-700'}`}>
              Forward Rates
            </span>
          )}
        </div>
      </div>

      {/* Event Banner */}
      {(hasEvent || isPreEvent || isPostEvent) && (
        <div className={`mb-4 p-3 rounded-lg border ${
          isDarkMode ? 'bg-yellow-900/30 border-yellow-700' : 'bg-yellow-50 border-yellow-300'
        }`}>
          <div className="flex items-center gap-2 flex-wrap">
            {events.on_date.map(evt => (
              <span key={evt} className={`px-3 py-1 rounded-full text-sm font-bold ${
                evt === 'FOMC' 
                  ? isDarkMode ? 'bg-red-900/60 text-red-300' : 'bg-red-100 text-red-700'
                  : evt === 'CPI'
                    ? isDarkMode ? 'bg-orange-900/60 text-orange-300' : 'bg-orange-100 text-orange-700'
                    : isDarkMode ? 'bg-blue-900/60 text-blue-300' : 'bg-blue-100 text-blue-700'
              }`}>
                {evt} Day
              </span>
            ))}
            {isPreEvent && !hasEvent && (
              <span className={`px-3 py-1 rounded-full text-sm font-semibold ${
                isDarkMode ? 'bg-yellow-900/60 text-yellow-300' : 'bg-yellow-100 text-yellow-700'
              }`}>
                Pre-Event (major event tomorrow)
              </span>
            )}
            {isPostEvent && !hasEvent && (
              <span className={`px-3 py-1 rounded-full text-sm ${
                isDarkMode ? 'bg-gray-700 text-gray-300' : 'bg-gray-100 text-gray-600'
              }`}>
                Post-Event
              </span>
            )}
          </div>
          {events.nearby?.length > 0 && (
            <div className={`mt-2 text-xs ${isDarkMode ? 'text-yellow-200/70' : 'text-yellow-700/70'}`}>
              Nearby: {events.nearby.map(e => `${e.type} (${e.date})`).join(', ')}
            </div>
          )}
        </div>
      )}

      {/* Main Grid */}
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3">

        {/* SOFR Spot */}
        <div className={`p-3 rounded-lg ${isDarkMode ? 'bg-gray-700' : 'bg-gray-50'}`}>
          <div className={`text-xs uppercase tracking-wider ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>
            SOFR Spot
          </div>
          <div className={`text-xl font-bold font-mono ${isDarkMode ? 'text-cyan-400' : 'text-cyan-700'}`}>
            {formatPct(rates?.sofr_spot)}
          </div>
          {rate_stability && (
            <div className={`text-xs mt-1 ${
              rate_stability.is_stable
                ? isDarkMode ? 'text-green-400' : 'text-green-600'
                : isDarkMode ? 'text-red-400' : 'text-red-600'
            }`}>
              {rate_stability.sofr_1d_change_bps > 0 ? '+' : ''}{rate_stability.sofr_1d_change_bps} bps
              {rate_stability.is_stable ? ' (stable)' : ' (moving!)'}
            </div>
          )}
        </div>

        {/* Fed Funds Target */}
        <div className={`p-3 rounded-lg ${isDarkMode ? 'bg-gray-700' : 'bg-gray-50'}`}>
          <div className={`text-xs uppercase tracking-wider ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>
            Fed Funds Target
          </div>
          <div className={`text-xl font-bold font-mono ${isDarkMode ? 'text-cyan-400' : 'text-cyan-700'}`}>
            {rates?.fed_funds_target_lower != null && rates?.fed_funds_target_upper != null
              ? `${formatPct(rates.fed_funds_target_lower)}–${formatPct(rates.fed_funds_target_upper)}`
              : '-'
            }
          </div>
        </div>

        {/* Forward Rate (3m) */}
        <div className={`p-3 rounded-lg ${isDarkMode ? 'bg-gray-700' : 'bg-gray-50'}`}>
          <div className={`text-xs uppercase tracking-wider ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>
            Implied Rate 3M
          </div>
          <div className={`text-xl font-bold font-mono ${isDarkMode ? 'text-indigo-400' : 'text-indigo-700'}`}>
            {formatPct(rates?.implied_rate_3m)}
          </div>
          {rates?.rate_source === 'spot_fallback' && (
            <div className={`text-xs mt-1 ${isDarkMode ? 'text-gray-500' : 'text-gray-400'}`}>= spot (fallback)</div>
          )}
        </div>

        {/* SPX Div Yield */}
        <div className={`p-3 rounded-lg ${isDarkMode ? 'bg-gray-700' : 'bg-gray-50'}`}>
          <div className={`text-xs uppercase tracking-wider ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>
            SPX Div Yield
          </div>
          <div className={`text-xl font-bold font-mono ${isDarkMode ? 'text-emerald-400' : 'text-emerald-700'}`}>
            {formatPct(rates?.spx_div_yield_trail)}
          </div>
          <div className={`text-xs mt-1 ${isDarkMode ? 'text-gray-500' : 'text-gray-400'}`}>trailing 12m</div>
        </div>

        {/* Net Carry */}
        <div className={`p-3 rounded-lg ${isDarkMode ? 'bg-gray-700' : 'bg-gray-50'}`}>
          <div className={`text-xs uppercase tracking-wider ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>
            Net Carry (Spot)
          </div>
          <div className={`text-xl font-bold font-mono ${
            rates?.net_carry_spot > 0
              ? isDarkMode ? 'text-green-400' : 'text-green-600'
              : isDarkMode ? 'text-red-400' : 'text-red-600'
          }`}>
            {rates?.net_carry_spot != null
              ? `${rates.net_carry_spot > 0 ? '+' : ''}${formatPct(rates.net_carry_spot)}`
              : '-'
            }
          </div>
          <div className={`text-xs mt-1 ${isDarkMode ? 'text-gray-500' : 'text-gray-400'}`}>
            {rates?.net_carry_spot > 0 ? 'contango' : rates?.net_carry_spot < 0 ? 'backwardation' : ''}
          </div>
        </div>

        {/* VIX */}
        <div className={`p-3 rounded-lg ${isDarkMode ? 'bg-gray-700' : 'bg-gray-50'}`}>
          <div className={`text-xs uppercase tracking-wider ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>
            VIX Close
          </div>
          <div className={`text-xl font-bold font-mono ${
            volatility?.vix_close > 25
              ? isDarkMode ? 'text-red-400' : 'text-red-600'
              : volatility?.vix_close > 18
                ? isDarkMode ? 'text-yellow-400' : 'text-yellow-600'
                : isDarkMode ? 'text-green-400' : 'text-green-600'
          }`}>
            {volatility?.vix_close?.toFixed(2) || '-'}
          </div>
          {volatility?.vix_rv_ratio != null && (
            <div className={`text-xs mt-1 ${
              volatility.vix_rv_ratio > 1.3
                ? isDarkMode ? 'text-yellow-400' : 'text-yellow-600'
                : volatility.vix_rv_ratio < 0.8
                  ? isDarkMode ? 'text-blue-400' : 'text-blue-600'
                  : isDarkMode ? 'text-gray-400' : 'text-gray-500'
            }`}>
              VIX/RV: {volatility.vix_rv_ratio.toFixed(2)}
              {volatility.vix_rv_ratio > 1.3 ? ' (fear)' : volatility.vix_rv_ratio < 0.8 ? ' (calm)' : ''}
            </div>
          )}
        </div>
      </div>

      {/* Regime Context Row */}
      {regime && (
        <div className="mt-3 flex items-center gap-3 flex-wrap">
          <span className={`px-3 py-1 rounded-full text-xs font-semibold ${
            regime.vol_regime === 'high'
              ? isDarkMode ? 'bg-red-900/50 text-red-300' : 'bg-red-100 text-red-700'
              : regime.vol_regime === 'medium'
                ? isDarkMode ? 'bg-yellow-900/50 text-yellow-300' : 'bg-yellow-100 text-yellow-700'
                : isDarkMode ? 'bg-green-900/50 text-green-300' : 'bg-green-100 text-green-700'
          }`}>
            Vol: {regime.vol_regime || '-'}
          </span>
          <span className={`px-3 py-1 rounded-full text-xs font-semibold ${
            regime.trend_regime === 'up'
              ? isDarkMode ? 'bg-green-900/50 text-green-300' : 'bg-green-100 text-green-700'
              : isDarkMode ? 'bg-red-900/50 text-red-300' : 'bg-red-100 text-red-700'
          }`}>
            Trend: {regime.trend_regime || '-'}
          </span>
          {regime.in_roll_window && (
            <span className={`px-3 py-1 rounded-full text-xs font-semibold ${
              isDarkMode ? 'bg-purple-900/50 text-purple-300' : 'bg-purple-100 text-purple-700'
            }`}>
              Roll Window
            </span>
          )}
          {regime.days_to_next_roll != null && (
            <span className={`text-xs ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>
              {regime.days_to_next_roll}d to roll
            </span>
          )}
          {regime.is_quad_witching && (
            <span className={`px-3 py-1 rounded-full text-xs font-semibold ${
              isDarkMode ? 'bg-orange-900/50 text-orange-300' : 'bg-orange-100 text-orange-700'
            }`}>
              Quad Witching
            </span>
          )}
        </div>
      )}

      {/* Forward Rate Term Structure (mini chart) */}
      {rates && (rates.implied_rate_1m != null || rates.implied_rate_3m != null) && (
        <div className={`mt-4 p-3 rounded-lg ${isDarkMode ? 'bg-gray-900/50' : 'bg-gray-50'}`}>
          <div className={`text-xs uppercase tracking-wider mb-2 ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>
            Implied Rate Curve
          </div>
          <div className="flex items-end gap-1 h-16">
            {[
              { label: 'Spot', val: rates.sofr_spot },
              { label: '1M', val: rates.implied_rate_1m },
              { label: '2M', val: rates.implied_rate_2m },
              { label: '3M', val: rates.implied_rate_3m },
              { label: '6M', val: rates.implied_rate_6m },
            ].map(({ label, val }) => {
              if (val == null) return null;
              const maxRate = Math.max(
                rates.sofr_spot || 0, rates.implied_rate_1m || 0,
                rates.implied_rate_3m || 0, rates.implied_rate_6m || 0, 0.001
              );
              const height = Math.max(4, (val / maxRate) * 100);
              return (
                <div key={label} className="flex flex-col items-center flex-1">
                  <div className={`text-xs font-mono mb-1 ${isDarkMode ? 'text-gray-300' : 'text-gray-700'}`}>
                    {(val * 100).toFixed(2)}
                  </div>
                  <div
                    className={`w-full rounded-t ${
                      label === 'Spot'
                        ? isDarkMode ? 'bg-cyan-500' : 'bg-cyan-400'
                        : isDarkMode ? 'bg-indigo-500' : 'bg-indigo-400'
                    }`}
                    style={{ height: `${height}%`, minHeight: '4px' }}
                  />
                  <div className={`text-xs mt-1 ${isDarkMode ? 'text-gray-500' : 'text-gray-400'}`}>{label}</div>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
};

// =============================================================================
// INSTRUMENTS TABLE COMPONENT (Enhanced)
// =============================================================================

const InstrumentsTable = ({ 
  instruments, 
  selectedDate, 
  frontMonth,
  summary,
  loading, 
  isDarkMode,
  selectedInstrumentSymbol,
  onSelectInstrument,
}) => {
  const [expandedRow, setExpandedRow] = useState(null);
  
  if (loading) {
    return (
      <div className={`rounded-lg shadow-md p-8 ${isDarkMode ? 'bg-gray-800' : 'bg-white'}`}>
        <div className="flex items-center justify-center">
          <div className={`animate-spin rounded-full h-8 w-8 border-b-2 ${
            isDarkMode ? 'border-blue-400' : 'border-blue-600'
          }`} />
          <span className={`ml-3 ${isDarkMode ? 'text-gray-300' : 'text-gray-600'}`}>
            Loading instruments for {selectedDate}...
          </span>
        </div>
      </div>
    );
  }

  if (!selectedDate) {
    return (
      <div className={`rounded-lg shadow-md p-8 ${isDarkMode ? 'bg-gray-800' : 'bg-white'}`}>
        <div className="text-center">
          <div className="text-4xl mb-4">📅</div>
          <h3 className={`text-lg font-medium mb-2 ${isDarkMode ? 'text-gray-300' : 'text-gray-900'}`}>
            Select a Trading Day
          </h3>
          <p className={isDarkMode ? 'text-gray-400' : 'text-gray-500'}>
            Click on a green day in the calendar to see active outrights and spreads
          </p>
        </div>
      </div>
    );
  }

  if (!instruments || instruments.length === 0) {
    return (
      <div className={`rounded-lg shadow-md p-8 ${isDarkMode ? 'bg-gray-800' : 'bg-white'}`}>
        <div className="text-center">
          <div className="text-4xl mb-4">📊</div>
          <h3 className={`text-lg font-medium mb-2 ${isDarkMode ? 'text-gray-300' : 'text-gray-900'}`}>
            No Instruments Found
          </h3>
          <p className={isDarkMode ? 'text-gray-400' : 'text-gray-500'}>
            No tradeable instruments with volume on {selectedDate}
          </p>
        </div>
      </div>
    );
  }

  return (
    <>
      {/* Daily Summary */}
      <DailySummaryPanel summary={summary} frontMonth={frontMonth} isDarkMode={isDarkMode} />
      
      {/* Instruments Table */}
      <div className={`rounded-lg shadow-md ${isDarkMode ? 'bg-gray-800' : 'bg-white'}`}>
        {/* Header */}
        <div className={`px-6 py-4 border-b ${isDarkMode ? 'border-gray-700' : 'border-gray-200'}`}>
          <div className="flex items-center justify-between">
            <div>
              <h3 className={`text-lg font-semibold ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>
                Instruments for {selectedDate}
              </h3>
              <p className={`text-sm ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>
                {instruments.length} instruments sorted by volume • Click a row for details
              </p>
            </div>
          </div>
        </div>

        {/* Table */}
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200">
            <thead className={isDarkMode ? 'bg-gray-700' : 'bg-gray-50'}>
              <tr>
                <th className={`px-3 py-3 text-left text-xs font-medium uppercase tracking-wider ${
                  isDarkMode ? 'text-gray-400' : 'text-gray-500'
                }`}>
                  #
                </th>
                <th className={`px-3 py-3 text-left text-xs font-medium uppercase tracking-wider ${
                  isDarkMode ? 'text-gray-400' : 'text-gray-500'
                }`}>
                  Type
                </th>
                <th className={`px-3 py-3 text-left text-xs font-medium uppercase tracking-wider ${
                  isDarkMode ? 'text-gray-400' : 'text-gray-500'
                }`}>
                  Symbol
                </th>
                <th className={`px-3 py-3 text-right text-xs font-medium uppercase tracking-wider ${
                  isDarkMode ? 'text-gray-400' : 'text-gray-500'
                }`}>
                  Volume
                </th>
                <th className={`px-3 py-3 text-center text-xs font-medium uppercase tracking-wider ${
                  isDarkMode ? 'text-gray-400' : 'text-gray-500'
                }`}>
                  Buy/Sell
                </th>
                <th className={`px-3 py-3 text-right text-xs font-medium uppercase tracking-wider ${
                  isDarkMode ? 'text-gray-400' : 'text-gray-500'
                }`}>
                  Close
                </th>
                <th className={`px-3 py-3 text-right text-xs font-medium uppercase tracking-wider ${
                  isDarkMode ? 'text-gray-400' : 'text-gray-500'
                }`}>
                  Range
                </th>
                <th className={`px-3 py-3 text-right text-xs font-medium uppercase tracking-wider ${
                  isDarkMode ? 'text-gray-400' : 'text-gray-500'
                }`}>
                  Trades
                </th>
                <th className={`px-3 py-3 text-right text-xs font-medium uppercase tracking-wider ${
                  isDarkMode ? 'text-gray-400' : 'text-gray-500'
                }`}>
                  Large
                </th>
                <th className={`px-3 py-3 text-right text-xs font-medium uppercase tracking-wider ${
                  isDarkMode ? 'text-gray-400' : 'text-gray-500'
                }`}>
                  Avg Size
                </th>
              </tr>
            </thead>
            <tbody className={`divide-y ${isDarkMode ? 'bg-gray-800 divide-gray-700' : 'bg-white divide-gray-200'}`}>
              {instruments.map((inst, index) => (
                <tr 
                  key={`${inst.type}-${inst.symbol}`}
                  onClick={() => {
                    const isSameRow = selectedInstrumentSymbol === inst.symbol;
                    setExpandedRow(isSameRow ? null : index);
                    onSelectInstrument?.(isSameRow ? null : inst.symbol);
                  }}
                  className={`transition-colors cursor-pointer ${
                    selectedInstrumentSymbol === inst.symbol
                      ? 'ring-2 ring-cyan-500 ring-inset'
                      : ''
                  } ${
                    index === 0
                      ? isDarkMode
                        ? 'bg-green-900/30 hover:bg-green-900/50'
                        : 'bg-green-50 hover:bg-green-100'
                      : isDarkMode 
                        ? 'hover:bg-gray-700' 
                        : 'hover:bg-gray-50'
                  }`}
                >
                  <td className={`px-3 py-3 whitespace-nowrap text-sm font-mono ${
                    isDarkMode ? 'text-gray-400' : 'text-gray-500'
                  }`}>
                    {index + 1}
                    {index === 0 && <span className="ml-1 text-yellow-500">🏆</span>}
                  </td>
                  <td className="px-3 py-3 whitespace-nowrap">
                    <span className={`px-2 py-1 rounded text-xs font-medium ${
                      inst.type === 'outright'
                        ? isDarkMode 
                          ? 'bg-blue-900/50 text-blue-300' 
                          : 'bg-blue-100 text-blue-700'
                        : isDarkMode
                          ? 'bg-purple-900/50 text-purple-300'
                          : 'bg-purple-100 text-purple-700'
                    }`}>
                      {inst.type === 'outright' ? '📈' : '🔄'}
                    </span>
                  </td>
                  <td className={`px-3 py-3 whitespace-nowrap text-sm font-mono font-bold ${
                    isDarkMode ? 'text-white' : 'text-gray-900'
                  }`}>
                    {inst.symbol}
                    {inst.vol_vs_avg_pct && (
                      <span className={`ml-2 text-xs ${
                        inst.vol_vs_avg_pct > 120 
                          ? 'text-green-500' 
                          : inst.vol_vs_avg_pct < 80 
                            ? 'text-red-400' 
                            : 'text-gray-400'
                      }`}>
                        {inst.vol_vs_avg_pct > 100 ? '↑' : '↓'}{inst.vol_vs_avg_pct}%
                      </span>
                    )}
                  </td>
                  <td className={`px-3 py-3 whitespace-nowrap text-sm font-mono text-right ${
                    isDarkMode ? 'text-green-400' : 'text-green-600'
                  }`}>
                    {inst.volume?.toLocaleString() || '0'}
                  </td>
                  <td className="px-3 py-3 whitespace-nowrap">
                    <BuySellBar 
                      buyVolume={inst.buy_volume} 
                      sellVolume={inst.sell_volume} 
                      isDarkMode={isDarkMode} 
                    />
                  </td>
                  <td className={`px-3 py-3 whitespace-nowrap text-sm font-mono text-right ${
                    isDarkMode ? 'text-gray-300' : 'text-gray-900'
                  }`}>
                    {formatPrice(inst.close)}
                  </td>
                  <td className={`px-3 py-3 whitespace-nowrap text-sm font-mono text-right ${
                    isDarkMode ? 'text-gray-400' : 'text-gray-500'
                  }`}>
                    {inst.daily_range?.toFixed(2) || '-'}
                  </td>
                  <td className={`px-3 py-3 whitespace-nowrap text-sm font-mono text-right ${
                    isDarkMode ? 'text-gray-400' : 'text-gray-500'
                  }`}>
                    {inst.trade_count?.toLocaleString() || '-'}
                  </td>
                  <td className={`px-3 py-3 whitespace-nowrap text-sm font-mono text-right ${
                    inst.large_trades > 0 
                      ? isDarkMode ? 'text-yellow-400' : 'text-yellow-600'
                      : isDarkMode ? 'text-gray-500' : 'text-gray-400'
                  }`}>
                    {inst.large_trades || '-'}
                  </td>
                  <td className={`px-3 py-3 whitespace-nowrap text-sm font-mono text-right ${
                    isDarkMode ? 'text-gray-400' : 'text-gray-500'
                  }`}>
                    {inst.avg_trade_size?.toFixed(1) || '-'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </>
  );
};

// =============================================================================
// MAIN DATABENTO DASHBOARD COMPONENT
// =============================================================================

const DBentoDashboard = ({ isDarkMode = false }) => {
  // Date state
  const [currentMonth, setCurrentMonth] = useState(() => new Date().getMonth());
  const [currentYear, setCurrentYear] = useState(() => new Date().getFullYear());
  const [selectedDate, setSelectedDate] = useState(null);
  
  // Data state
  const [tradingDates, setTradingDates] = useState([]);
  const [instruments, setInstruments] = useState([]);
  const [frontMonth, setFrontMonth] = useState(null);
  const [summary, setSummary] = useState(null);
  const [dateRange, setDateRange] = useState(null);
  const [forwardData, setForwardData] = useState(null);
  
  // Loading state
  const [loadingDates, setLoadingDates] = useState(false);
  const [loadingInstruments, setLoadingInstruments] = useState(false);
  const [loadingForward, setLoadingForward] = useState(false);
  const [forwardError, setForwardError] = useState(null);
  const [error, setError] = useState(null);
  const [selectedInstrumentSymbol, setSelectedInstrumentSymbol] = useState(null);

  // Fetch date range on mount
  useEffect(() => {
    const fetchDateRange = async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/calendar/date-range`);
        if (!response.ok) throw new Error('Failed to fetch date range');
        const data = await response.json();
        setDateRange(data);
        
        // Set initial view to the latest available date
        if (data.latest) {
          const latestDate = new Date(data.latest);
          setCurrentYear(latestDate.getFullYear());
          setCurrentMonth(latestDate.getMonth());
        }
      } catch (err) {
        console.error('Error fetching date range:', err);
        setError(err.message);
      }
    };
    fetchDateRange();
  }, []);

  // Fetch trading dates for current month view
  useEffect(() => {
    const fetchDates = async () => {
      setLoadingDates(true);
      setError(null);
      
      try {
        // Calculate date range for the visible month
        // Use Date constructor to get the correct last day of the month
        const startDate = `${currentYear}-${String(currentMonth + 1).padStart(2, '0')}-01`;
        const lastDayOfMonth = new Date(currentYear, currentMonth + 1, 0).getDate();
        const endDate = `${currentYear}-${String(currentMonth + 1).padStart(2, '0')}-${String(lastDayOfMonth).padStart(2, '0')}`;
        
        const response = await fetch(
          `${API_BASE_URL}/calendar/dates?start_date=${startDate}&end_date=${endDate}&limit=50`
        );
        
        if (!response.ok) throw new Error('Failed to fetch trading dates');
        const data = await response.json();
        setTradingDates(data.dates || []);
      } catch (err) {
        console.error('Error fetching dates:', err);
        setError(err.message);
      } finally {
        setLoadingDates(false);
      }
    };
    
    fetchDates();
  }, [currentMonth, currentYear]);

  // Fetch instruments when a date is selected
  useEffect(() => {
    if (!selectedDate) {
      setInstruments([]);
      setFrontMonth(null);
      setSummary(null);
      setForwardData(null);
      setSelectedInstrumentSymbol(null);
      return;
    }

    const fetchInstruments = async () => {
      setLoadingInstruments(true);
      setError(null);
      
      try {
        const response = await fetch(`${API_BASE_URL}/instruments/${selectedDate}`);
        if (!response.ok) throw new Error('Failed to fetch instruments');
        const data = await response.json();
        setInstruments(data.instruments || []);
        setFrontMonth(data.front_month);
        setSummary(data.summary || null);
      } catch (err) {
        console.error('Error fetching instruments:', err);
        setError(err.message);
      } finally {
        setLoadingInstruments(false);
      }
    };
    
    fetchInstruments();
  }, [selectedDate]);

  // Fetch forward-looking data when a date is selected
  useEffect(() => {
    if (!selectedDate) {
      setForwardData(null);
      return;
    }

    const fetchForwardData = async () => {
      setLoadingForward(true);
      setForwardError(null);
      try {
        const response = await fetch(`${API_BASE_URL}/forward-data/${selectedDate}`);
        if (!response.ok) throw new Error(`Forward data endpoint returned ${response.status} — restart your backend server`);
        const data = await response.json();
        setForwardData(data);
      } catch (err) {
        console.error('Error fetching forward data:', err);
        setForwardError(err.message);
        setForwardData(null);
      } finally {
        setLoadingForward(false);
      }
    };
    
    fetchForwardData();
  }, [selectedDate]);

  // Navigation handlers
  const handlePrevMonth = useCallback(() => {
    if (currentMonth === 0) {
      setCurrentMonth(11);
      setCurrentYear(y => y - 1);
    } else {
      setCurrentMonth(m => m - 1);
    }
  }, [currentMonth]);

  const handleNextMonth = useCallback(() => {
    if (currentMonth === 11) {
      setCurrentMonth(0);
      setCurrentYear(y => y + 1);
    } else {
      setCurrentMonth(m => m + 1);
    }
  }, [currentMonth]);

  const handleSelectDate = useCallback((date) => {
    setSelectedDate(date);
  }, []);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className={`p-6 rounded-lg ${
        isDarkMode 
          ? 'bg-gradient-to-r from-amber-800 via-orange-700 to-red-800' 
          : 'bg-gradient-to-r from-amber-500 to-orange-500'
      } text-white`}>
        <div className="flex items-center">
          <h2 className="text-3xl font-black tracking-tight mb-2" style={{ textShadow: '0 2px 4px rgba(0,0,0,0.3)' }}>
            Day<span className="text-amber-200">gent</span>{' '}
            <span className="text-xl font-semibold text-amber-100">DataBento Dashboard</span>
          </h2>
          <InfoTooltip 
            id="databento-dashboard" 
            content={
              <div>
                <p className="font-semibold mb-2">📊 DataBento Dashboard</p>
                <p className="mb-2">ES Futures calendar trading view:</p>
                <ul className="list-disc list-inside space-y-1 text-xs">
                  <li><strong>Calendar View:</strong> Click any trading day to see instruments</li>
                  <li><strong>Outrights:</strong> Individual ES contracts (ESH6, ESM6, etc.)</li>
                  <li><strong>Spreads:</strong> Calendar spreads sorted by liquidity</li>
                  <li><strong>Volume Ranking:</strong> #1 = most liquid instrument to trade</li>
                </ul>
              </div>
            } 
            isDarkMode={isDarkMode} 
          />
        </div>
        <p className="text-amber-100">ES Futures & Calendar Spreads • Daily Liquidity Analysis</p>
        
        {dateRange && (
          <div className={`mt-4 text-sm rounded p-2 ${isDarkMode ? 'bg-black/30' : 'bg-white/20'}`}>
            📅 Data Range: {dateRange.earliest} to {dateRange.latest} ({dateRange.total_days} trading days)
            {selectedDate && (
              <span className="ml-4 px-2 py-1 rounded text-xs bg-amber-600">
                📍 Selected: {selectedDate}
              </span>
            )}
          </div>
        )}
      </div>

      {/* Error Display */}
      {error && (
        <div className={`rounded-lg p-4 border ${
          isDarkMode 
            ? 'bg-red-900/20 border-red-800 text-red-300' 
            : 'bg-red-50 border-red-200 text-red-800'
        }`}>
          <div className="flex items-center">
            <span className="mr-2">⚠️</span>
            <span>{error}</span>
          </div>
        </div>
      )}

      {/* Main Content Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Calendar */}
        <TradingCalendar
          tradingDates={tradingDates}
          selectedDate={selectedDate}
          onSelectDate={handleSelectDate}
          currentMonth={currentMonth}
          currentYear={currentYear}
          onPrevMonth={handlePrevMonth}
          onNextMonth={handleNextMonth}
          isDarkMode={isDarkMode}
        />

        {/* Quick Stats */}
        <div className={`rounded-lg shadow-md p-6 ${isDarkMode ? 'bg-gray-800' : 'bg-white'}`}>
          <h3 className={`text-lg font-semibold mb-4 ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>
            Quick Stats
          </h3>
          
          <div className="grid grid-cols-2 gap-4">
            <div className={`p-4 rounded-lg ${isDarkMode ? 'bg-gray-700' : 'bg-gray-50'}`}>
              <div className="text-3xl mb-1">📅</div>
              <div className={`text-sm ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>
                Trading Days This Month
              </div>
              <div className={`text-2xl font-bold ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>
                {tradingDates.length}
              </div>
            </div>
            
            <div className={`p-4 rounded-lg ${isDarkMode ? 'bg-gray-700' : 'bg-gray-50'}`}>
              <div className="text-3xl mb-1">📊</div>
              <div className={`text-sm ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>
                Instruments Selected
              </div>
              <div className={`text-2xl font-bold ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>
                {instruments.length}
              </div>
            </div>
            
            <div className={`p-4 rounded-lg ${isDarkMode ? 'bg-gray-700' : 'bg-gray-50'}`}>
              <div className="text-3xl mb-1">📈</div>
              <div className={`text-sm ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>
                Outrights
              </div>
              <div className={`text-2xl font-bold text-blue-500`}>
                {instruments.filter(i => i.type === 'outright').length}
              </div>
            </div>
            
            <div className={`p-4 rounded-lg ${isDarkMode ? 'bg-gray-700' : 'bg-gray-50'}`}>
              <div className="text-3xl mb-1">🔄</div>
              <div className={`text-sm ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>
                Spreads
              </div>
              <div className={`text-2xl font-bold text-purple-500`}>
                {instruments.filter(i => i.type === 'spread').length}
              </div>
            </div>
          </div>

          {/* Top Instrument Highlight */}
          {instruments.length > 0 && (
            <div className={`mt-4 p-4 rounded-lg border-2 ${
              isDarkMode 
                ? 'bg-green-900/30 border-green-700' 
                : 'bg-green-50 border-green-300'
            }`}>
              <div className={`text-sm font-semibold mb-1 ${
                isDarkMode ? 'text-green-400' : 'text-green-700'
              }`}>
                🏆 Most Liquid Instrument
              </div>
              <div className={`text-xl font-bold font-mono ${
                isDarkMode ? 'text-white' : 'text-gray-900'
              }`}>
                {instruments[0].symbol}
              </div>
              <div className={`text-sm ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>
                {instruments[0].type === 'outright' ? '📈 Outright' : '🔄 Spread'} • 
                Volume: {instruments[0].volume?.toLocaleString()}
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Forward-Looking Data Panel */}
      {selectedDate && (
        <ForwardDataPanel
          forwardData={forwardData}
          selectedDate={selectedDate}
          loading={loadingForward}
          error={forwardError}
          isDarkMode={isDarkMode}
        />
      )}

      {/* Instruments Table */}
      <InstrumentsTable
        instruments={instruments}
        selectedDate={selectedDate}
        frontMonth={frontMonth}
        summary={summary}
        loading={loadingInstruments}
        isDarkMode={isDarkMode}
        selectedInstrumentSymbol={selectedInstrumentSymbol}
        onSelectInstrument={setSelectedInstrumentSymbol}
      />

      {/* Instrument 1m chart (pop out when row is clicked) */}
      {selectedDate && selectedInstrumentSymbol && (
        <div className="mt-6">
          <InstrumentChart
            selectedDate={selectedDate}
            symbol={selectedInstrumentSymbol}
            isDarkMode={isDarkMode}
          />
        </div>
      )}
    </div>
  );
};

export default DBentoDashboard;
