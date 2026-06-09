import { useState, useEffect, useRef, useCallback, useMemo } from 'react';
// Lightweight Charts v5 imports
import { createChart, CandlestickSeries, LineSeries, AreaSeries, ColorType, createSeriesMarkers, CrosshairMode } from 'lightweight-charts';
import { useTrading } from '../context/TradingContext';
import { useDateRanges } from '../hooks/useDateRanges';
import { FETCH_MODES, DATE_RANGE_TYPES } from '../utils/constants';
import SelectedCandlesPanel from './shared/SelectedCandlesPanel.jsx';
import GameModeController from './Game.jsx';

// ============================================================================
// CONSTANTS
// ============================================================================

const API_BASE_URL = 'http://localhost:8000/api/trading';

const CHART_COLORS = {
  light: {
    background: '#ffffff',
    text: '#374151',
    grid: '#f3f4f6',
    border: '#d1d5db',
    crosshair: '#3b82f6',
    upColor: '#22c55e',
    downColor: '#ef4444',
    selectedColor: '#3b82f6',
    selectedBorder: '#2563eb',
  },
  dark: {
    background: '#1f2937',
    text: '#e5e7eb',
    grid: '#374151',
    border: '#4b5563',
    crosshair: '#60a5fa',
    upColor: '#22c55e',
    downColor: '#ef4444',
    selectedColor: '#3b82f6',
    selectedBorder: '#2563eb',
  }
};

const MARKER_CONFIG = {
  tjrHigh: { color: '#22c55e', shape: 'arrowUp', text: 'T', size: 2 },
  tjrLow: { color: '#ef4444', shape: 'arrowDown', text: '⊥', size: 2 },
  swingHigh: { color: '#2563eb', shape: 'arrowUp', text: '▲', size: 2 },
  swingLow: { color: '#f59e0b', shape: 'arrowDown', text: '▼', size: 2 },
  fvgGreen: { color: '#22c55e', shape: 'square', text: 'FVG+', size: 3 },
  fvgRed: { color: '#ef4444', shape: 'square', text: 'FVG-', size: 3 },
};

const TIMEFRAME_MINUTES = {
  '1m': 1, '5m': 5, '15m': 15, '30m': 30,
  '1h': 60, '4h': 240, '1d': 1440
};

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

const formatPrice = (price) => price ? price.toFixed(4) : 'N/A';
const formatTimestamp = (timestamp) => new Date(timestamp).toLocaleString();

const formatTimeRange = (ms) => {
  const hours = Math.floor(ms / (1000 * 60 * 60));
  const minutes = Math.floor((ms % (1000 * 60 * 60)) / (1000 * 60));
  if (hours > 24) return `${Math.floor(hours / 24)}d ${hours % 24}h`;
  if (hours > 0) return `${hours}h ${minutes}m`;
  return `${minutes}m`;
};

const timestampToUnix = (ts) => Math.floor(new Date(ts).getTime() / 1000);

// ============================================================================
// REUSABLE COMPONENTS
// ============================================================================

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

// ============================================================================
// DATA SELECTION CONTROLS COMPONENT
// ============================================================================

const DataSelectionControls = ({ handleRefresh, isDarkMode, dashboardType = 'chart' }) => {
  const {
    selectedSymbol, setSelectedSymbol,
    selectedTimeframe, setSelectedTimeframe,
    rowLimit, setRowLimit,
    sortOrder, setSortOrder,
    fetchMode, setFetchMode,
    dateRangeType, setDateRangeType,
    startDate, setStartDate,
    endDate, setEndDate
  } = useTrading();

  const { dateRanges, loading: dateRangesLoading } = useDateRanges(selectedSymbol, selectedTimeframe);
  const [pendingStartDate, setPendingStartDate] = useState(startDate);
  const [pendingEndDate, setPendingEndDate] = useState(endDate);
  const [pendingDateRangeType, setPendingDateRangeType] = useState(dateRangeType);

  useEffect(() => {
    setPendingStartDate(startDate);
    setPendingEndDate(endDate);
    setPendingDateRangeType(dateRangeType);
  }, [startDate, endDate, dateRangeType]);

  useEffect(() => {
    if (!dateRanges) return;
    if (pendingDateRangeType === DATE_RANGE_TYPES.EARLIEST_TO_DATE) {
        setPendingStartDate(dateRanges.earliest);
    } else if (pendingDateRangeType === DATE_RANGE_TYPES.DATE_TO_LATEST) {
        setPendingEndDate(dateRanges.latest);
    } else if (pendingDateRangeType === DATE_RANGE_TYPES.DATE_TO_DATE) {
        if (!pendingStartDate) setPendingStartDate(dateRanges.earliest);
        if (!pendingEndDate) setPendingEndDate(dateRanges.latest);
    }
  }, [pendingDateRangeType, dateRanges]);

  const formatDatePart = (ds) => ds ? new Date(ds).toISOString().slice(0, 10) : '';
  const formatTimePart = (ds) => ds ? new Date(ds).toISOString().slice(11, 16) : '';
  const combineDateAndTime = (d, t) => d ? new Date(`${d}T${t || '00:00'}:00.000Z`).toISOString() : null;

  const hasPendingChanges = () => 
    pendingDateRangeType !== dateRangeType || pendingStartDate !== startDate || pendingEndDate !== endDate;

  const applyDateChanges = () => {
    setDateRangeType?.(pendingDateRangeType);
    setStartDate?.(pendingStartDate);
    setEndDate?.(pendingEndDate);
  };

  const resetDateChanges = () => {
    setPendingDateRangeType(dateRangeType);
    setPendingStartDate(startDate);
    setPendingEndDate(endDate);
  };

  const isStartDateDisabled = pendingDateRangeType === DATE_RANGE_TYPES.EARLIEST_TO_DATE;
  const isEndDateDisabled = pendingDateRangeType === DATE_RANGE_TYPES.DATE_TO_LATEST;

  const inputClass = (disabled) => `w-full rounded-md px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 transition-colors ${
    disabled 
      ? isDarkMode ? 'border-gray-600 bg-gray-600 text-gray-400 cursor-not-allowed' : 'border-gray-200 bg-gray-100 text-gray-500 cursor-not-allowed'
      : isDarkMode ? 'border-gray-600 bg-gray-800 text-white' : 'border-gray-300 bg-white text-gray-900'
  }`;

  return (
    <div className={`rounded-lg shadow-md p-6 mb-8 transition-colors ${isDarkMode ? 'bg-gray-800' : 'bg-white'}`}>
      <div className="flex items-center justify-between mb-4">
        <h3 className={`text-lg font-semibold ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>Data Selection & Controls</h3>
          <button
            onClick={handleRefresh}
            className={`px-3 py-2 text-sm rounded-md transition-colors flex items-center gap-1 ${
            isDarkMode ? 'bg-blue-800 hover:bg-blue-700 text-blue-300' : 'bg-blue-100 hover:bg-blue-200 text-blue-700'
            }`}
          >
          🔄 Refresh {dashboardType === 'chart' ? 'Chart' : 'Data'}
          </button>
      </div>

      {/* Fetch Mode Toggle */}
      <div className="mb-4">
        <div className="flex items-center space-x-4">
          <span className={`text-sm font-medium ${isDarkMode ? 'text-gray-300' : 'text-gray-700'}`}>Fetch Mode:</span>
          <div className={`flex rounded-lg p-1 ${isDarkMode ? 'bg-gray-700' : 'bg-gray-100'}`}>
            {[FETCH_MODES.LIMIT, FETCH_MODES.DATE_RANGE].map(mode => (
            <button
                key={mode}
                onClick={() => setFetchMode(mode)}
                className={`px-3 py-1 rounded-md text-sm font-medium transition-all ${
                  fetchMode === mode
                    ? isDarkMode ? 'bg-gray-600 text-white shadow-sm' : 'bg-white text-gray-900 shadow-sm'
                    : isDarkMode ? 'text-gray-300 hover:text-white' : 'text-gray-600 hover:text-gray-800'
                }`}
              >
                {mode === FETCH_MODES.LIMIT ? '📊 Record Limit' : '📅 Date Range'}
            </button>
            ))}
          </div>
          {dateRanges && (
            <div className={`text-xs ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>
              Available: {new Date(dateRanges.earliest).toLocaleDateString()} - {new Date(dateRanges.latest).toLocaleDateString()}
              {dateRangesLoading && ' (loading...)'}
            </div>
          )}
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <div>
          <label className={`block text-sm font-medium mb-2 ${isDarkMode ? 'text-gray-300' : 'text-gray-700'}`}>Symbol</label>
          <select
            value={selectedSymbol}
            onChange={(e) => setSelectedSymbol(e.target.value)}
            className={`w-full rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500 ${
              isDarkMode ? 'border-gray-600 bg-gray-700 text-white' : 'border-gray-300 bg-white text-gray-900'
            }`}
          >
            <option value="es">ES (E-mini S&P 500)</option>
            <option value="eurusd">EURUSD (Euro/US Dollar)</option>
            <option value="spy">SPY (SPDR S&P 500 ETF)</option>
          </select>
        </div>
        
        <div>
          <label className={`block text-sm font-medium mb-2 ${isDarkMode ? 'text-gray-300' : 'text-gray-700'}`}>Timeframe</label>
          <select
            value={selectedTimeframe}
            onChange={(e) => setSelectedTimeframe(e.target.value)}
            className={`w-full rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500 ${
              isDarkMode ? 'border-gray-600 bg-gray-700 text-white' : 'border-gray-300 bg-white text-gray-900'
            }`}
          >
            {Object.keys(TIMEFRAME_MINUTES).map(tf => (
              <option key={tf} value={tf}>{tf === '1d' ? '1 Day' : tf}</option>
            ))}
          </select>
        </div>

        {fetchMode === FETCH_MODES.LIMIT ? (
          <div>
            <label className={`block text-sm font-medium mb-2 ${isDarkMode ? 'text-gray-300' : 'text-gray-700'}`}>Data Limit</label>
            <select
              value={rowLimit}
              onChange={(e) => setRowLimit(Number(e.target.value))}
              className={`w-full rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500 ${
                isDarkMode ? 'border-gray-600 bg-gray-700 text-white' : 'border-gray-300 bg-white text-gray-900'
              }`}
            >
              {[25, 50, 100, 250, 500, 1000, 2000].map(n => (
                <option key={n} value={n}>{n} records</option>
              ))}
            </select>
          </div>
        ) : (
          <div>
            <label className={`block text-sm font-medium mb-2 ${isDarkMode ? 'text-gray-300' : 'text-gray-700'}`}>Date Range Type</label>
            <select
              value={pendingDateRangeType || DATE_RANGE_TYPES.EARLIEST_TO_DATE}
              onChange={(e) => setPendingDateRangeType(e.target.value)}
              className={`w-full rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500 ${
                isDarkMode ? 'border-gray-600 bg-gray-700 text-white' : 'border-gray-300 bg-white text-gray-900'
              }`}
            >
              <option value={DATE_RANGE_TYPES.EARLIEST_TO_DATE}>📅 Earliest to Date</option>
              <option value={DATE_RANGE_TYPES.DATE_TO_DATE}>📅 Date to Date</option>
              <option value={DATE_RANGE_TYPES.DATE_TO_LATEST}>📅 Date to Latest</option>
            </select>
          </div>
        )}

        <div>
          <label className={`block text-sm font-medium mb-2 ${isDarkMode ? 'text-gray-300' : 'text-gray-700'}`}>Sort Order</label>
          <select
            value={sortOrder || 'desc'}
            onChange={(e) => setSortOrder?.(e.target.value)}
            className={`w-full rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500 ${
              isDarkMode ? 'border-gray-600 bg-gray-700 text-white' : 'border-gray-300 bg-white text-gray-900'
            }`}
          >
            <option value="desc">⬇ Descending (Newest first)</option>
            <option value="asc">⬆ Ascending (Oldest first)</option>
          </select>
        </div>
      </div>

      {/* Date Range Inputs */}
      {fetchMode === FETCH_MODES.DATE_RANGE && (
        <div className={`mt-4 p-4 rounded-lg border ${isDarkMode ? 'bg-gray-700 border-gray-600' : 'bg-gray-50 border-gray-200'}`}>
          <h4 className={`text-md font-medium mb-3 ${isDarkMode ? 'text-white' : 'text-gray-800'}`}>Date Range Configuration</h4>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className={`block text-sm font-medium mb-2 ${isDarkMode ? 'text-gray-300' : 'text-gray-700'}`}>
                Start Date {isStartDateDisabled && <span className="text-xs text-gray-500">(Auto: Earliest)</span>}
              </label>
              <div className="grid grid-cols-2 gap-2">
                <input type="date" value={formatDatePart(pendingStartDate)} disabled={isStartDateDisabled}
                  onChange={(e) => setPendingStartDate(combineDateAndTime(e.target.value, formatTimePart(pendingStartDate)))}
                  className={inputClass(isStartDateDisabled)} />
                <input type="time" value={formatTimePart(pendingStartDate)} disabled={isStartDateDisabled}
                  onChange={(e) => setPendingStartDate(combineDateAndTime(formatDatePart(pendingStartDate), e.target.value))}
                  className={inputClass(isStartDateDisabled)} />
                </div>
                </div>
            <div>
              <label className={`block text-sm font-medium mb-2 ${isDarkMode ? 'text-gray-300' : 'text-gray-700'}`}>
                End Date {isEndDateDisabled && <span className="text-xs text-gray-500">(Auto: Latest)</span>}
              </label>
              <div className="grid grid-cols-2 gap-2">
                <input type="date" value={formatDatePart(pendingEndDate)} disabled={isEndDateDisabled}
                  onChange={(e) => setPendingEndDate(combineDateAndTime(e.target.value, formatTimePart(pendingEndDate)))}
                  className={inputClass(isEndDateDisabled)} />
                <input type="time" value={formatTimePart(pendingEndDate)} disabled={isEndDateDisabled}
                  onChange={(e) => setPendingEndDate(combineDateAndTime(formatDatePart(pendingEndDate), e.target.value))}
                  className={inputClass(isEndDateDisabled)} />
                </div>
            </div>
          </div>

          <div className="flex justify-end gap-2 mt-4">
            {hasPendingChanges() && (
              <button onClick={resetDateChanges} className={`px-3 py-2 text-sm rounded-md ${
                isDarkMode ? 'bg-gray-600 text-gray-200 hover:bg-gray-500' : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
              }`}>Reset</button>
            )}
            <button onClick={applyDateChanges} disabled={!hasPendingChanges()}
              className={`px-4 py-2 text-sm rounded-md ${
                !hasPendingChanges()
                  ? isDarkMode ? 'bg-gray-600 text-gray-500 cursor-not-allowed' : 'bg-gray-300 text-gray-500 cursor-not-allowed'
                  : isDarkMode ? 'bg-blue-600 text-white hover:bg-blue-500' : 'bg-blue-500 text-white hover:bg-blue-600'
              }`}
            >{hasPendingChanges() ? 'Apply Date Range' : 'No Changes'}</button>
          </div>
        </div>
      )}
    </div>
  );
};

// ============================================================================
// MARKER LABELS CONTROL COMPONENT (Combined TJR, Swing, FVG)
// ============================================================================

const MarkersControl = ({ isDarkMode, markerState, setMarkerState, labelsData }) => {
  const { selectedSymbol, selectedTimeframe } = useTrading();
  const { tjrLabels, swingLabels, fvgLabels, loading, error } = labelsData;

  const hasTjrData = tjrLabels && tjrLabels.length > 0;
  const hasSwingData = swingLabels && swingLabels.length > 0;
  const hasFvgData = fvgLabels && fvgLabels.length > 0;

  const ToggleButton = ({ label, isOn, onToggle, colorOn, colorClass }) => (
    <div className="flex items-center gap-2">
      <label className={`text-sm font-medium ${isDarkMode ? colorClass : colorClass}`}>{label}:</label>
      <button
        onClick={onToggle}
        className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors focus:outline-none focus:ring-2 focus:ring-offset-2 ${
          isOn ? `${colorOn} focus:ring-${colorOn.split('-')[1]}-500` : isDarkMode ? 'bg-gray-600' : 'bg-gray-200'
        }`}
      >
        <span className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${isOn ? 'translate-x-6' : 'translate-x-1'}`} />
      </button>
      <span className={`text-xs ${isOn ? colorClass : isDarkMode ? 'text-gray-500' : 'text-gray-400'}`}>{isOn ? 'ON' : 'OFF'}</span>
    </div>
  );

  return (
    <div className={`rounded-lg shadow-md ${isDarkMode ? 'bg-gray-800' : 'bg-white'}`}>
      <div className={`px-6 py-4 border-b ${isDarkMode ? 'border-gray-700' : 'border-gray-200'}`}>
        <div className="flex items-center justify-between flex-wrap gap-4">
          <div className="flex items-center">
            <h3 className={`text-lg font-semibold ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>
              Chart Markers
              {(hasTjrData || hasSwingData || hasFvgData) && (
                <span className="ml-2 text-xs px-2 py-1 rounded-full bg-green-100 text-green-800">✅ Available</span>
              )}
            </h3>
            <InfoTooltip id="markers-info" content={
              <div>
                <p className="font-semibold mb-2">🏷️ Chart Markers</p>
                <ul className="list-disc list-inside space-y-1 text-xs">
                  <li><strong>TJR High/Low:</strong> Trend Junction markers</li>
                  <li><strong>Swing High/Low:</strong> Pivot point markers</li>
                  <li><strong>FVG Green/Red:</strong> Fair Value Gap boxes</li>
                </ul>
              </div>
            } isDarkMode={isDarkMode} />
      </div>
      
          <div className="flex items-center gap-4 flex-wrap">
            {hasTjrData && (
              <>
                <ToggleButton label="TJR Highs" isOn={markerState.showTjrHighs} colorOn="bg-green-600" colorClass="text-green-600"
                  onToggle={() => setMarkerState(s => ({ ...s, showTjrHighs: !s.showTjrHighs }))} />
                <ToggleButton label="TJR Lows" isOn={markerState.showTjrLows} colorOn="bg-red-600" colorClass="text-red-600"
                  onToggle={() => setMarkerState(s => ({ ...s, showTjrLows: !s.showTjrLows }))} />
              </>
            )}
            {hasSwingData && (
              <>
                <ToggleButton label="Swing Highs" isOn={markerState.showSwingHighs} colorOn="bg-blue-600" colorClass="text-blue-600"
                  onToggle={() => setMarkerState(s => ({ ...s, showSwingHighs: !s.showSwingHighs }))} />
                <ToggleButton label="Swing Lows" isOn={markerState.showSwingLows} colorOn="bg-orange-500" colorClass="text-orange-500"
                  onToggle={() => setMarkerState(s => ({ ...s, showSwingLows: !s.showSwingLows }))} />
              </>
            )}
            {hasFvgData && (
              <>
                <ToggleButton label="FVG Green" isOn={markerState.showFvgGreens} colorOn="bg-green-600" colorClass="text-green-600"
                  onToggle={() => setMarkerState(s => ({ ...s, showFvgGreens: !s.showFvgGreens }))} />
                <ToggleButton label="FVG Red" isOn={markerState.showFvgReds} colorOn="bg-red-600" colorClass="text-red-600"
                  onToggle={() => setMarkerState(s => ({ ...s, showFvgReds: !s.showFvgReds }))} />
                  </>
                )}
              </div>
              </div>

        {loading && <div className={`mt-2 text-sm ${isDarkMode ? 'text-blue-400' : 'text-blue-600'}`}>🔄 Loading labels...</div>}
        {error && <div className={`mt-2 text-sm ${isDarkMode ? 'text-red-400' : 'text-red-600'}`}>❌ {error}</div>}
        
                <div className={`mt-2 text-xs ${isDarkMode ? 'text-gray-500' : 'text-gray-500'}`}>
          {hasTjrData && <span className="mr-4">📊 TJR: {tjrLabels.length} labels</span>}
          {hasSwingData && <span className="mr-4">🔵 Swing: {swingLabels.length} labels</span>}
          {hasFvgData && <span>🟢 FVG: {fvgLabels.length} gaps</span>}
                </div>
              </div>
    </div>
  );
};

// ============================================================================
// CUSTOM HOOK: USE LABELS DATA
// ============================================================================

const useLabelsData = (symbol, timeframe) => {
  const [state, setState] = useState({
    tjrLabels: [],
    swingLabels: [],
    fvgLabels: [],
    loading: false,
    error: null
  });

  useEffect(() => {
    if (!symbol || !timeframe) return;

    let isCancelled = false;
    setState(s => ({ ...s, loading: true, error: null }));

    const fetchAll = async () => {
      try {
        const [tjrRes, swingRes, fvgRes] = await Promise.allSettled([
          fetch(`${API_BASE_URL}/labels/${symbol}/${timeframe}`),
          fetch(`${API_BASE_URL}/swing-labels/${symbol}/${timeframe}`),
          fetch(`${API_BASE_URL}/fvg-labels/${symbol}/${timeframe}`)
        ]);

        // Don't update state if cancelled
        if (isCancelled) return;

        const tjrLabels = tjrRes.status === 'fulfilled' && tjrRes.value.ok ? await tjrRes.value.json() : [];
        const swingLabels = swingRes.status === 'fulfilled' && swingRes.value.ok ? await swingRes.value.json() : [];
        const fvgLabels = fvgRes.status === 'fulfilled' && fvgRes.value.ok ? await fvgRes.value.json() : [];

        if (isCancelled) return;

        setState({
          tjrLabels: Array.isArray(tjrLabels) ? tjrLabels : [],
          swingLabels: Array.isArray(swingLabels) ? swingLabels : [],
          fvgLabels: Array.isArray(fvgLabels) ? fvgLabels : [],
          loading: false,
          error: null
        });
      } catch (err) {
        if (!isCancelled) {
          setState(s => ({ ...s, loading: false, error: err.message }));
        }
      }
    };

    fetchAll();
    return () => { isCancelled = true; };
  }, [symbol, timeframe]);

  return state;
};

// ============================================================================
// CUSTOM HOOK: USE CHART DATA
// ============================================================================

const useChartData = (symbol, timeframe, rowLimit, sortOrder, fetchMode, dateRangeType, startDate, endDate) => {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const isMountedRef = useRef(true);

  const fetchData = useCallback(async () => {
    if (!symbol || !timeframe) return;
      
      setLoading(true);
      setError(null);
      
      try {
      // Try hybrid endpoint first for seamless backtest+fronttest data
      let url = `${API_BASE_URL}/hybrid/${symbol}/${timeframe}`;
      const params = new URLSearchParams();

      if (fetchMode === FETCH_MODES.DATE_RANGE) {
        if (dateRangeType === DATE_RANGE_TYPES.EARLIEST_TO_DATE && endDate) params.append('end_date', endDate);
        else if (dateRangeType === DATE_RANGE_TYPES.DATE_TO_LATEST && startDate) params.append('start_date', startDate);
        else if (dateRangeType === DATE_RANGE_TYPES.DATE_TO_DATE) {
          if (startDate) params.append('start_date', startDate);
          if (endDate) params.append('end_date', endDate);
        }
        params.append('limit', '10000');
          } else {
        params.append('limit', String(rowLimit));
      }

      console.log(`📊 Chart fetching: ${url}?${params}`);
      let response = await fetch(`${url}?${params}`);
      let result = null;

      if (response.ok) {
        result = await response.json();
        console.log(`✅ Chart hybrid data received: ${result?.data?.length || 0} candles`);
        } else {
        // Fallback to regular data endpoint
        console.log(`⚠️ Hybrid failed (${response.status}), trying regular endpoint...`);
        url = `${API_BASE_URL}/data/${symbol}/${timeframe}`;
        params.append('order', sortOrder || 'desc');
        params.append('sort_by', 'timestamp');
        response = await fetch(`${url}?${params}`);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        result = await response.json();
        console.log(`✅ Chart regular data received: ${result?.data?.length || 0} candles`);
      }

      // Don't update state if unmounted
      if (!isMountedRef.current) return;

      // Normalize and sort data ascending for chart display
      let chartData = result?.data || [];
      chartData = [...chartData].sort((a, b) => new Date(a.timestamp) - new Date(b.timestamp));

      // Apply row limit if using limit mode
      if (fetchMode === FETCH_MODES.LIMIT && chartData.length > rowLimit) {
        chartData = chartData.slice(-rowLimit);
      }

      console.log(`📊 Chart data prepared: ${chartData.length} candles after processing`);
      setData({ ...result, data: chartData });
    } catch (err) {
      console.error(`❌ Chart fetch error:`, err);
      if (isMountedRef.current) {
          setError(err.message);
        }
      } finally {
      if (isMountedRef.current) {
        setLoading(false);
      }
    }
  }, [symbol, timeframe, rowLimit, sortOrder, fetchMode, dateRangeType, startDate, endDate]);

  useEffect(() => {
    isMountedRef.current = true;
    fetchData();
    return () => { isMountedRef.current = false; };
  }, [fetchData]);

  return { data, loading, error, refetch: fetchData };
};

// ============================================================================
// MAIN CHART COMPONENT
// ============================================================================

const TradingChartDashboard = ({ tables, isDarkMode }) => {
  const {
    selectedSymbol, selectedTimeframe, rowLimit, sortOrder, selectedCandles,
    addSelectedCandle, removeSelectedCandle, fetchMode, dateRangeType, startDate, endDate
  } = useTrading();

  // Chart state
  const [chartType, setChartType] = useState('candlestick');
  const [isSelectionMode, setIsSelectionMode] = useState(false);
  const [gameMode, setGameMode] = useState(false);
  const [chartReady, setChartReady] = useState(false);

  // Marker state (consolidated)
  const [markerState, setMarkerState] = useState({
    showTjrHighs: false, showTjrLows: false,
    showSwingHighs: false, showSwingLows: false,
    showFvgGreens: false, showFvgReds: false
  });

  // Refs for chart instances - STABLE, created once
  const chartContainerRef = useRef(null);
  const [chartContainerEl, setChartContainerEl] = useState(null);
  const chartRef = useRef(null);
  const seriesRef = useRef(null);
  const markersPrimitiveRef = useRef(null);

  // Callback ref so chart initialization can run when the container mounts
  // (important because we sometimes early-return during initial loading).
  const setChartContainerNode = useCallback((node) => {
    chartContainerRef.current = node;
    setChartContainerEl(node);
  }, []);

  // Data refs to avoid stale closures
  const chartDataRef = useRef([]);
  const labelsDataRef = useRef({ tjrLabels: [], swingLabels: [], fvgLabels: [] });

  // Colors based on theme - memoized to prevent chart recreation
  const colors = useMemo(() => isDarkMode ? CHART_COLORS.dark : CHART_COLORS.light, [isDarkMode]);

  // Fetch chart data
  const { data: chartData, loading, error, refetch } = useChartData(
    selectedSymbol, selectedTimeframe, rowLimit, sortOrder, fetchMode, dateRangeType, startDate, endDate
  );

  // Fetch labels data
  const labelsData = useLabelsData(selectedSymbol, selectedTimeframe);

  // Keep refs up to date
  useEffect(() => {
    if (chartData?.data) chartDataRef.current = chartData.data;
  }, [chartData]);

  useEffect(() => {
    labelsDataRef.current = labelsData;
  }, [labelsData]);

  // ========================================================================
  // BUILD MARKERS - Pure function, no side effects
  // ========================================================================
  const buildMarkers = useCallback((candles, labels, markers, selectedSet) => {
      const allMarkers = [];
    const { tjrLabels, swingLabels, fvgLabels } = labels;
    const { showTjrHighs, showTjrLows, showSwingHighs, showSwingLows, showFvgGreens, showFvgReds } = markers;

    if (!candles || candles.length === 0) return allMarkers;

    // Create timestamp lookup for quick matching
    const candlesByTime = new Map();
    candles.forEach((c, idx) => {
      const time = new Date(c.timestamp).getTime();
      candlesByTime.set(time, { candle: c, index: idx });
    });

    // Process TJR labels
    if ((showTjrHighs || showTjrLows) && tjrLabels && tjrLabels.length > 0) {
      const labelGroups = new Map();
      
      tjrLabels.forEach(label => {
        if (!label.pointer || label.pointer.length < 2) return;
        const key = `${label.label}_${label.pointer[0]}_${label.pointer[1]}`;
        if (!labelGroups.has(key)) labelGroups.set(key, { label, candles: [] });
        
        const startMs = new Date(label.pointer[0]).getTime();
        const endMs = new Date(label.pointer[1]).getTime();
        
        candles.forEach(c => {
          const cTime = new Date(c.timestamp).getTime();
          if (cTime >= startMs && cTime <= endMs) {
            labelGroups.get(key).candles.push(c);
          }
        });
      });

      labelGroups.forEach(({ label, candles: matched }) => {
        if (matched.length === 0) return;
        
        if (label.label === 'tjr_high' && showTjrHighs) {
          const highest = matched.reduce((a, b) => parseFloat(b.high) > parseFloat(a.high) ? b : a);
            allMarkers.push({
            time: timestampToUnix(highest.timestamp),
              position: 'aboveBar',
            ...MARKER_CONFIG.tjrHigh
          });
        }
        if (label.label === 'tjr_low' && showTjrLows) {
          const lowest = matched.reduce((a, b) => parseFloat(b.low) < parseFloat(a.low) ? b : a);
            allMarkers.push({
            time: timestampToUnix(lowest.timestamp),
              position: 'belowBar',
            ...MARKER_CONFIG.tjrLow
            });
          }
        });
      }
      
    // Process Swing labels (exact timestamp match)
    if ((showSwingHighs || showSwingLows) && swingLabels && swingLabels.length > 0) {
      swingLabels.forEach(label => {
        if (!label.pointer || label.pointer.length < 1) return;
        const labelMs = new Date(label.pointer[0]).getTime();
        const match = candlesByTime.get(labelMs);
        
        if (match) {
          if (label.label === 'swing_high' && showSwingHighs) {
              allMarkers.push({
              time: timestampToUnix(match.candle.timestamp),
                position: 'aboveBar',
              ...MARKER_CONFIG.swingHigh
              });
            }
          if (label.label === 'swing_low' && showSwingLows) {
              allMarkers.push({
              time: timestampToUnix(match.candle.timestamp),
                position: 'belowBar',
              ...MARKER_CONFIG.swingLow
            });
          }
        }
      });
    }

    // Process FVG labels (3-candle gaps)
    if ((showFvgGreens || showFvgReds) && fvgLabels && fvgLabels.length > 0) {
      fvgLabels.forEach(label => {
        if (!label.pointer || label.pointer.length < 3) return;
        
        const isGreen = label.label === 'fvg_green';
        if ((isGreen && !showFvgGreens) || (!isGreen && !showFvgReds)) return;

        const timestamps = label.pointer.map(p => new Date(p).getTime());
        const matchedCandles = timestamps.map(t => candlesByTime.get(t)).filter(Boolean);

        if (matchedCandles.length === 3) {
          // Add marker at the middle candle
          const midCandle = matchedCandles[1].candle;
          allMarkers.push({
            time: timestampToUnix(midCandle.timestamp),
            position: 'inBar',
            ...(isGreen ? MARKER_CONFIG.fvgGreen : MARKER_CONFIG.fvgRed)
          });
        }
      });
    }

    // Sort markers by time (required by lightweight-charts)
    allMarkers.sort((a, b) => a.time - b.time);
    
    return allMarkers;
  }, []);

  // ========================================================================
  // PREPARE CANDLESTICK DATA
  // ========================================================================
  const prepareCandlestickData = useCallback((candles, selectedSet) => {
    return candles.map(c => {
      const time = timestampToUnix(c.timestamp);
      const isSelected = selectedSet.has(c.timestamp);
      const isBullish = parseFloat(c.close) >= parseFloat(c.open);

      const base = {
        time,
        open: parseFloat(c.open),
        high: parseFloat(c.high),
        low: parseFloat(c.low),
        close: parseFloat(c.close),
      };

      if (isSelected) {
          return {
          ...base,
          color: colors.selectedColor,
          borderColor: colors.selectedBorder,
          wickColor: isBullish ? colors.upColor : colors.downColor,
        };
      }

      return base;
    });
  }, [colors]);

  // ========================================================================
  // CHART INITIALIZATION - Only runs once per container
  // ========================================================================
  useEffect(() => {
    // IMPORTANT: `chartContainerRef.current` may be null on the first effect run
    // if we early-return in render during loading. Using `chartContainerEl`
    // ensures we re-run init as soon as the container is actually mounted.
    if (!chartContainerEl) {
      console.log('⚠️ Chart container not ready');
      return;
    }

    console.log('🎨 Initializing chart...');

    // Clean up existing chart if any
    if (chartRef.current) {
      console.log('🧹 Cleaning up existing chart');
      try {
      chartRef.current.remove();
      } catch (e) {
        console.warn('Chart cleanup warning:', e);
      }
      chartRef.current = null;
      seriesRef.current = null;
      markersPrimitiveRef.current = null;
    }

    const container = chartContainerEl;
    const width = container.clientWidth || 800;
    const height = 400;

    console.log(`📐 Chart dimensions: ${width}x${height}`);

    // Create chart instance
    const chart = createChart(container, {
      width,
      height,
        layout: {
        background: { type: ColorType.Solid, color: colors.background },
        textColor: colors.text,
        },
        grid: {
        vertLines: { color: colors.grid },
        horzLines: { color: colors.grid },
        },
        crosshair: {
        mode: CrosshairMode.Normal,
        vertLine: { color: colors.crosshair, style: 2, width: 1, labelBackgroundColor: colors.crosshair },
        horzLine: { color: colors.crosshair, style: 2, width: 1, labelBackgroundColor: colors.crosshair },
      },
      rightPriceScale: { borderColor: colors.border },
      timeScale: { borderColor: colors.border, timeVisible: true, secondsVisible: false },
      });

      chartRef.current = chart;

    // Create series based on chart type
    let series;
      if (chartType === 'candlestick') {
      series = chart.addSeries(CandlestickSeries, {
        upColor: colors.upColor,
        downColor: colors.downColor,
          borderVisible: false,
        wickUpColor: colors.upColor,
        wickDownColor: colors.downColor,
        });
      } else if (chartType === 'line') {
      series = chart.addSeries(LineSeries, {
        color: colors.crosshair,
          lineWidth: 2,
        });
      } else {
      series = chart.addSeries(AreaSeries, {
          topColor: 'rgba(59,130,246,0.4)',
          bottomColor: 'rgba(59,130,246,0.0)',
        lineColor: colors.crosshair,
          lineWidth: 2,
        });
    }
    seriesRef.current = series;

    // Create markers primitive ONCE
    markersPrimitiveRef.current = createSeriesMarkers(series);

    console.log('✅ Chart initialized successfully');
    setChartReady(true);

      // Handle resize
      const handleResize = () => {
      if (chartRef.current && chartContainerRef.current) {
        chartRef.current.applyOptions({ width: chartContainerRef.current.clientWidth });
      }
    };
      window.addEventListener('resize', handleResize);

      return () => {
      console.log('🧹 Chart cleanup triggered');
        window.removeEventListener('resize', handleResize);
      if (chartRef.current) {
        try {
          chartRef.current.remove();
        } catch (e) {
          console.warn('Chart removal warning:', e);
        }
        chartRef.current = null;
        seriesRef.current = null;
        markersPrimitiveRef.current = null;
      }
      setChartReady(false);
    };
  }, [chartType, colors, chartContainerEl]); // Re-init when container mounts, chart type changes, or theme changes

  // ========================================================================
  // UPDATE CHART DATA - Updates existing series, no recreation
  // ========================================================================
  useEffect(() => {
    console.log(`🔄 Data update effect: chartReady=${chartReady}, hasData=${!!chartData?.data}, dataLength=${chartData?.data?.length || 0}`);
    
    if (!chartReady) {
      console.log('⏳ Waiting for chart to be ready...');
      return;
    }
    
    if (!seriesRef.current) {
      console.log('⏳ Waiting for series ref...');
      return;
    }
    
    if (!chartData?.data || chartData.data.length === 0) {
      console.log('⏳ Waiting for chart data...');
      return;
    }

    const candles = chartData.data;
    console.log(`📊 Updating chart with ${candles.length} candles`);
    
    const selectedSet = new Set(
      selectedCandles
        .filter(c => c.symbol === selectedSymbol && c.timeframe === selectedTimeframe)
        .map(c => c.timestamp)
    );

    try {
      // Prepare data based on chart type
      if (chartType === 'candlestick') {
        const candlestickData = prepareCandlestickData(candles, selectedSet);
        console.log(`🕯️ Setting ${candlestickData.length} candlestick data points`);
        seriesRef.current.setData(candlestickData);
      } else {
        const lineData = candles.map(c => ({
          time: timestampToUnix(c.timestamp),
          value: parseFloat(c.close)
        }));
        console.log(`📈 Setting ${lineData.length} line data points`);
        seriesRef.current.setData(lineData);
      }

      // Fit content only on initial data load (not on selection changes)
      if (!gameMode && !isSelectionMode && chartRef.current) {
        chartRef.current.timeScale().fitContent();
        console.log('✅ Chart data set and content fit');
      }
    } catch (e) {
      console.error('❌ Error setting chart data:', e);
    }
  }, [chartData, chartReady, chartType, selectedSymbol, selectedTimeframe, selectedCandles, prepareCandlestickData, gameMode, isSelectionMode]);

  // ========================================================================
  // UPDATE MARKERS - Updates existing primitive, no recreation
  // ========================================================================
  useEffect(() => {
    if (!chartReady || !markersPrimitiveRef.current) return;

    const candles = chartDataRef.current;
    const labels = labelsDataRef.current;
    const selectedSet = new Set(
      selectedCandles
        .filter(c => c.symbol === selectedSymbol && c.timeframe === selectedTimeframe)
        .map(c => c.timestamp)
    );

    const markers = buildMarkers(candles, labels, markerState, selectedSet);
    
    // Update markers primitive (not recreate)
    try {
      markersPrimitiveRef.current.setMarkers(markers);
      console.log(`✅ Updated ${markers.length} markers`);
    } catch (e) {
      console.warn('Could not update markers:', e);
    }
  }, [chartReady, markerState, labelsData, selectedCandles, selectedSymbol, selectedTimeframe, buildMarkers]);

  // ========================================================================
  // SELECTION HANDLING
  // ========================================================================
  const handleCandleClick = useCallback((e) => {
    if (!isSelectionMode || !chartRef.current || !chartData?.data) return;

    const container = chartContainerRef.current;
    if (!container) return;

    const rect = container.getBoundingClientRect();
    const x = e.clientX - rect.left;

    const timeScale = chartRef.current.timeScale();
    const logical = timeScale.coordinateToLogical(x);
    if (logical === null) return;

    const index = Math.round(logical);
    const candle = chartData.data[index];
    if (!candle) return;
    
    const candleId = `${selectedSymbol}_${selectedTimeframe}_${candle.timestamp}`;
    const isSelected = selectedCandles.some(c => c.id === candleId);
    
    if (isSelected) {
      removeSelectedCandle(candleId);
    } else {
      addSelectedCandle({
        id: candleId,
        symbol: selectedSymbol,
        timeframe: selectedTimeframe,
        timestamp: candle.timestamp,
        open: candle.open,
        high: candle.high,
        low: candle.low,
        close: candle.close,
        volume: candle.volume,
        change: candle.close - candle.open,
        changePercent: ((candle.close - candle.open) / candle.open) * 100,
        sourceIndex: index
      });
    }
  }, [isSelectionMode, chartData, selectedSymbol, selectedTimeframe, selectedCandles, addSelectedCandle, removeSelectedCandle]);

  // Keyboard handler for selection mode
  useEffect(() => {
    if (!isSelectionMode) return;
    
    const handleKeyDown = (e) => {
      if (e.key === 'Escape') setIsSelectionMode(false);
      if (e.key === 'Delete') {
        selectedCandles
          .filter(c => c.symbol === selectedSymbol && c.timeframe === selectedTimeframe)
          .forEach(c => removeSelectedCandle(c.id));
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isSelectionMode, selectedCandles, selectedSymbol, selectedTimeframe, removeSelectedCandle]);

  // ========================================================================
  // RENDER
  // ========================================================================
  if (loading && !chartData) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className={`animate-spin rounded-full h-12 w-12 border-b-2 ${isDarkMode ? 'border-blue-400' : 'border-blue-600'}`} />
        <span className={`ml-3 ${isDarkMode ? 'text-gray-300' : 'text-gray-600'}`}>Loading chart data...</span>
      </div>
    );
  }

  if (error) {
    return (
      <div className={`rounded-lg p-4 border ${isDarkMode ? 'bg-red-900/20 border-red-800' : 'bg-red-50 border-red-200'}`}>
        <div className="text-red-500 font-medium">Error loading chart data</div>
        <div className="text-red-500 text-sm mt-1">{error}</div>
        <button onClick={refetch} className="mt-2 px-3 py-1 bg-red-600 text-white rounded text-sm hover:bg-red-700">Retry</button>
      </div>
    );
  }

  const currentSelection = selectedCandles.filter(c => c.symbol === selectedSymbol && c.timeframe === selectedTimeframe);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className={`p-6 rounded-lg ${isDarkMode ? 'bg-gradient-to-r from-green-800 via-green-700 to-blue-800' : 'bg-gradient-to-r from-green-600 to-blue-600'} text-white`}>
        <div className="flex items-center">
          <h2 className="text-3xl font-black tracking-tight mb-2" style={{ textShadow: '0 2px 4px rgba(0,0,0,0.3)' }}>
            Day<span className="text-green-200">gent</span> <span className="text-xl font-semibold text-green-100">Chart Analysis</span>
          </h2>
        <InfoTooltip id="chart-dashboard" content={
          <div>
              <p className="font-semibold mb-2">📈 Chart Dashboard</p>
              <p>Professional trading charts with markers, selection tools, and game mode.</p>
          </div>
        } isDarkMode={isDarkMode} />
        </div>
        <p className="text-green-100">Interactive charting & technical analysis</p>
        <div className={`mt-4 text-sm rounded p-2 ${isDarkMode ? 'bg-black/30' : 'bg-white/20'}`}>
          📈 {selectedSymbol?.toUpperCase()} • {selectedTimeframe} • {chartData?.data?.length || 0} candles
          {currentSelection.length > 0 && (
            <span className="ml-4 px-2 py-1 rounded text-xs bg-blue-600">🔵 {currentSelection.length} selected</span>
          )}
        </div>
      </div>

      {/* Controls */}
      <DataSelectionControls handleRefresh={refetch} isDarkMode={isDarkMode} dashboardType="chart" />

      {/* Chart Container */}
      <div className={`rounded-lg shadow-md ${isDarkMode ? 'bg-gray-800' : 'bg-white'}`}>
        <div className={`px-6 py-4 border-b flex items-center justify-between ${isDarkMode ? 'border-gray-700' : 'border-gray-200'}`}>
          <div className="flex items-center">
            <h3 className={`text-lg font-semibold ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>Price Chart</h3>
            {chartReady && <GameModeController chartRef={chartRef} containerRef={chartContainerRef} isDarkMode={isDarkMode} />}
            <InfoTooltip id="price-chart" content={
              <div>
                <p className="font-semibold mb-2">📈 Interactive Chart</p>
                <p>Candlestick, line, or area charts with markers and selection.</p>
              </div>
            } isDarkMode={isDarkMode} />
          </div>
          
          <div className="flex items-center gap-3">
            {/* Chart Type */}
              <select
                value={chartType}
                onChange={(e) => setChartType(e.target.value)}
              className={`px-3 py-1 text-sm rounded-md border ${
                isDarkMode ? 'border-gray-600 bg-gray-700 text-white' : 'border-gray-300 bg-white text-gray-900'
                }`}
              >
                <option value="candlestick">🕯️ Candlestick</option>
              <option value="line">📈 Line</option>
              <option value="area">📊 Area</option>
              </select>

            {/* Selection Mode */}
                <button
              onClick={() => setIsSelectionMode(!isSelectionMode)}
              className={`flex items-center px-3 py-2 rounded-lg transition-all ${
                  isSelectionMode
                  ? 'bg-blue-500 text-white shadow-lg'
                  : isDarkMode ? 'bg-gray-700 text-gray-300 hover:bg-gray-600' : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
              }`}
            >
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" className="mr-2">
                <line x1="12" y1="2" x2="12" y2="22" />
                <line x1="2" y1="12" x2="22" y2="12" />
                <circle cx="12" cy="12" r="3" fill="none" />
                </svg>
              Select
              </button>
          </div>
        </div>

        {/* Chart Area */}
        <div className="p-6">
          {isSelectionMode && (
            <div className={`mb-4 p-3 rounded-lg border ${isDarkMode ? 'bg-blue-900/20 border-blue-800 text-blue-300' : 'bg-blue-50 border-blue-200 text-blue-700'}`}>
                    <span className="text-lg mr-2">🎯</span>
              <strong>Selection Mode:</strong> Click candles to select/deselect. Press <kbd className="px-1 rounded bg-blue-600 text-white">Esc</kbd> to exit.
            </div>
          )}
          
          <div 
            ref={setChartContainerNode}
            onClick={handleCandleClick}
            className={`w-full rounded border ${isDarkMode ? 'border-gray-600 bg-gray-900' : 'border-gray-200 bg-gray-50'}`}
            style={{ minHeight: '400px', cursor: isSelectionMode ? 'crosshair' : 'default' }}
          />

          {!chartData?.data && (
            <div className="flex items-center justify-center h-96 -mt-96">
              <div className="text-center">
                <div className="text-4xl mb-4">📊</div>
                <h3 className={`text-lg font-medium mb-2 ${isDarkMode ? 'text-gray-300' : 'text-gray-900'}`}>No Chart Data</h3>
                <p className={isDarkMode ? 'text-gray-400' : 'text-gray-500'}>Select a symbol and timeframe to load data</p>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Markers Control */}
      <MarkersControl
        isDarkMode={isDarkMode} 
        markerState={markerState}
        setMarkerState={setMarkerState}
        labelsData={labelsData}
            />

      {/* Selected Candles Panel */}
      <SelectedCandlesPanel isDarkMode={isDarkMode} canSelectCandles={true} />

      {/* OHLC Data Table */}
      {chartData?.data && chartData.data.length > 0 && (
        <div className={`rounded-lg shadow-md ${isDarkMode ? 'bg-gray-800' : 'bg-white'}`}>
          <div className={`px-6 py-4 border-b ${isDarkMode ? 'border-gray-700' : 'border-gray-200'}`}>
            <h3 className={`text-lg font-semibold ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>Recent OHLC Data</h3>
            <p className={`text-sm ${isDarkMode ? 'text-gray-400' : 'text-gray-600'}`}>Last 10 candles • {chartData.data.length} total</p>
          </div>
          
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className={isDarkMode ? 'bg-gray-700' : 'bg-gray-50'}>
                <tr>
                  {['Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Change'].map(h => (
                    <th key={h} className={`px-4 py-3 text-left text-xs font-medium uppercase ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody className={`divide-y ${isDarkMode ? 'bg-gray-800 divide-gray-700' : 'bg-white divide-gray-200'}`}>
                {chartData.data.slice(-10).reverse().map((c, i) => {
                  const change = c.close - c.open;
                  const pct = (change / c.open) * 100;
                  return (
                    <tr key={i} className={isDarkMode ? 'hover:bg-gray-700' : 'hover:bg-gray-50'}>
                      <td className={`px-4 py-3 text-sm font-mono ${isDarkMode ? 'text-gray-300' : 'text-gray-900'}`}>
                        {new Date(c.timestamp).toLocaleTimeString()}
                      </td>
                      <td className={`px-4 py-3 text-sm font-mono ${isDarkMode ? 'text-gray-300' : 'text-gray-900'}`}>{formatPrice(c.open)}</td>
                      <td className="px-4 py-3 text-sm font-mono font-semibold text-green-600">{formatPrice(c.high)}</td>
                      <td className="px-4 py-3 text-sm font-mono font-semibold text-red-600">{formatPrice(c.low)}</td>
                      <td className={`px-4 py-3 text-sm font-mono font-bold ${change >= 0 ? 'text-green-600' : 'text-red-600'}`}>{formatPrice(c.close)}</td>
                      <td className={`px-4 py-3 text-sm font-mono ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>{c.volume ? c.volume.toLocaleString() : 'N/A'}</td>
                      <td className={`px-4 py-3 text-sm font-mono ${change >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                        {change >= 0 ? '+' : ''}{formatPrice(change)}<br />
                        <span className="text-xs">{change >= 0 ? '+' : ''}{pct.toFixed(2)}%</span>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
};

export default TradingChartDashboard; 
