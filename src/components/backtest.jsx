import { useEffect, useMemo, useRef, useState, useCallback } from 'react';
import { createChart, CandlestickSeries, ColorType } from 'lightweight-charts';
import InfoTooltip from './shared/InfoTooltip.jsx';
import { useTrading } from '../context/TradingContext';
import { useDateRanges } from '../hooks/useDateRanges';
import { API_BASE_URL, FETCH_MODES, DATE_RANGE_TYPES } from '../utils/constants';

// Backtest Settings panel
const BacktestSettings = ({
  isDarkMode,
  fullData,
  currentIndex,
  setCurrentIndex,
  isPlaying,
  setIsPlaying,
  speedMs,
  setSpeedMs,
  stepSize,
  setStepSize,
  loopPlayback,
  setLoopPlayback
}) => {
  const total = fullData?.length || 0;
  const current = Math.min(Math.max(currentIndex, 0), Math.max(total - 1, 0));
  const currentCandle = total > 0 ? fullData[current] : null;

  const formatTs = (ts) => (ts ? new Date(ts).toLocaleString() : 'N/A');

  return (
    <div className={`rounded-lg shadow-md p-6 mb-6 transition-colors duration-200 ${
      isDarkMode ? 'bg-gray-800' : 'bg-white'
    }`}>
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center">
          <h3 className={`text-lg font-semibold ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>
            Backtest Settings
          </h3>
          <InfoTooltip
            id="backtest-settings"
            isDarkMode={isDarkMode}
            content={(
              <div>
                <p className="font-semibold mb-2">🧪 Backtest Playback</p>
                <ul className="list-disc list-inside text-xs space-y-1">
                  <li><strong>Happened:</strong> Candles up to the current index are rendered.</li>
                  <li><strong>Coming:</strong> Remaining candles are hidden until playback advances.</li>
                  <li><strong>Speed:</strong> Controls delay between candles during autoplay.</li>
                  <li><strong>Step:</strong> Move forward/back by N candles.</li>
                </ul>
              </div>
            )}
          />
        </div>
        <div className="text-xs opacity-75">
          {total > 0 ? (
            <span>
              Loaded {total} candles • Range {formatTs(fullData[0]?.timestamp)} → {formatTs(fullData[total - 1]?.timestamp)}
            </span>
          ) : (
            <span>No data loaded</span>
          )}
        </div>
      </div>

      {/* Playback controls */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {/* Position / progress */}
        <div>
          <label className={`block text-sm font-medium mb-2 ${isDarkMode ? 'text-gray-300' : 'text-gray-700'}`}>
            Progress
          </label>
          <input
            type="range"
            min={0}
            max={Math.max(total - 1, 0)}
            value={current}
            onChange={(e) => setCurrentIndex(parseInt(e.target.value))}
            className="w-full"
          />
          <div className={`mt-2 text-xs ${isDarkMode ? 'text-gray-400' : 'text-gray-600'}`}>
            {current + 1} / {total} candles happened
            {currentCandle && (
              <span className="ml-2">
                • Current: {formatTs(currentCandle.timestamp)}
              </span>
            )}
          </div>
        </div>

        {/* Speed / step */}
        <div>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className={`block text-sm font-medium mb-2 ${isDarkMode ? 'text-gray-300' : 'text-gray-700'}`}>Speed</label>
              <select
                value={speedMs}
                onChange={(e) => setSpeedMs(parseInt(e.target.value))}
                className={`w-full rounded-md px-3 py-2 transition-colors ${isDarkMode ? 'border-gray-600 bg-gray-700 text-white' : 'border-gray-300 bg-white text-gray-900'}`}
              >
                <option value={1500}>0.5x (1500ms)</option>
                <option value={1000}>1x (1000ms)</option>
                <option value={500}>2x (500ms)</option>
                <option value={250}>4x (250ms)</option>
                <option value={100}>10x (100ms)</option>
              </select>
            </div>
            <div>
              <label className={`block text-sm font-medium mb-2 ${isDarkMode ? 'text-gray-300' : 'text-gray-700'}`}>Step Size</label>
              <select
                value={stepSize}
                onChange={(e) => setStepSize(parseInt(e.target.value))}
                className={`w-full rounded-md px-3 py-2 transition-colors ${isDarkMode ? 'border-gray-600 bg-gray-700 text-white' : 'border-gray-300 bg-white text-gray-900'}`}
              >
                <option value={1}>1 candle</option>
                <option value={5}>5 candles</option>
                <option value={10}>10 candles</option>
                <option value={25}>25 candles</option>
              </select>
            </div>
          </div>
        </div>

        {/* Buttons / toggles */}
        <div className="flex items-end gap-2">
          <button
            onClick={() => setIsPlaying(!isPlaying)}
            className={`px-4 py-2 rounded-md text-sm font-medium ${
              isDarkMode ? 'bg-blue-600 hover:bg-blue-500 text-white' : 'bg-blue-600 hover:bg-blue-700 text-white'
            }`}
          >
            {isPlaying ? 'Pause' : 'Play'}
          </button>
          <button
            onClick={() => setCurrentIndex(Math.max(current - stepSize, 0))}
            className={`px-3 py-2 rounded-md text-sm ${isDarkMode ? 'bg-gray-700 hover:bg-gray-600 text-gray-200' : 'bg-gray-100 hover:bg-gray-200 text-gray-800'}`}
          >
            ◀ Step Back
          </button>
          <button
            onClick={() => setCurrentIndex(Math.min(current + stepSize, Math.max(total - 1, 0)))}
            className={`px-3 py-2 rounded-md text-sm ${isDarkMode ? 'bg-gray-700 hover:bg-gray-600 text-gray-200' : 'bg-gray-100 hover:bg-gray-200 text-gray-800'}`}
          >
            Step Fwd ▶
          </button>
          <label className="ml-auto inline-flex items-center gap-2 text-sm">
            <input type="checkbox" checked={loopPlayback} onChange={(e) => setLoopPlayback(e.target.checked)} />
            <span className={isDarkMode ? 'text-gray-300' : 'text-gray-700'}>Loop</span>
          </label>
        </div>
      </div>
    </div>
  );
};

// Data Selection & Controls (mirrors the one inside Chart.jsx)
const DataSelectionControls = ({ handleRefresh, isDarkMode }) => {
  const {
    selectedSymbol,
    setSelectedSymbol,
    selectedTimeframe,
    setSelectedTimeframe,
    rowLimit,
    setRowLimit,
    sortOrder,
    setSortOrder,
    fetchMode,
    setFetchMode,
    dateRangeType,
    setDateRangeType,
    startDate,
    setStartDate,
    endDate,
    setEndDate
  } = useTrading();

  const { dateRanges } = useDateRanges(selectedSymbol, selectedTimeframe);

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
    switch (pendingDateRangeType) {
      case DATE_RANGE_TYPES.EARLIEST_TO_DATE:
        setPendingStartDate(dateRanges.earliest);
        break;
      case DATE_RANGE_TYPES.DATE_TO_LATEST:
        setPendingEndDate(dateRanges.latest);
        break;
      case DATE_RANGE_TYPES.DATE_TO_DATE:
        if (!pendingStartDate) setPendingStartDate(dateRanges.earliest);
        if (!pendingEndDate) setPendingEndDate(dateRanges.latest);
        break;
    }
  }, [pendingDateRangeType, dateRanges]);

  const formatDatePart = (dateString) => {
    if (!dateString) return '';
    const date = new Date(dateString);
    return date.toISOString().slice(0, 10);
  };

  const formatTimePart = (dateString) => {
    if (!dateString) return '';
    const date = new Date(dateString);
    return date.toISOString().slice(11, 16);
  };

  const combineDateAndTime = (dateValue, timeValue) => {
    if (!dateValue) return null;
    const timeToUse = timeValue || '00:00';
    return new Date(`${dateValue}T${timeToUse}:00.000Z`).toISOString();
  };

  const hasPendingChanges = () => (
    pendingDateRangeType !== dateRangeType ||
    pendingStartDate !== startDate ||
    pendingEndDate !== endDate
  );

  const applyDateChanges = () => {
    if (setDateRangeType) setDateRangeType(pendingDateRangeType);
    if (setStartDate) setStartDate(pendingStartDate);
    if (setEndDate) setEndDate(pendingEndDate);
  };

  const resetDateChanges = () => {
    setPendingDateRangeType(dateRangeType);
    setPendingStartDate(startDate);
    setPendingEndDate(endDate);
  };

  const isStartDateDisabled = pendingDateRangeType === DATE_RANGE_TYPES.EARLIEST_TO_DATE;
  const isEndDateDisabled = pendingDateRangeType === DATE_RANGE_TYPES.DATE_TO_LATEST;

  return (
    <div className={`rounded-lg shadow-md p-6 mb-6 transition-colors duration-200 ${
      isDarkMode ? 'bg-gray-800' : 'bg-white'
    }`}>
      <div className="flex items-center justify-between mb-4">
        <h3 className={`text-lg font-semibold ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>
          Data Selection & Controls
        </h3>
        <button
          onClick={handleRefresh}
          className={`px-3 py-2 text-sm rounded-md transition-colors flex items-center gap-1 ${
            isDarkMode ? 'bg-blue-800 hover:bg-blue-700 text-blue-300' : 'bg-blue-100 hover:bg-blue-200 text-blue-700'
          }`}
        >
          🔄 Refresh Chart
        </button>
      </div>

      {/* Top controls */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <div>
          <label className={`block text-sm font-medium mb-2 ${isDarkMode ? 'text-gray-300' : 'text-gray-700'}`}>Symbol</label>
          <select
            value={selectedSymbol}
            onChange={(e) => setSelectedSymbol(e.target.value)}
            className={`w-full rounded-md px-3 py-2 transition-colors ${isDarkMode ? 'border-gray-600 bg-gray-700 text-white' : 'border-gray-300 bg-white text-gray-900'}`}
          >
            <option value="es">ES (E-mini S&P 500)</option>
            <option value="eurusd">EURUSD</option>
            <option value="spy">SPY</option>
          </select>
        </div>

        <div>
          <label className={`block text-sm font-medium mb-2 ${isDarkMode ? 'text-gray-300' : 'text-gray-700'}`}>Timeframe</label>
          <select
            value={selectedTimeframe}
            onChange={(e) => setSelectedTimeframe(e.target.value)}
            className={`w-full rounded-md px-3 py-2 transition-colors ${isDarkMode ? 'border-gray-600 bg-gray-700 text-white' : 'border-gray-300 bg-white text-gray-900'}`}
          >
            <option value="1m">1 Minute</option>
            <option value="5m">5 Minutes</option>
            <option value="15m">15 Minutes</option>
            <option value="30m">30 Minutes</option>
            <option value="1h">1 Hour</option>
            <option value="4h">4 Hours</option>
            <option value="1d">1 Day</option>
          </select>
        </div>

        <div>
          <label className={`block text-sm font-medium mb-2 ${isDarkMode ? 'text-gray-300' : 'text-gray-700'}`}>Data Limit</label>
          <select
            value={rowLimit}
            onChange={(e) => setRowLimit(parseInt(e.target.value))}
            className={`w-full rounded-md px-3 py-2 transition-colors ${isDarkMode ? 'border-gray-600 bg-gray-700 text-white' : 'border-gray-300 bg-white text-gray-900'}`}
          >
            {[25, 50, 100, 250, 500, 1000, 2000].map((n) => (
              <option key={n} value={n}>{n} records</option>
            ))}
          </select>
        </div>

        <div>
          <label className={`block text-sm font-medium mb-2 ${isDarkMode ? 'text-gray-300' : 'text-gray-700'}`}>Sort Order</label>
          <select
            value={sortOrder}
            onChange={(e) => setSortOrder(e.target.value)}
            className={`w-full rounded-md px-3 py-2 transition-colors ${isDarkMode ? 'border-gray-600 bg-gray-700 text-white' : 'border-gray-300 bg-white text-gray-900'}`}
          >
            <option value="desc">⬇ Descending (Newest first)</option>
            <option value="asc">⬆ Ascending (Oldest first)</option>
          </select>
        </div>
      </div>

      {/* Date range section */}
      <div className="mt-6">
        <div className="flex items-center gap-4 mb-3">
          <span className={isDarkMode ? 'text-gray-300 text-sm' : 'text-gray-700 text-sm'}>Fetch Mode:</span>
          <div className={`flex rounded-lg p-1 ${isDarkMode ? 'bg-gray-700' : 'bg-gray-100'}`}>
            <button
              onClick={() => setFetchMode(FETCH_MODES.LIMIT)}
              className={`px-3 py-1 text-sm rounded-md ${
                fetchMode === FETCH_MODES.LIMIT
                  ? isDarkMode ? 'bg-blue-600 text-white' : 'bg-blue-600 text-white'
                  : isDarkMode ? 'text-gray-300' : 'text-gray-600'
              }`}
            >
              📊 Record Limit
            </button>
            <button
              onClick={() => setFetchMode(FETCH_MODES.DATE_RANGE)}
              className={`ml-1 px-3 py-1 text-sm rounded-md ${
                fetchMode === FETCH_MODES.DATE_RANGE
                  ? isDarkMode ? 'bg-blue-600 text-white' : 'bg-blue-600 text-white'
                  : isDarkMode ? 'text-gray-300' : 'text-gray-600'
              }`}
            >
              📅 Date Range
            </button>
          </div>
        </div>

        {fetchMode === FETCH_MODES.DATE_RANGE && (
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-4">
            <div className="lg:col-span-4">
              <label className={`block text-sm font-medium mb-2 ${isDarkMode ? 'text-gray-300' : 'text-gray-700'}`}>Range Type</label>
              <select
                value={pendingDateRangeType}
                onChange={(e) => setPendingDateRangeType(e.target.value)}
                className={`w-full rounded-md px-3 py-2 transition-colors ${isDarkMode ? 'border-gray-600 bg-gray-700 text-white' : 'border-gray-300 bg-white text-gray-900'}`}
              >
                <option value={DATE_RANGE_TYPES.EARLIEST_TO_DATE}>Earliest → Date</option>
                <option value={DATE_RANGE_TYPES.DATE_TO_DATE}>Date → Date</option>
                <option value={DATE_RANGE_TYPES.DATE_TO_LATEST}>Date → Latest</option>
              </select>
            </div>

            {/* Start */}
            <div className="lg:col-span-4">
              <label className={`block text-sm font-medium mb-1 ${isDarkMode ? 'text-gray-300' : 'text-gray-700'}`}>Start</label>
              <div className="grid grid-cols-2 gap-2">
                <input
                  type="date"
                  value={formatDatePart(pendingStartDate)}
                  onChange={(e) => setPendingStartDate(combineDateAndTime(e.target.value, formatTimePart(pendingStartDate)))}
                  disabled={isStartDateDisabled}
                  className={`rounded-md px-3 py-2 ${isDarkMode ? 'border-gray-600 bg-gray-700 text-white' : 'border-gray-300 bg-white text-gray-900'}`}
                />
                <input
                  type="time"
                  value={formatTimePart(pendingStartDate)}
                  onChange={(e) => setPendingStartDate(combineDateAndTime(formatDatePart(pendingStartDate), e.target.value))}
                  disabled={isStartDateDisabled}
                  className={`rounded-md px-3 py-2 ${isDarkMode ? 'border-gray-600 bg-gray-700 text-white' : 'border-gray-300 bg-white text-gray-900'}`}
                />
              </div>
            </div>

            {/* End */}
            <div className="lg:col-span-4">
              <label className={`block text-sm font-medium mb-1 ${isDarkMode ? 'text-gray-300' : 'text-gray-700'}`}>End</label>
              <div className="grid grid-cols-2 gap-2">
                <input
                  type="date"
                  value={formatDatePart(pendingEndDate)}
                  onChange={(e) => setPendingEndDate(combineDateAndTime(e.target.value, formatTimePart(pendingEndDate)))}
                  disabled={isEndDateDisabled}
                  className={`rounded-md px-3 py-2 ${isDarkMode ? 'border-gray-600 bg-gray-700 text-white' : 'border-gray-300 bg-white text-gray-900'}`}
                />
                <input
                  type="time"
                  value={formatTimePart(pendingEndDate)}
                  onChange={(e) => setPendingEndDate(combineDateAndTime(formatDatePart(pendingEndDate), e.target.value))}
                  disabled={isEndDateDisabled}
                  className={`rounded-md px-3 py-2 ${isDarkMode ? 'border-gray-600 bg-gray-700 text-white' : 'border-gray-300 bg-white text-gray-900'}`}
                />
              </div>
            </div>
          </div>
        )}

        {/* Apply/reset */}
        {fetchMode === FETCH_MODES.DATE_RANGE && (
          <div className="mt-3 flex gap-2">
            <button
              onClick={applyDateChanges}
              disabled={!hasPendingChanges()}
              className={`px-3 py-2 rounded-md text-sm ${
                hasPendingChanges()
                  ? isDarkMode ? 'bg-green-700 hover:bg-green-600 text-white' : 'bg-green-600 hover:bg-green-700 text-white'
                  : isDarkMode ? 'bg-gray-700 text-gray-400' : 'bg-gray-200 text-gray-500'
              }`}
            >
              Apply Range
            </button>
            <button
              onClick={resetDateChanges}
              className={`px-3 py-2 rounded-md text-sm ${isDarkMode ? 'bg-gray-700 hover:bg-gray-600 text-gray-200' : 'bg-gray-100 hover:bg-gray-200 text-gray-800'}`}
            >
              Reset
            </button>
          </div>
        )}
      </div>
    </div>
  );
};

// Main Backtest Dashboard component
const BacktestDashboard = ({ isDarkMode = false }) => {
  const {
    selectedSymbol,
    selectedTimeframe,
    rowLimit,
    sortOrder,
    fetchMode,
    dateRangeType,
    startDate,
    endDate
  } = useTrading();

  const chartContainerRef = useRef(null);
  const chartRef = useRef(null);
  const priceSeriesRef = useRef(null);

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [chartData, setChartData] = useState(null); // full payload from API
  const fullData = chartData?.data || [];

  // Backtest state
  const [currentIndex, setCurrentIndex] = useState(0); // inclusive index of latest happened candle
  const [isPlaying, setIsPlaying] = useState(false);
  const [speedMs, setSpeedMs] = useState(1000);
  const [stepSize, setStepSize] = useState(1);
  const [loopPlayback, setLoopPlayback] = useState(false);

  // Build display data whenever currentIndex or fullData changes
  const visibleCandles = useMemo(() => {
    if (!fullData || fullData.length === 0) return [];
    const slice = fullData.slice(0, Math.min(currentIndex + 1, fullData.length));
    return slice.map(c => ({
      time: Math.floor(new Date(c.timestamp).getTime() / 1000),
      open: parseFloat(c.open),
      high: parseFloat(c.high),
      low: parseFloat(c.low),
      close: parseFloat(c.close)
    }));
  }, [fullData, currentIndex]);

  const initializeChart = useCallback(() => {
    if (!chartContainerRef.current || visibleCandles.length === 0) return;

    // Cleanup previous chart
    if (chartRef.current) {
      try { chartRef.current.remove(); } catch {}
      chartRef.current = null;
      priceSeriesRef.current = null;
    }

    const width = chartContainerRef.current.clientWidth || 800;
    const height = 420;

    const chart = createChart(chartContainerRef.current, {
      width,
      height,
      layout: {
        background: { type: ColorType.Solid, color: isDarkMode ? '#1f2937' : '#ffffff' },
        textColor: isDarkMode ? '#e5e7eb' : '#374151'
      },
      grid: {
        vertLines: { color: isDarkMode ? '#374151' : '#f3f4f6' },
        horzLines: { color: isDarkMode ? '#374151' : '#f3f4f6' }
      },
      rightPriceScale: {
        borderColor: isDarkMode ? '#4b5563' : '#d1d5db'
      },
      timeScale: {
        borderColor: isDarkMode ? '#4b5563' : '#d1d5db',
        timeVisible: true,
        secondsVisible: false
      }
    });

    const series = chart.addSeries(CandlestickSeries, {
      upColor: '#22c55e',
      downColor: '#ef4444',
      borderVisible: false,
      wickUpColor: '#22c55e',
      wickDownColor: '#ef4444'
    });
    series.setData(visibleCandles);

    chartRef.current = chart;
    priceSeriesRef.current = series;

    chart.timeScale().fitContent();

    const onResize = () => {
      const newWidth = chartContainerRef.current?.clientWidth || 800;
      chart.applyOptions({ width: newWidth });
    };
    window.addEventListener('resize', onResize);
    chart._onResize = onResize;
  }, [visibleCandles, isDarkMode]);

  // Update chart data when visibleCandles changes
  useEffect(() => {
    if (!priceSeriesRef.current) {
      initializeChart();
      return;
    }
    priceSeriesRef.current.setData(visibleCandles);
  }, [visibleCandles, initializeChart]);

  // Autoplay timer
  useEffect(() => {
    if (!isPlaying || fullData.length === 0) return;
    const atEnd = currentIndex >= fullData.length - 1;
    if (atEnd) {
      if (loopPlayback) {
        setCurrentIndex(0);
      } else {
        setIsPlaying(false);
      }
      return;
    }
    const id = setTimeout(() => {
      setCurrentIndex((i) => Math.min(i + 1, fullData.length - 1));
    }, speedMs);
    return () => clearTimeout(id);
  }, [isPlaying, speedMs, currentIndex, fullData.length, loopPlayback]);

  // Fetch chart data (mirrors Chart.jsx logic)
  const fetchChartData = useCallback(async () => {
    setLoading(true);
    setError(null);
    setChartData(null);
    try {
      let url = `${API_BASE_URL}/data/${selectedSymbol}/${selectedTimeframe}`;
      const queryParams = new URLSearchParams();

      if (fetchMode === FETCH_MODES.DATE_RANGE) {
        switch (dateRangeType) {
          case DATE_RANGE_TYPES.EARLIEST_TO_DATE:
            if (endDate) queryParams.append('end_date', endDate);
            break;
          case DATE_RANGE_TYPES.DATE_TO_DATE:
            if (startDate) queryParams.append('start_date', startDate);
            if (endDate) queryParams.append('end_date', endDate);
            break;
          case DATE_RANGE_TYPES.DATE_TO_LATEST:
            if (startDate) queryParams.append('start_date', startDate);
            break;
        }
        queryParams.append('limit', '10000');
      } else {
        queryParams.append('limit', String(rowLimit));
      }

      queryParams.append('order', sortOrder || 'desc');
      queryParams.append('sort_by', 'timestamp');

      const response = await fetch(`${url}?${queryParams.toString()}`);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const data = await response.json();

      if (data.data && Array.isArray(data.data)) {
        const sorted = [...data.data].sort((a, b) => new Date(a.timestamp) - new Date(b.timestamp));
        data.data = sorted;
      }

      setChartData(data);
      // Start with first 50 candles or available
      const startIdx = Math.min((data.data?.length || 1) - 1, 50);
      setCurrentIndex(Math.max(startIdx, 0));
    } catch (err) {
      setError(`Failed to fetch chart data: ${err.message}`);
    } finally {
      setLoading(false);
    }
  }, [selectedSymbol, selectedTimeframe, rowLimit, sortOrder, fetchMode, dateRangeType, startDate, endDate]);

  // Fetch on inputs change
  useEffect(() => {
    if (selectedSymbol && selectedTimeframe) fetchChartData();
    // Cleanup chart resize listener on unmount
    return () => {
      if (chartRef.current?._onResize) window.removeEventListener('resize', chartRef.current._onResize);
      if (chartRef.current) try { chartRef.current.remove(); } catch {}
    };
  }, [selectedSymbol, selectedTimeframe, rowLimit, sortOrder, fetchMode, dateRangeType, startDate, endDate, fetchChartData]);

  // Header info
  const headerBadge = (
    <div className={`mt-4 text-sm rounded p-2 ${isDarkMode ? 'bg-black/30' : 'bg-white/20'}`}>
      🧪 Backtest • {selectedSymbol?.toUpperCase()} • {selectedTimeframe} • {fullData.length} candles loaded
    </div>
  );

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className={`p-6 rounded-lg transition-colors duration-200 ${
        isDarkMode ? 'bg-gradient-to-r from-purple-800 via-indigo-700 to-blue-800 text-white' : 'bg-gradient-to-r from-purple-600 to-blue-600 text-white'
      }`}>
        <div className="flex items-center">
          <h2 className="text-3xl font-black tracking-tight mb-2" style={{
            fontFamily: 'system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
            letterSpacing: '-0.02em',
            textShadow: '0 2px 4px rgba(0,0,0,0.3)'
          }}>
            Day<span className="text-green-200">gent</span> <span className="text-xl font-semibold text-green-100">Backtest Dashboard</span>
          </h2>
          <InfoTooltip
            id="backtest-dashboard"
            isDarkMode={isDarkMode}
            content={(
              <div>
                <p className="font-semibold mb-2">🧪 Backtest Dashboard Overview</p>
                <p className="mb-2">Replay historical candles with controllable speed and stepping.</p>
                <ul className="list-disc list-inside text-xs space-y-1">
                  <li>Use the Data Selection box to load symbol/timeframe data.</li>
                  <li>Adjust playback speed and step size below.</li>
                  <li>Move the slider to set what has "happened" so far.</li>
                </ul>
              </div>
            )}
          />
        </div>
        {headerBadge}
      </div>

      {/* Data selection & controls */}
      <DataSelectionControls handleRefresh={fetchChartData} isDarkMode={isDarkMode} />

      {/* Backtest settings */}
      <BacktestSettings
        isDarkMode={isDarkMode}
        fullData={fullData}
        currentIndex={currentIndex}
        setCurrentIndex={setCurrentIndex}
        isPlaying={isPlaying}
        setIsPlaying={setIsPlaying}
        speedMs={speedMs}
        setSpeedMs={setSpeedMs}
        stepSize={stepSize}
        setStepSize={setStepSize}
        loopPlayback={loopPlayback}
        setLoopPlayback={setLoopPlayback}
      />

      {/* Chart */}
      <div className={`rounded-lg shadow-md transition-colors duration-200 ${isDarkMode ? 'bg-gray-800' : 'bg-white'}`}>
        <div className={`px-6 py-4 border-b flex items-center justify-between transition-colors duration-200 ${
          isDarkMode ? 'border-gray-700' : 'border-gray-200'
        }`}>
          <div className="flex items-center">
            <h3 className={`text-lg font-semibold ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>Price Chart</h3>
          </div>
          <div className={`text-xs ${isDarkMode ? 'text-gray-400' : 'text-gray-600'}`}>
            {currentIndex + 1} / {fullData.length} candles shown
          </div>
        </div>

        <div className="p-6">
          {loading && (
            <div className="flex items-center justify-center h-96">
              <div className={`animate-spin rounded-full h-12 w-12 border-b-2 ${isDarkMode ? 'border-blue-400' : 'border-blue-600'}`}></div>
            </div>
          )}
          {error && (
            <div className={`rounded-lg p-4 border ${isDarkMode ? 'bg-red-900/20 border-red-800 text-red-300' : 'bg-red-50 border-red-200 text-red-700'}`}>
              {error}
            </div>
          )}
          {!loading && !error && (
            <div
              ref={chartContainerRef}
              className={`w-full h-96 rounded border transition-colors duration-200 ${
                isDarkMode ? 'border-gray-600 bg-gray-900' : 'border-gray-200 bg-gray-50'
              }`}
              style={{ minHeight: '400px' }}
            />
          )}
        </div>
      </div>
    </div>
  );
};

export default BacktestDashboard;


