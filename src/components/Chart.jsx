import { useState, useEffect, useRef, useCallback } from 'react';
// Import lightweight-charts v5.0 - install with: npm install lightweight-charts
import { createChart, CandlestickSeries, LineSeries, AreaSeries, ColorType, createSeriesMarkers, CrosshairMode } from 'lightweight-charts';
import { useTrading } from '../context/TradingContext';
import { useDateRanges } from '../hooks/useDateRanges';
import { FETCH_MODES, DATE_RANGE_TYPES } from '../utils/constants';
import SelectedCandlesPanel from './shared/SelectedCandlesPanel.jsx';
import GameModeController from './Game.jsx';

// FVG markers are now handled via series markers for proper layering and positioning

// Reusable InfoTooltip component (shared across dashboards)
const InfoTooltip = ({ id, content, isDarkMode, asSpan = false }) => {
  const [activeTooltip, setActiveTooltip] = useState(null);
  const isActive = activeTooltip === id;
  
  const triggerClasses = `ml-2 w-4 h-4 rounded-full border text-xs font-bold transition-all duration-200 cursor-pointer ${
    isDarkMode 
      ? 'border-gray-500 text-gray-400 hover:border-blue-400 hover:text-blue-400' 
      : 'border-gray-400 text-gray-500 hover:border-blue-500 hover:text-blue-600'
  }`;

  const triggerProps = {
    onClick: () => setActiveTooltip(isActive ? null : id),
    onMouseEnter: () => setActiveTooltip(id),
    onMouseLeave: () => setActiveTooltip(null),
    className: triggerClasses
  };
  
  return (
    <div className="relative">
      {asSpan ? (
        <span {...triggerProps}>i</span>
      ) : (
        <button {...triggerProps}>i</button>
      )}
      {isActive && (
        <div className={`absolute z-50 bottom-full left-1/2 transform -translate-x-1/2 mb-2 w-80 p-4 rounded-lg shadow-lg border transition-colors duration-200 ${
          isDarkMode 
            ? 'bg-gray-800 border-gray-600 text-gray-200' 
            : 'bg-white border-gray-200 text-gray-800'
        }`}>
          <div className="text-sm leading-relaxed">{content}</div>
          {/* Arrow pointing down */}
          <div className={`absolute top-full left-1/2 transform -translate-x-1/2 w-0 h-0 border-l-[8px] border-r-[8px] border-t-[8px] border-l-transparent border-r-transparent ${
            isDarkMode ? 'border-t-gray-800' : 'border-t-white'
          }`}></div>
        </div>
      )}
    </div>
  );
};

// Reusable Data Selection & Controls component (shared across all dashboards)
const DataSelectionControls = ({ 
  handleRefresh, 
  isDarkMode,
  dashboardType = 'chart'
}) => {
  // Use shared trading context
  const {
    selectedSymbol,
    setSelectedSymbol,
    selectedTimeframe,
    setSelectedTimeframe,
    rowLimit,
    setRowLimit,
    sortOrder,
    setSortOrder,
    // Date range functionality
    fetchMode,
    setFetchMode,
    dateRangeType,
    setDateRangeType,
    startDate,
    setStartDate,
    endDate,
    setEndDate
  } = useTrading();

  // Fetch available date ranges for current symbol/timeframe
  const { dateRanges, loading: dateRangesLoading } = useDateRanges(selectedSymbol, selectedTimeframe);

  // Local state for pending date changes (before applying)
  const [pendingStartDate, setPendingStartDate] = useState(startDate);
  const [pendingEndDate, setPendingEndDate] = useState(endDate);
  const [pendingDateRangeType, setPendingDateRangeType] = useState(dateRangeType);

  // Update pending states when actual values change (from other tabs or initial load)
  useEffect(() => {
    setPendingStartDate(startDate);
    setPendingEndDate(endDate);
    setPendingDateRangeType(dateRangeType);
  }, [startDate, endDate, dateRangeType]);

  // Auto-fill dates when date range type changes
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
        // Keep both dates as user set them, or use defaults if empty
        if (!pendingStartDate) setPendingStartDate(dateRanges.earliest);
        if (!pendingEndDate) setPendingEndDate(dateRanges.latest);
        break;
    }
  }, [pendingDateRangeType, dateRanges]);

  // Helper function to format date for input
  const formatDateForInput = (dateString) => {
    if (!dateString) return '';
    const date = new Date(dateString);
    return date.toISOString().slice(0, 16); // Format: YYYY-MM-DDTHH:MM
  };

  // Helper function to parse date from input
  const parseDateFromInput = (inputValue) => {
    if (!inputValue) return null;
    return new Date(inputValue).toISOString();
  };

  // Helper functions for the new separate date/time inputs
  const formatDatePart = (dateString) => {
    if (!dateString) return '';
    const date = new Date(dateString);
    return date.toISOString().slice(0, 10); // YYYY-MM-DD
  };

  const formatTimePart = (dateString) => {
    if (!dateString) return '';
    const date = new Date(dateString);
    return date.toISOString().slice(11, 16); // HH:MM
  };

  const combineDateAndTime = (dateValue, timeValue) => {
    if (!dateValue) return null;
    const timeToUse = timeValue || '00:00';
    return new Date(`${dateValue}T${timeToUse}:00.000Z`).toISOString();
  };

  // Check if there are pending changes
  const hasPendingChanges = () => {
    return (
      pendingDateRangeType !== dateRangeType ||
      pendingStartDate !== startDate ||
      pendingEndDate !== endDate
    );
  };

  // Apply pending changes
  const applyDateChanges = () => {
    if (setDateRangeType) setDateRangeType(pendingDateRangeType);
    if (setStartDate) setStartDate(pendingStartDate);
    if (setEndDate) setEndDate(pendingEndDate);
  };

  // Reset pending changes to current values
  const resetDateChanges = () => {
    setPendingDateRangeType(dateRangeType);
    setPendingStartDate(startDate);
    setPendingEndDate(endDate);
  };

  // Handle date range type change
  const handleDateRangeTypeChange = (newType) => {
    setPendingDateRangeType(newType);
  };

  // Determine if inputs should be disabled based on date range type
  const isStartDateDisabled = pendingDateRangeType === DATE_RANGE_TYPES.EARLIEST_TO_DATE;
  const isEndDateDisabled = pendingDateRangeType === DATE_RANGE_TYPES.DATE_TO_LATEST;

  return (
    <div className={`rounded-lg shadow-md p-6 mb-8 transition-colors duration-200 ${
      isDarkMode ? 'bg-gray-800' : 'bg-white'
    }`}>
      <div className="flex items-center justify-between mb-4">
        <h3 className={`text-lg font-semibold ${
          isDarkMode ? 'text-white' : 'text-gray-900'
        }`}>Data Selection & Controls</h3>
        <div className="flex gap-2">
          <button
            onClick={handleRefresh}
            className={`px-3 py-2 text-sm rounded-md transition-colors flex items-center gap-1 ${
              isDarkMode 
                ? 'bg-blue-800 hover:bg-blue-700 text-blue-300' 
                : 'bg-blue-100 hover:bg-blue-200 text-blue-700'
            }`}
          >
            🔄 Refresh {dashboardType === 'chart' ? 'Chart' : dashboardType === 'vector' ? 'Vectors' : 'Data'}
          </button>
        </div>
      </div>

      {/* Fetch Mode Toggle */}
      <div className="mb-4">
        <div className="flex items-center space-x-4">
          <span className={`text-sm font-medium ${
            isDarkMode ? 'text-gray-300' : 'text-gray-700'
          }`}>
            Fetch Mode:
          </span>
          <div className={`flex rounded-lg p-1 transition-colors duration-200 ${
            isDarkMode ? 'bg-gray-700' : 'bg-gray-100'
          }`}>
            <button
              onClick={() => setFetchMode(FETCH_MODES.LIMIT)}
              className={`px-3 py-1 rounded-md text-sm font-medium transition-all duration-200 ${
                fetchMode === FETCH_MODES.LIMIT
                  ? isDarkMode 
                    ? 'bg-gray-600 text-white shadow-sm' 
                    : 'bg-white text-gray-900 shadow-sm'
                  : isDarkMode
                    ? 'text-gray-300 hover:text-white'
                    : 'text-gray-600 hover:text-gray-800'
              }`}
            >
              📊 Record Limit
            </button>
            <button
              onClick={() => setFetchMode(FETCH_MODES.DATE_RANGE)}
              className={`px-3 py-1 rounded-md text-sm font-medium transition-all duration-200 ${
                fetchMode === FETCH_MODES.DATE_RANGE
                  ? isDarkMode 
                    ? 'bg-gray-600 text-white shadow-sm' 
                    : 'bg-white text-gray-900 shadow-sm'
                  : isDarkMode
                    ? 'text-gray-300 hover:text-white'
                    : 'text-gray-600 hover:text-gray-800'
              }`}
            >
              📅 Date Range
            </button>
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
          <label className={`block text-sm font-medium mb-2 ${
            isDarkMode ? 'text-gray-300' : 'text-gray-700'
          }`}>
            Symbol
          </label>
          <select
            value={selectedSymbol}
            onChange={(e) => setSelectedSymbol(e.target.value)}
            className={`w-full rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500 transition-colors duration-200 ${
              isDarkMode 
                ? 'border-gray-600 bg-gray-700 text-white' 
                : 'border-gray-300 bg-white text-gray-900'
            }`}
          >
            <option value="es">ES (E-mini S&P 500)</option>
            <option value="eurusd">EURUSD (Euro/US Dollar)</option>
            <option value="spy">SPY (SPDR S&P 500 ETF)</option>
          </select>
        </div>
        
        <div>
          <label className={`block text-sm font-medium mb-2 ${
            isDarkMode ? 'text-gray-300' : 'text-gray-700'
          }`}>
            Timeframe
          </label>
          <select
            value={selectedTimeframe}
            onChange={(e) => setSelectedTimeframe(e.target.value)}
            className={`w-full rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500 transition-colors duration-200 ${
              isDarkMode 
                ? 'border-gray-600 bg-gray-700 text-white' 
                : 'border-gray-300 bg-white text-gray-900'
            }`}
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

        {fetchMode === FETCH_MODES.LIMIT && (
          <div>
            <label className={`block text-sm font-medium mb-2 ${
              isDarkMode ? 'text-gray-300' : 'text-gray-700'
            }`}>
              Data Limit
            </label>
            <select
              value={rowLimit}
              onChange={(e) => setRowLimit(Number(e.target.value))}
              className={`w-full rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500 transition-colors duration-200 ${
                isDarkMode 
                  ? 'border-gray-600 bg-gray-700 text-white' 
                  : 'border-gray-300 bg-white text-gray-900'
              }`}
            >
              <option value={25}>25 records</option>
              <option value={50}>50 records</option>
              <option value={100}>100 records</option>
              <option value={250}>250 records</option>
              <option value={500}>500 records</option>
              <option value={1000}>1000 records</option>
              <option value={2000}>2000 records</option>
            </select>
          </div>
        )}

        {fetchMode === FETCH_MODES.DATE_RANGE && (
          <div>
            <label className={`block text-sm font-medium mb-2 ${
              isDarkMode ? 'text-gray-300' : 'text-gray-700'
            }`}>
              Date Range Type
            </label>
            <select
              value={pendingDateRangeType || DATE_RANGE_TYPES.EARLIEST_TO_DATE}
              onChange={(e) => handleDateRangeTypeChange(e.target.value)}
              className={`w-full rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500 transition-colors duration-200 ${
                isDarkMode 
                  ? 'border-gray-600 bg-gray-700 text-white' 
                  : 'border-gray-300 bg-white text-gray-900'
              }`}
            >
              <option value={DATE_RANGE_TYPES.EARLIEST_TO_DATE}>📅 Earliest to Date</option>
              <option value={DATE_RANGE_TYPES.DATE_TO_DATE}>📅 Date to Date</option>
              <option value={DATE_RANGE_TYPES.DATE_TO_LATEST}>📅 Date to Latest</option>
            </select>
          </div>
        )}

        <div>
          <div className="flex items-center mb-2">
            <label className={`block text-sm font-medium ${
              isDarkMode ? 'text-gray-300' : 'text-gray-700'
            }`}>
              Sort Order
            </label>
            <InfoTooltip id="sort-order" content={
              <div>
                <p className="font-semibold mb-2">⬇️ Sort Order Options</p>
                <ul className="list-disc list-inside space-y-1 text-xs">
                  <li><strong>Descending (Newest first):</strong> Shows most recent trading data</li>
                  <li><strong>Ascending (Oldest first):</strong> Shows historical data from the beginning</li>
                </ul>
                <p className="mt-2 text-xs"><strong>Note:</strong> Sort order is synchronized across all dashboards. This affects both data tables and charts.</p>
              </div>
            } isDarkMode={isDarkMode} />
          </div>
          <select
            value={sortOrder || 'desc'}
            onChange={(e) => setSortOrder && setSortOrder(e.target.value)}
            className={`w-full rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500 transition-colors duration-200 ${
              isDarkMode 
                ? 'border-gray-600 bg-gray-700 text-white' 
                : 'border-gray-300 bg-white text-gray-900'
            }`}
          >
            <option value="desc">⬇ Descending (Newest first)</option>
            <option value="asc">⬆ Ascending (Oldest first)</option>
          </select>
        </div>
      </div>

      {/* Date Range Inputs */}
      {fetchMode === FETCH_MODES.DATE_RANGE && (
        <div className={`mt-4 p-4 rounded-lg border transition-colors duration-200 ${
          isDarkMode ? 'bg-gray-700 border-gray-600' : 'bg-gray-50 border-gray-200'
        }`}>
          <h4 className={`text-md font-medium mb-3 ${
            isDarkMode ? 'text-white' : 'text-gray-800'
          }`}>Date Range Configuration</h4>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {/* Start Date - Always visible */}
            <div>
              <label className={`block text-sm font-medium mb-2 ${
                isDarkMode ? 'text-gray-300' : 'text-gray-700'
              }`}>
                Start Date {isStartDateDisabled && (
                  <span className={`text-xs ${isDarkMode ? 'text-gray-500' : 'text-gray-400'}`}>
                    (Auto-filled: Earliest)
                  </span>
                )}
              </label>
              <div className="grid grid-cols-2 gap-2">
                <div>
                  <label className={`block text-xs mb-1 ${isDarkMode ? 'text-gray-400' : 'text-gray-600'}`}>
                    Date
                  </label>
                  <input
                    type="date"
                    value={formatDatePart(pendingStartDate)}
                    onChange={(e) => {
                      const newDate = combineDateAndTime(e.target.value, formatTimePart(pendingStartDate));
                      setPendingStartDate(newDate);
                    }}
                    disabled={isStartDateDisabled}
                    min={dateRanges ? formatDatePart(dateRanges.earliest) : ''}
                    max={dateRanges ? formatDatePart(dateRanges.latest) : ''}
                    className={`w-full rounded-md px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 transition-colors duration-200 ${
                      isStartDateDisabled 
                        ? isDarkMode 
                          ? 'border-gray-600 bg-gray-600 text-gray-400 cursor-not-allowed' 
                          : 'border-gray-200 bg-gray-100 text-gray-500 cursor-not-allowed'
                        : isDarkMode 
                          ? 'border-gray-600 bg-gray-800 text-white' 
                          : 'border-gray-300 bg-white text-gray-900'
                    }`}
                  />
                </div>
                <div>
                  <label className={`block text-xs mb-1 ${isDarkMode ? 'text-gray-400' : 'text-gray-600'}`}>
                    Time
                  </label>
                  <input
                    type="time"
                    value={formatTimePart(pendingStartDate)}
                    onChange={(e) => {
                      const newDate = combineDateAndTime(formatDatePart(pendingStartDate), e.target.value);
                      setPendingStartDate(newDate);
                    }}
                    disabled={isStartDateDisabled}
                    className={`w-full rounded-md px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 transition-colors duration-200 ${
                      isStartDateDisabled 
                        ? isDarkMode 
                          ? 'border-gray-600 bg-gray-600 text-gray-400 cursor-not-allowed' 
                          : 'border-gray-200 bg-gray-100 text-gray-500 cursor-not-allowed'
                        : isDarkMode 
                          ? 'border-gray-600 bg-gray-800 text-white' 
                          : 'border-gray-300 bg-white text-gray-900'
                    }`}
                  />
                </div>
              </div>
              {dateRanges && (
                <div className={`text-xs mt-1 ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>
                  Available from: {new Date(dateRanges.earliest).toLocaleDateString()}
                </div>
              )}
            </div>

            {/* End Date - Always visible */}
            <div>
              <label className={`block text-sm font-medium mb-2 ${
                isDarkMode ? 'text-gray-300' : 'text-gray-700'
              }`}>
                End Date {isEndDateDisabled && (
                  <span className={`text-xs ${isDarkMode ? 'text-gray-500' : 'text-gray-400'}`}>
                    (Auto-filled: Latest)
                  </span>
                )}
              </label>
              <div className="grid grid-cols-2 gap-2">
                <div>
                  <label className={`block text-xs mb-1 ${isDarkMode ? 'text-gray-400' : 'text-gray-600'}`}>
                    Date
                  </label>
                  <input
                    type="date"
                    value={formatDatePart(pendingEndDate)}
                    onChange={(e) => {
                      const newDate = combineDateAndTime(e.target.value, formatTimePart(pendingEndDate));
                      setPendingEndDate(newDate);
                    }}
                    disabled={isEndDateDisabled}
                    min={dateRanges ? formatDatePart(dateRanges.earliest) : ''}
                    max={dateRanges ? formatDatePart(dateRanges.latest) : ''}
                    className={`w-full rounded-md px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 transition-colors duration-200 ${
                      isEndDateDisabled 
                        ? isDarkMode 
                          ? 'border-gray-600 bg-gray-600 text-gray-400 cursor-not-allowed' 
                          : 'border-gray-200 bg-gray-100 text-gray-500 cursor-not-allowed'
                        : isDarkMode 
                          ? 'border-gray-600 bg-gray-800 text-white' 
                          : 'border-gray-300 bg-white text-gray-900'
                    }`}
                  />
                </div>
                <div>
                  <label className={`block text-xs mb-1 ${isDarkMode ? 'text-gray-400' : 'text-gray-600'}`}>
                    Time
                  </label>
                  <input
                    type="time"
                    value={formatTimePart(pendingEndDate)}
                    onChange={(e) => {
                      const newDate = combineDateAndTime(formatDatePart(pendingEndDate), e.target.value);
                      setPendingEndDate(newDate);
                    }}
                    disabled={isEndDateDisabled}
                    className={`w-full rounded-md px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 transition-colors duration-200 ${
                      isEndDateDisabled 
                        ? isDarkMode 
                          ? 'border-gray-600 bg-gray-600 text-gray-400 cursor-not-allowed' 
                          : 'border-gray-200 bg-gray-100 text-gray-500 cursor-not-allowed'
                        : isDarkMode 
                          ? 'border-gray-600 bg-gray-800 text-white' 
                          : 'border-gray-300 bg-white text-gray-900'
                    }`}
                  />
                </div>
              </div>
              {dateRanges && (
                <div className={`text-xs mt-1 ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>
                  Available until: {new Date(dateRanges.latest).toLocaleDateString()}
                </div>
              )}
            </div>
          </div>

          {/* Apply/Reset buttons */}
          <div className="flex justify-end gap-2 mt-4">
            {hasPendingChanges() && (
              <button
                onClick={resetDateChanges}
                className={`px-3 py-2 text-sm rounded-md transition-colors duration-200 ${
                  isDarkMode 
                    ? 'bg-gray-600 text-gray-200 hover:bg-gray-500' 
                    : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                }`}
              >
                Reset
              </button>
            )}
            <button
              onClick={applyDateChanges}
              disabled={!hasPendingChanges()}
              className={`px-4 py-2 text-sm rounded-md transition-colors duration-200 ${
                !hasPendingChanges()
                  ? isDarkMode 
                    ? 'bg-gray-600 text-gray-500 cursor-not-allowed' 
                    : 'bg-gray-300 text-gray-500 cursor-not-allowed'
                  : isDarkMode 
                    ? 'bg-blue-600 text-white hover:bg-blue-500' 
                    : 'bg-blue-500 text-white hover:bg-blue-600'
              }`}
            >
              {!hasPendingChanges() ? 'No Changes' : 'Apply Date Range'}
            </button>
          </div>

          {/* Current applied range info */}
          {(startDate || endDate) && (
            <div className={`mt-3 pt-3 border-t text-xs ${
              isDarkMode ? 'border-gray-600 text-gray-400' : 'border-gray-300 text-gray-500'
            }`}>
              <div className="font-medium mb-1">Currently Applied:</div>
              <div>
                {startDate && `From: ${new Date(startDate).toLocaleString()}`}
                {startDate && endDate && ' • '}
                {endDate && `To: ${new Date(endDate).toLocaleString()}`}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

// --- TJR LABELS CONTROL ---
const TjrLabelsControl = ({ isDarkMode, chartRef, priceSeriesRef, chartDataRef, gameMode, isSelectionMode }) => {
  const [showTjrHighs, setShowTjrHighs] = useState(false);
  const [showTjrLows, setShowTjrLows] = useState(false);
  // Swing toggles
  const [showSwingHighs, setShowSwingHighs] = useState(false);
  const [showSwingLows, setShowSwingLows] = useState(false);
  // Swing label data
  const [swingLabelsData, setSwingLabelsData] = useState(null);
  const [matchingSwingLabels, setMatchingSwingLabels] = useState([]);
  const [labelsData, setLabelsData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [matchingLabels, setMatchingLabels] = useState([]);
  const [hasLabelsTable, setHasLabelsTable] = useState(false);
  const [availableLabeledTables, setAvailableLabeledTables] = useState([]);
  const [availableSwingTables, setAvailableSwingTables] = useState([]);
  const [isLabelsLoaded, setIsLabelsLoaded] = useState(false);

  const {
    selectedSymbol,
    selectedTimeframe
  } = useTrading();

  // Check which labeled and swing tables are available
  useEffect(() => {
    const checkAvailableTables = async () => {
      const spyTimeframes = ['1m', '5m', '15m', '30m', '1h', '4h', '1d'];
      const availableLabeled = [];
      const availableSwing = [];
      
      for (const tf of spyTimeframes) {
        // Check labeled tables
        try {
          const response = await fetch(`http://localhost:8000/api/trading/labels/spy/${tf}`);
          if (response.ok) {
            const data = await response.json();
            if (Array.isArray(data) && data.length > 0) {
              availableLabeled.push(`spy${tf}_labeled`);
            }
          }
        } catch (err) {
          // Silently ignore errors for table checking
        }
        
        // Check swing tables
        try {
          const response = await fetch(`http://localhost:8000/api/trading/swing-labels/spy/${tf}`);
          if (response.ok) {
            const data = await response.json();
            if (Array.isArray(data) && data.length > 0) {
              availableSwing.push(`spy${tf}_swings`);
            }
          }
        } catch (err) {
          // Silently ignore errors for table checking
        }
      }
      
      setAvailableLabeledTables(availableLabeled);
      setAvailableSwingTables(availableSwing);
    };
    
    checkAvailableTables();
  }, []);

  // Fetch labels data when component mounts
  useEffect(() => {
    const fetchLabelsData = async () => {
      if (!selectedSymbol || !selectedTimeframe) return;
      
      setLoading(true);
      setError(null);
      
      try {
        // Build the API endpoint based on symbol and timeframe
        const endpoint = `http://localhost:8000/api/trading/labels/${selectedSymbol}/${selectedTimeframe}`;
        console.log('🏷️ Fetching labels from:', endpoint);
        
        const response = await fetch(endpoint);
        if (!response.ok) {
          // If the generic endpoint fails, try the legacy spy1h endpoint as fallback
          if (selectedSymbol === 'spy' && selectedTimeframe === '1h') {
            const fallbackEndpoint = 'http://localhost:8000/api/trading/labels/spy1h';
            console.log('🔄 Trying fallback endpoint:', fallbackEndpoint);
            const fallbackResponse = await fetch(fallbackEndpoint);
            if (fallbackResponse.ok) {
              const fallbackData = await fallbackResponse.json();
              if (Array.isArray(fallbackData)) {
                setLabelsData(fallbackData);
                setHasLabelsTable(fallbackData.length > 0);
                console.log('📊 Labels stats (fallback):', {
                  total: fallbackData.length,
                  tjr_high: fallbackData.filter(l => l.label === 'tjr_high').length,
                  tjr_low: fallbackData.filter(l => l.label === 'tjr_low').length
                });
                setIsLabelsLoaded(true);
                return;
              }
            }
          }
          // Don't throw error for 404/500, just set empty data
          console.warn(`Labels endpoint returned ${response.status} for ${selectedSymbol}${selectedTimeframe}`);
          setLabelsData([]);
          setHasLabelsTable(false);
          setIsLabelsLoaded(true);
          return;
        }
        
        const data = await response.json();
        console.log('🏷️ Labels data received:', data);
        
        if (Array.isArray(data)) {
          setLabelsData(data);
          setHasLabelsTable(true);
          console.log('📊 Labels stats:', {
            total: data.length,
            tjr_high: data.filter(l => l.label === 'tjr_high').length,
            tjr_low: data.filter(l => l.label === 'tjr_low').length
          });
          setIsLabelsLoaded(true);
        } else {
          setLabelsData([]);
          setHasLabelsTable(false);
          setIsLabelsLoaded(true);
        }
        
              } catch (err) {
          console.error('❌ Error fetching labels:', err);
          // Only set error for actual network/parsing errors, not missing tables
          if (err.message.includes('Failed to fetch') || err.message.includes('JSON')) {
            setError(err.message);
          } else {
            setError(null); // Clear any previous errors
          }
          setLabelsData([]);
          setHasLabelsTable(false);
          setIsLabelsLoaded(true);
        } finally {
          setLoading(false);
        }
    };

    fetchLabelsData();
  }, [selectedSymbol, selectedTimeframe]);

  // Scan current candles for matching labels
  useEffect(() => {
    if (!labelsData || labelsData.length === 0 || !chartDataRef?.current) {
      setMatchingLabels([]);
      return;
    }

    console.log('🔍 Scanning candles for TJR labels...');
    const candles = chartDataRef.current;
    const matches = [];

    candles.forEach(candle => {
      const candleTime = new Date(candle.timestamp);
      
      // Check each label to see if this candle falls within its pointer range
      labelsData.forEach(label => {
        if (!label.pointer || !Array.isArray(label.pointer) || label.pointer.length < 2) {
          return;
        }
        
        const startTime = new Date(label.pointer[0]);
        const endTime = new Date(label.pointer[1]);
        
        // Check if candle timestamp falls within the label's time range
        if (candleTime >= startTime && candleTime <= endTime) {
          matches.push({
            candle: candle,
            label: label,
            candleTime: candleTime,
            labelTimeRange: [startTime, endTime]
          });
        }
      });
    });

    console.log(`🎯 Found ${matches.length} matching TJR candles:`, matches);
    setMatchingLabels(matches);
  }, [labelsData, chartDataRef?.current]);

  // Fetch swing labels for all SPY timeframes
  useEffect(() => {
    if (selectedSymbol === 'spy') {
      const fetchSwingLabels = async () => {
        try {
          // Try the generic endpoint first
          const endpoint = `http://localhost:8000/api/trading/swing-labels/${selectedSymbol}/${selectedTimeframe}`;
          const response = await fetch(endpoint);
          if (response.ok) {
            const data = await response.json();
            setSwingLabelsData(Array.isArray(data) ? data : []);
            console.log(`📊 Swing labels loaded for ${selectedSymbol}${selectedTimeframe}:`, data.length);
          } else {
            // Fallback to spy1h endpoint if it's spy1h and generic fails
            if (selectedTimeframe === '1h') {
              const fallbackEndpoint = 'http://localhost:8000/api/trading/labels/spy1h_swings';
              const fallbackResponse = await fetch(fallbackEndpoint);
              if (fallbackResponse.ok) {
                const fallbackData = await fallbackResponse.json();
                setSwingLabelsData(Array.isArray(fallbackData) ? fallbackData : []);
                console.log(`📊 Swing labels loaded (fallback) for spy1h:`, fallbackData.length);
              } else {
                setSwingLabelsData([]);
              }
            } else {
              setSwingLabelsData([]);
            }
          }
        } catch (err) {
          console.warn(`No swing labels available for ${selectedSymbol}${selectedTimeframe}`);
          setSwingLabelsData([]);
        }
      };
      fetchSwingLabels();
    } else {
      setSwingLabelsData(null);
    }
  }, [selectedSymbol, selectedTimeframe]);

  // Scan chart candles for swing markers
  useEffect(() => {
    if (!swingLabelsData || !chartDataRef?.current) {
      setMatchingSwingLabels([]);
      return;
    }
    
    console.log('🔍 Scanning candles for swing markers...');
    const candles = chartDataRef.current;
    const matches = [];
    
    swingLabelsData.forEach(label => {
      if (!label.pointer || !Array.isArray(label.pointer) || label.pointer.length < 1) return;
      
      const labelMillis = new Date(label.pointer[0]).getTime();
      
      candles.forEach(candle => {
        const candleMillis = new Date(candle.timestamp).getTime();
        if (candleMillis === labelMillis) {
          matches.push({ candle, label, candleMillis, labelMillis });
          console.log(`🟦 Matched swing marker: label ${label.label} at ${label.pointer[0]} (${labelMillis}) to candle at ${candle.timestamp} (${candleMillis})`);
        }
      });
    });
    
    console.log('🟦 Swing marker matching summary:', { 
      totalSwingLabels: swingLabelsData.length, 
      totalCandles: candles.length,
      matches: matches.length,
      swingHighs: matches.filter(m => m.label.label === 'swing_high').length,
      swingLows: matches.filter(m => m.label.label === 'swing_low').length
    });
    setMatchingSwingLabels(matches);
  }, [swingLabelsData, chartDataRef?.current]);

  // Handle toggle changes and update chart markers
  useEffect(() => {
    if (!chartRef?.current || !priceSeriesRef?.current) {
      return;
    }

    console.log('🔄 Updating chart markers:', { 
      showTjrHighs, 
      showTjrLows, 
      showSwingHighs, 
      showSwingLows,
      tjrMatches: matchingLabels.length,
      swingMatches: matchingSwingLabels.length 
    });
    
    // Trigger the main chart highlighting update which now includes both TJR and swing markers
    const updateEvent = new CustomEvent('updateChartHighlights');
    document.dispatchEvent(updateEvent);
    
    // Fit content - but not in game mode or selection mode to avoid conflicts
    if (!gameMode && !isSelectionMode) {
      try { 
        chartRef.current.timeScale().fitContent(); 
      } catch (e) {
        console.warn('Could not fit content:', e);
      }
    }
  }, [showTjrHighs, showTjrLows, showSwingHighs, showSwingLows, matchingLabels, matchingSwingLabels, chartRef?.current, priceSeriesRef?.current, gameMode, isSelectionMode]);



  return (
    <div 
      className={`rounded-lg shadow-md transition-colors duration-200 ${
        isDarkMode ? 'bg-gray-800' : 'bg-white'
      }`}
      data-tjr-labels={JSON.stringify({ showTjrHighs, showTjrLows, matchingLabels, showSwingHighs, showSwingLows, matchingSwingLabels })}
    >
      <div className={`px-6 py-4 border-b transition-colors duration-200 ${
        isDarkMode ? 'border-gray-700' : 'border-gray-200'
      }`}>
        <div className="flex items-center justify-between">
          <div className="flex items-center">
            <h3 className={`text-lg font-semibold ${
              isDarkMode ? 'text-white' : 'text-gray-900'
            }`}>
              TJR & Swing Markers
              {hasLabelsTable && (
                <span className="ml-2 text-xs px-2 py-1 rounded-full bg-green-100 text-green-800">
                  ✅ Available
                </span>
              )}
              {!hasLabelsTable && !loading && !error && (
                <span className="ml-2 text-xs px-2 py-1 rounded-full bg-gray-100 text-gray-600">
                  ⚠️ No Data
                </span>
              )}
            </h3>
            <InfoTooltip id="tjr-labels" content={
              <div>
                <p className="font-semibold mb-2">🏷️ TJR & Swing Trading Markers</p>
                <p className="mb-2">Visual markers showing trading point indicators:</p>
                <ul className="list-disc list-inside space-y-1 text-xs">
                  <li><strong>TJR High:</strong> Green "T" symbol above candles for significant highs</li>
                  <li><strong>TJR Low:</strong> Red "⊥" symbol below candles for significant lows</li>
                  <li><strong>Swing High:</strong> Blue "▲" symbol above candles (all SPY timeframes)</li>
                  <li><strong>Swing Low:</strong> Orange "▼" symbol below candles (all SPY timeframes)</li>
                  <li><strong>Toggle Display:</strong> Show/hide each marker type independently</li>
                  <li><strong>Table Support:</strong> Works with any SPY table that has labeled data</li>
                </ul>
                <p className="mt-2 text-xs">Toggle these on to see symbol markers on your chart.</p>
              </div>
            } isDarkMode={isDarkMode} />
          </div>
          <div className="flex items-center gap-4">
            {/* TJR Highs Toggle */}
            <div className="flex items-center gap-2">
              <label className={`text-sm font-medium ${
                isDarkMode ? 'text-gray-300' : 'text-gray-700'
              }`}>
                TJR Highs:
              </label>
              <button
                onClick={() => setShowTjrHighs(!showTjrHighs)}
                className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors focus:outline-none focus:ring-2 focus:ring-offset-2 ${
                  showTjrHighs
                    ? 'bg-green-600 focus:ring-green-500'
                    : isDarkMode
                      ? 'bg-gray-600 focus:ring-gray-500'
                      : 'bg-gray-200 focus:ring-gray-500'
                }`}
              >
                <span
                  className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                    showTjrHighs ? 'translate-x-6' : 'translate-x-1'
                  }`}
                />
              </button>
              <span className={`text-xs ${
                showTjrHighs ? 'text-green-600' : isDarkMode ? 'text-gray-500' : 'text-gray-400'
              }`}>
                {showTjrHighs ? 'ON' : 'OFF'}
              </span>
            </div>
            {/* TJR Lows Toggle */}
            <div className="flex items-center gap-2">
              <label className={`text-sm font-medium ${
                isDarkMode ? 'text-gray-300' : 'text-gray-700'
              }`}>
                TJR Lows:
              </label>
              <button
                onClick={() => setShowTjrLows(!showTjrLows)}
                className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors focus:outline-none focus:ring-2 focus:ring-offset-2 ${
                  showTjrLows
                    ? 'bg-red-600 focus:ring-red-500'
                    : isDarkMode
                      ? 'bg-gray-600 focus:ring-gray-500'
                      : 'bg-gray-200 focus:ring-gray-500'
                }`}
              >
                <span
                  className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                    showTjrLows ? 'translate-x-6' : 'translate-x-1'
                  }`}
                />
              </button>
              <span className={`text-xs ${
                showTjrLows ? 'text-red-600' : isDarkMode ? 'text-gray-500' : 'text-gray-400'
              }`}>
                {showTjrLows ? 'ON' : 'OFF'}
              </span>
            </div>
            {/* Swing Highs Toggle (all SPY timeframes) */}
            {selectedSymbol === 'spy' && swingLabelsData && (
              <div className="flex items-center gap-2">
                <label className={`text-sm font-medium ${
                  isDarkMode ? 'text-blue-300' : 'text-blue-700'
                }`}>
                  Swing Highs:
                </label>
                <button
                  onClick={() => setShowSwingHighs(!showSwingHighs)}
                  className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors focus:outline-none focus:ring-2 focus:ring-offset-2 ${
                    showSwingHighs
                      ? 'bg-blue-600 focus:ring-blue-500'
                      : isDarkMode
                        ? 'bg-gray-600 focus:ring-gray-500'
                        : 'bg-gray-200 focus:ring-gray-500'
                  }`}
                >
                  <span
                    className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                      showSwingHighs ? 'translate-x-6' : 'translate-x-1'
                    }`}
                  />
                </button>
                <span className={`text-xs ${
                  showSwingHighs ? 'text-blue-600' : isDarkMode ? 'text-gray-500' : 'text-gray-400'
                }`}>
                  {showSwingHighs ? 'ON' : 'OFF'}
                </span>
              </div>
            )}
            {/* Swing Lows Toggle (all SPY timeframes) */}
            {selectedSymbol === 'spy' && swingLabelsData && (
              <div className="flex items-center gap-2">
                <label className={`text-sm font-medium ${
                  isDarkMode ? 'text-orange-300' : 'text-orange-700'
                }`}>
                  Swing Lows:
                </label>
                <button
                  onClick={() => setShowSwingLows(!showSwingLows)}
                  className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors focus:outline-none focus:ring-2 focus:ring-offset-2 ${
                    showSwingLows
                      ? 'bg-orange-500 focus:ring-orange-400'
                      : isDarkMode
                        ? 'bg-gray-600 focus:ring-gray-500'
                        : 'bg-gray-200 focus:ring-gray-500'
                  }`}
                >
                  <span
                    className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                      showSwingLows ? 'translate-x-6' : 'translate-x-1'
                    }`}
                  />
                </button>
                <span className={`text-xs ${
                  showSwingLows ? 'text-orange-500' : isDarkMode ? 'text-gray-500' : 'text-gray-400'
                }`}>
                  {showSwingLows ? 'ON' : 'OFF'}
                </span>
              </div>
            )}
          </div>
        </div>
        {(showTjrHighs || showTjrLows || showSwingHighs || showSwingLows) && (
          <div className={`mt-3 text-sm ${isDarkMode ? 'text-gray-400' : 'text-gray-600'}`}>
            Displaying: {showTjrHighs ? 'TJR High "T" markers' : ''} {showTjrHighs && showTjrLows ? ' & ' : ''} {showTjrLows ? 'TJR Low "⊥" markers' : ''} {(showTjrHighs || showTjrLows) && (showSwingHighs || showSwingLows) ? ' & ' : ''} {showSwingHighs ? 'Swing High "▲" markers' : ''} {showSwingHighs && showSwingLows ? ' & ' : ''} {showSwingLows ? 'Swing Low "▼" markers' : ''} on chart
          </div>
        )}
        {/* Show which tables support TJR and swing labels */}
        <div className={`mt-2 text-xs ${isDarkMode ? 'text-gray-500' : 'text-gray-500'}`}>
          <div className="mb-1">
            💡 TJR labels available for: SPY tables with labeled data
            {availableLabeledTables.length > 0 ? (
              <span className="ml-1">({availableLabeledTables.join(', ')})</span>
            ) : (
              <span className="ml-1">(checking availability...)</span>
            )}
          </div>
          <div>
            🔵 Swing labels available for: SPY tables with swing data
            {availableSwingTables.length > 0 ? (
              <span className="ml-1">({availableSwingTables.join(', ')})</span>
            ) : (
              <span className="ml-1">(checking availability...)</span>
            )}
          </div>
          {selectedSymbol !== 'spy' && (
            <div className="mt-1">
              <span className="text-orange-500">⚠️ Current symbol ({selectedSymbol}) may not have labeled or swing data</span>
            </div>
          )}
        </div>
      </div>
      
      {/* Status Display */}
      {(loading || error || labelsData || matchingLabels.length > 0) && (
        <div className={`px-6 py-3 border-t transition-colors duration-200 ${
          isDarkMode ? 'border-gray-700 bg-gray-900' : 'border-gray-200 bg-gray-50'
        }`}>
          <div className="space-y-2">
            {loading && (
              <div className={`text-sm ${isDarkMode ? 'text-blue-400' : 'text-blue-600'}`}>
                🔄 Loading labels data...
              </div>
            )}
            
            {error && (
              <div className={`text-sm ${isDarkMode ? 'text-red-400' : 'text-red-600'}`}>
                ❌ Network Error: {error}
                <div className={`text-xs mt-1 ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>
                  Unable to connect to labels service. Check your connection.
                </div>
              </div>
            )}
            
            {!loading && !error && (
              <div className={`text-xs ${isDarkMode ? 'text-gray-400' : 'text-gray-600'}`}>
                {labelsData && labelsData.length > 0 ? (
                  <>
                    📊 Labels loaded from {selectedSymbol}{selectedTimeframe}_labeled: {labelsData.length} total 
                    ({labelsData.filter(l => l.label === 'tjr_high').length} highs, {labelsData.filter(l => l.label === 'tjr_low').length} lows)
                  </>
                ) : (
                  <>
                    📊 No labeled data available for {selectedSymbol}{selectedTimeframe}. 
                    The {selectedSymbol}{selectedTimeframe}_labeled table may not exist or may be empty.
                  </>
                )}
              </div>
            )}
            
            {matchingLabels.length > 0 && (
              <div className={`text-xs ${isDarkMode ? 'text-gray-400' : 'text-gray-600'}`}>
                🎯 Found {matchingLabels.length} matching candles in current chart
                ({matchingLabels.filter(m => m.label.label === 'tjr_high').length} highs, {matchingLabels.filter(m => m.label.label === 'tjr_low').length} lows)
              </div>
            )}
            
            {((showTjrHighs || showTjrLows) && matchingLabels.length > 0) || ((showSwingHighs || showSwingLows) && matchingSwingLabels.length > 0) && (
              <div className={`mt-3 p-2 rounded border ${
                isDarkMode ? 'bg-gray-800 border-gray-600' : 'bg-white border-gray-300'
              }`}>
                <div className={`text-xs font-medium mb-2 ${isDarkMode ? 'text-gray-300' : 'text-gray-700'}`}>
                  📊 Marker Legend:
                </div>
                <div className="grid grid-cols-2 gap-3 text-xs">
                  <div>
                    <div className={`font-medium mb-1 ${isDarkMode ? 'text-gray-300' : 'text-gray-700'}`}>TJR Markers:</div>
                    {showTjrHighs && (
                      <div className="flex items-center gap-1 mb-1">
                        <div className="w-4 h-4 bg-green-500 rounded-full flex items-center justify-center text-white text-xs font-bold">T</div>
                        <span className={isDarkMode ? 'text-gray-400' : 'text-gray-600'}>TJR High (highest in range)</span>
                      </div>
                    )}
                    {showTjrLows && (
                      <div className="flex items-center gap-1">
                        <div className="w-4 h-4 bg-red-500 rounded-full flex items-center justify-center text-white text-xs font-bold">⊥</div>
                        <span className={isDarkMode ? 'text-gray-400' : 'text-gray-600'}>TJR Low (lowest in range)</span>
                      </div>
                    )}
                  </div>
                  <div>
                    <div className={`font-medium mb-1 ${isDarkMode ? 'text-gray-300' : 'text-gray-700'}`}>Swing Markers (SPY All Timeframes):</div>
                    {showSwingHighs && (
                      <div className="flex items-center gap-1 mb-1">
                        <div className="w-4 h-4 bg-blue-600 rounded flex items-center justify-center text-white text-xs font-bold">▲</div>
                        <span className={isDarkMode ? 'text-gray-400' : 'text-gray-600'}>Swing High (exact candle)</span>
                      </div>
                    )}
                    {showSwingLows && (
                      <div className="flex items-center gap-1">
                        <div className="w-4 h-4 bg-orange-500 rounded flex items-center justify-center text-white text-xs font-bold">▼</div>
                        <span className={isDarkMode ? 'text-gray-400' : 'text-gray-600'}>Swing Low (exact candle)</span>
                      </div>
                    )}
                  </div>
                </div>
                <div className={`mt-2 text-xs ${isDarkMode ? 'text-gray-500' : 'text-gray-500'}`}>
                  💡 TJR markers show the highest/lowest point within a time range, while swing markers show exact pivot points
                </div>
              </div>
            )}
            {matchingSwingLabels.length > 0 && (
              <div className={`text-xs ${isDarkMode ? 'text-blue-400' : 'text-blue-600'}`}>
                🟦 Found {matchingSwingLabels.length} matching swing candles in current chart
                ({matchingSwingLabels.filter(m => m.label.label === 'swing_high').length} swing highs, {matchingSwingLabels.filter(m => m.label.label === 'swing_low').length} swing lows)
              </div>
            )}
            
            {(showSwingHighs || showSwingLows) && matchingSwingLabels.length === 0 && swingLabelsData && swingLabelsData.length > 0 && (
              <div className={`text-xs ${isDarkMode ? 'text-orange-400' : 'text-orange-600'}`}>
                ⚠️ No swing markers matched any candles in the current chart. Check timestamp format or zoom out.
              </div>
            )}
            
            {selectedSymbol === 'spy' && swingLabelsData && swingLabelsData.length > 0 && (
              <div className={`text-xs ${isDarkMode ? 'text-blue-400' : 'text-blue-600'}`}>
                📊 Swing labels loaded for {selectedSymbol}{selectedTimeframe}: {swingLabelsData.length} total 
                ({swingLabelsData.filter(l => l.label === 'swing_high').length} highs, {swingLabelsData.filter(l => l.label === 'swing_low').length} lows)
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

const FvgLabelsControl = ({ isDarkMode, chartRef, priceSeriesRef, chartDataRef, persistentFvgDataRef, gameMode, isSelectionMode }) => {
  const [showFvgGreens, setShowFvgGreens] = useState(false);
  const [showFvgReds, setShowFvgReds] = useState(false);
  const [fvgLabelsData, setFvgLabelsData] = useState(null);
  const [matchingFvgLabels, setMatchingFvgLabels] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [hasFvgTable, setHasFvgTable] = useState(false);
  const [availableFvgTables, setAvailableFvgTables] = useState([]);
  const [isFvgLoaded, setIsFvgLoaded] = useState(false);

  const {
    selectedSymbol,
    selectedTimeframe
  } = useTrading();

  // Check which FVG tables are available with retry logic
  useEffect(() => {
    const checkAvailableFvgTables = async (retryCount = 0) => {
      const spyTimeframes = ['1m', '5m', '15m', '30m', '1h', '4h', '1d'];
      const availableFvg = [];
      
      for (const tf of spyTimeframes) {
        try {
          const response = await fetch(`http://localhost:8000/api/trading/fvg-labels/spy/${tf}`);
          if (response.ok) {
            const data = await response.json();
            if (Array.isArray(data) && data.length > 0) {
              availableFvg.push(`spy${tf}_fvg`);
            } else if (Array.isArray(data)) {
              // Table exists but is empty
              availableFvg.push(`spy${tf}_fvg (empty)`);
            }
          } else if (response.status === 404) {
            console.log(`FVG table spy${tf}_fvg not found (404)`);
          } else {
            console.warn(`FVG endpoint returned ${response.status} for spy${tf}`);
          }
        } catch (err) {
          console.warn(`Error checking FVG table spy${tf}:`, err);
          
          // If this is the first attempt and we get a connection error, retry in 2 seconds
          if (retryCount === 0 && err.message.includes('Failed to fetch')) {
            console.log('Backend may not be ready yet, will retry FVG table check in 2 seconds...');
            setTimeout(() => checkAvailableFvgTables(1), 2000);
            return;
          }
        }
      }
      
      setAvailableFvgTables(availableFvg);
      console.log(`🟢 FVG table availability check complete:`, availableFvg);
    };
    
    // Initial check with retry capability
    checkAvailableFvgTables();
  }, []);

  // Fetch FVG data when component mounts with retry logic
  useEffect(() => {
    const fetchFvgData = async (retryCount = 0) => {
      if (!selectedSymbol || !selectedTimeframe) return;
      
      setLoading(true);
      setError(null);
      
      try {
        const endpoint = `http://localhost:8000/api/trading/fvg-labels/${selectedSymbol}/${selectedTimeframe}`;
        console.log('🟢 Fetching FVG labels from:', endpoint);
        
        const response = await fetch(endpoint);
        if (!response.ok) {
          if (response.status === 404) {
            console.log(`✅ FVG table ${selectedSymbol}${selectedTimeframe}_fvg not found (expected if no FVG data)`);
            setFvgLabelsData([]);
            setHasFvgTable(false);
            setIsFvgLoaded(true);
            setError(null); // Clear any previous errors
          } else {
            console.warn(`FVG endpoint returned ${response.status} for ${selectedSymbol}${selectedTimeframe}`);
            setFvgLabelsData([]);
            setHasFvgTable(false);
            setIsFvgLoaded(true);
          }
          return;
        }
        
        const data = await response.json();
        console.log('🟢 FVG data received:', data);
        
        if (Array.isArray(data)) {
          setFvgLabelsData(data);
          setHasFvgTable(data.length > 0);
          console.log('📊 FVG stats:', {
            total: data.length,
            fvg_green: data.filter(l => l.label === 'fvg_green').length,
            fvg_red: data.filter(l => l.label === 'fvg_red').length
          });
          setIsFvgLoaded(true);
          setError(null); // Clear any previous errors
        } else {
          setFvgLabelsData([]);
          setHasFvgTable(false);
          setIsFvgLoaded(true);
        }
        
      } catch (err) {
        console.error('❌ Error fetching FVG labels:', err);
        
        // If this is the first attempt and we get a connection error, retry
        if (retryCount === 0 && err.message.includes('Failed to fetch')) {
          console.log('Backend may not be ready yet, retrying FVG fetch in 1 second...');
          setTimeout(() => fetchFvgData(1), 1000);
          return;
        }
        
        if (err.message.includes('Failed to fetch') || err.message.includes('JSON')) {
          setError(err.message);
        } else {
          setError(null);
        }
        setFvgLabelsData([]);
        setHasFvgTable(false);
        setIsFvgLoaded(true);
      } finally {
        setLoading(false);
      }
    };

    fetchFvgData();
  }, [selectedSymbol, selectedTimeframe]);

  // Scan current candles for matching FVG labels (3 candles per FVG)
  useEffect(() => {
    if (!fvgLabelsData || fvgLabelsData.length === 0 || !chartDataRef?.current) {
      setMatchingFvgLabels([]);
      return;
    }

    console.log('🔍 Scanning candles for FVG labels...');
    const candles = chartDataRef.current;
    const matches = [];

    fvgLabelsData.forEach(label => {
      if (!label.pointer || !Array.isArray(label.pointer) || label.pointer.length < 3) {
        return;
      }
      
      // FVG pointers should have 3 timestamps for the 3 candles
      const fvgTimestamps = label.pointer.map(ts => new Date(ts).getTime());
      const matchingCandles = [];
      
      // Find all 3 candles that match the FVG timestamps
      candles.forEach(candle => {
        const candleTime = new Date(candle.timestamp).getTime();
        if (fvgTimestamps.includes(candleTime)) {
          matchingCandles.push(candle);
        }
      });
      
      // If we found all 3 candles for this FVG, add them as a match
      if (matchingCandles.length === 3) {
        // Sort candles by timestamp to ensure proper order
        matchingCandles.sort((a, b) => new Date(a.timestamp) - new Date(b.timestamp));
        
        matches.push({
          candles: matchingCandles,
          label: label,
          fvgTimestamps: fvgTimestamps
        });
      }
    });

    console.log(`🎯 Found ${matches.length} matching FVG groups:`, matches);
    setMatchingFvgLabels(matches);
  }, [fvgLabelsData, chartDataRef?.current]);

  // Handle toggle changes and update chart markers
  useEffect(() => {
    if (!chartRef?.current || !priceSeriesRef?.current) {
      return;
    }

    console.log('🔄 Updating FVG chart markers:', { 
      showFvgGreens, 
      showFvgReds,
      fvgMatches: matchingFvgLabels.length 
    });
    
    // Trigger the main chart highlighting update which will include FVG markers
    const updateEvent = new CustomEvent('updateChartHighlights');
    document.dispatchEvent(updateEvent);
    
    // Fit content - but not in game mode or selection mode to avoid conflicts
    if (!gameMode && !isSelectionMode) {
      try { 
        chartRef.current.timeScale().fitContent(); 
      } catch (e) {
        console.warn('Could not fit content:', e);
      }
    }
  }, [showFvgGreens, showFvgReds, matchingFvgLabels, chartRef?.current, priceSeriesRef?.current, gameMode, isSelectionMode]);

  return (
    <div 
      className={`rounded-lg shadow-md transition-colors duration-200 mt-4 ${
        isDarkMode ? 'bg-gray-800' : 'bg-white'
      }`}
      data-fvg-labels={JSON.stringify({ showFvgGreens, showFvgReds, matchingFvgLabels })}
    >
      <div className={`px-6 py-4 border-b transition-colors duration-200 ${
        isDarkMode ? 'border-gray-700' : 'border-gray-200'
      }`}>
        <div className="flex items-center justify-between">
          <div className="flex items-center">
            <h3 className={`text-lg font-semibold ${
              isDarkMode ? 'text-white' : 'text-gray-900'
            }`}>
              FVG (Fair Value Gap) Markers
              {hasFvgTable && (
                <span className="ml-2 text-xs px-2 py-1 rounded-full bg-green-100 text-green-800">
                  ✅ Available
                </span>
              )}
              {!hasFvgTable && !loading && !error && (
                <span className="ml-2 text-xs px-2 py-1 rounded-full bg-gray-100 text-gray-600">
                  ⚠️ No Data
                </span>
              )}
            </h3>
            <InfoTooltip id="fvg-labels" content={
              <div>
                <p className="font-semibold mb-2">🟢 FVG (Fair Value Gap) Trading Markers</p>
                <p className="mb-2">Visual boxes highlighting fair value gaps on the chart:</p>
                <ul className="list-disc list-inside space-y-1 text-xs">
                  <li><strong>FVG Green:</strong> Green boxes around bullish fair value gaps (3 candles)</li>
                  <li><strong>FVG Red:</strong> Red boxes around bearish fair value gaps (3 candles)</li>
                  <li><strong>Gap Detection:</strong> Shows areas where price left liquidity gaps</li>
                  <li><strong>Toggle Display:</strong> Show/hide each FVG type independently</li>
                  <li><strong>Three Candle Pattern:</strong> Each FVG encompasses exactly 3 candles</li>
                  <li><strong>Liquidity Zones:</strong> Potential areas for price reaction and fills</li>
                </ul>
                <p className="mt-2 text-xs">Toggle these on to see FVG boxes highlighting gaps on your chart.</p>
              </div>
            } isDarkMode={isDarkMode} />
          </div>
          <div className="flex items-center gap-4">
            {/* FVG Green Toggle */}
            <div className="flex items-center gap-2">
              <label className={`text-sm font-medium ${
                isDarkMode ? 'text-green-300' : 'text-green-700'
              }`}>
                FVG Green:
              </label>
              <button
                onClick={() => {
                  const newValue = !showFvgGreens;
                  setShowFvgGreens(newValue);
                  persistentFvgDataRef.current.enabled.green = newValue;
                  console.log('🟢 FVG Green toggled to:', newValue);
                }}
                className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors focus:outline-none focus:ring-2 focus:ring-offset-2 ${
                  showFvgGreens
                    ? 'bg-green-600 focus:ring-green-500'
                    : isDarkMode
                      ? 'bg-gray-600 focus:ring-gray-500'
                      : 'bg-gray-200 focus:ring-gray-500'
                }`}
              >
                <span
                  className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                    showFvgGreens ? 'translate-x-6' : 'translate-x-1'
                  }`}
                />
              </button>
              <span className={`text-xs ${
                showFvgGreens ? 'text-green-600' : isDarkMode ? 'text-gray-500' : 'text-gray-400'
              }`}>
                {showFvgGreens ? 'ON' : 'OFF'}
              </span>
            </div>
            
            {/* FVG Red Toggle */}
            <div className="flex items-center gap-2">
              <label className={`text-sm font-medium ${
                isDarkMode ? 'text-red-300' : 'text-red-700'
              }`}>
                FVG Red:
              </label>
              <button
                onClick={() => {
                  const newValue = !showFvgReds;
                  setShowFvgReds(newValue);
                  persistentFvgDataRef.current.enabled.red = newValue;
                  console.log('🔴 FVG Red toggled to:', newValue);
                }}
                className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors focus:outline-none focus:ring-2 focus:ring-offset-2 ${
                  showFvgReds
                    ? 'bg-red-600 focus:ring-red-500'
                    : isDarkMode
                      ? 'bg-gray-600 focus:ring-gray-500'
                      : 'bg-gray-200 focus:ring-gray-500'
                }`}
              >
                <span
                  className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                    showFvgReds ? 'translate-x-6' : 'translate-x-1'
                  }`}
                />
              </button>
              <span className={`text-xs ${
                showFvgReds ? 'text-red-600' : isDarkMode ? 'text-gray-500' : 'text-gray-400'
              }`}>
                {showFvgReds ? 'ON' : 'OFF'}
              </span>
            </div>
            

          </div>
        </div>
        
        {(showFvgGreens || showFvgReds) && (
          <div className={`mt-3 text-sm ${isDarkMode ? 'text-gray-400' : 'text-gray-600'}`}>
            Displaying: {showFvgGreens ? 'Green FVG boxes' : ''} {showFvgGreens && showFvgReds ? ' & ' : ''} {showFvgReds ? 'Red FVG boxes' : ''} around 3-candle gaps
          </div>
        )}
        
        {/* Show which tables support FVG labels */}
        <div className={`mt-2 text-xs ${isDarkMode ? 'text-gray-500' : 'text-gray-500'}`}>
          <div className="mb-1">
            🟢 FVG labels available for: SPY tables with FVG data
            {availableFvgTables.length > 0 ? (
              <span className="ml-1">({availableFvgTables.join(', ')})</span>
            ) : (
              <span className="ml-1">(checking availability...)</span>
            )}
          </div>
          {selectedSymbol !== 'spy' && (
            <div className="mt-1">
              <span className="text-orange-500">⚠️ Current symbol ({selectedSymbol}) may not have FVG data</span>
            </div>
          )}
        </div>
      </div>
      
      {/* Status Display */}
      {(loading || error || fvgLabelsData || matchingFvgLabels.length > 0) && (
        <div className={`px-6 py-3 border-t transition-colors duration-200 ${
          isDarkMode ? 'border-gray-700 bg-gray-900' : 'border-gray-200 bg-gray-50'
        }`}>
          <div className="space-y-2">
            {loading && (
              <div className={`text-sm ${isDarkMode ? 'text-blue-400' : 'text-blue-600'}`}>
                🔄 Loading FVG data...
              </div>
            )}
            
            {error && (
              <div className={`text-sm ${isDarkMode ? 'text-red-400' : 'text-red-600'}`}>
                ❌ Network Error: {error}
                <div className={`text-xs mt-1 ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>
                  Unable to connect to FVG labels service. Check your connection.
                </div>
              </div>
            )}
            
            {!loading && !error && (
              <div className={`text-xs ${isDarkMode ? 'text-gray-400' : 'text-gray-600'}`}>
                {fvgLabelsData && fvgLabelsData.length > 0 ? (
                  <>
                    📊 FVG data loaded from {selectedSymbol}{selectedTimeframe}_fvg: {fvgLabelsData.length} total 
                    ({fvgLabelsData.filter(l => l.label === 'fvg_green').length} green, {fvgLabelsData.filter(l => l.label === 'fvg_red').length} red)
                  </>
                ) : (
                  <>
                    📊 No FVG data available for {selectedSymbol}{selectedTimeframe}. 
                    The {selectedSymbol}{selectedTimeframe}_fvg table may not exist or may be empty.
                  </>
                )}
              </div>
            )}
            
            {matchingFvgLabels.length > 0 && (
              <div className={`text-xs ${isDarkMode ? 'text-gray-400' : 'text-gray-600'}`}>
                🎯 Found {matchingFvgLabels.length} matching FVG groups in current chart
                ({matchingFvgLabels.filter(m => m.label.label === 'fvg_green').length} green, {matchingFvgLabels.filter(m => m.label.label === 'fvg_red').length} red)
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

const TradingChartDashboard = ({ 
  tables,
  isDarkMode 
}) => {
  // Use shared trading context
  const {
    selectedSymbol,
    setSelectedSymbol,
    selectedTimeframe,
    setSelectedTimeframe,
    rowLimit,
    setRowLimit,
    sortOrder,
    setSortOrder,
    selectedCandles,
    addSelectedCandle,
    removeSelectedCandle,
    // Date range functionality
    fetchMode,
    setFetchMode,
    dateRangeType,
    setDateRangeType,
    startDate,
    setStartDate,
    endDate,
    setEndDate
  } = useTrading();

  // Debug: Log when context values change (can be removed in production)
  useEffect(() => {
    console.log('📈 Chart Dashboard - Trading context changed:', {
      selectedSymbol,
      selectedTimeframe,
      rowLimit,
      sortOrder
    });
  }, [selectedSymbol, selectedTimeframe, rowLimit, sortOrder]);
  const [chartData, setChartData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [chartType, setChartType] = useState('candlestick'); // candlestick, line, area
  const [timeRange, setTimeRange] = useState('all'); // all, 1d, 7d, 30d, 90d
  
  // Candle selection states
  const [isSelectionMode, setIsSelectionMode] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState(null);
  const [dragEnd, setDragEnd] = useState(null);
  
  // Enhanced selection states
  const [hoveredCandle, setHoveredCandle] = useState(null);
  const [selectionStats, setSelectionStats] = useState(null);
  
  // Crosshair state for selection mode
  const [crosshairPosition, setCrosshairPosition] = useState(null);
  const [showCustomCrosshair, setShowCustomCrosshair] = useState(false);
  const [enableSelectionCrosshair, setEnableSelectionCrosshair] = useState(true);
  
  // Multi-timeframe analysis states
  const [showSecondaryChart, setShowSecondaryChart] = useState(false);
  const [secondaryTimeframe, setSecondaryTimeframe] = useState('15m');
  const [secondaryChartData, setSecondaryChartData] = useState(null);
  const [secondaryLoading, setSecondaryLoading] = useState(false);
  const [secondaryError, setSecondaryError] = useState(null);
  const [correlatedCandles, setCorrelatedCandles] = useState([]);
  const [hasSequentialSelection, setHasSequentialSelection] = useState(false);
  const [showBreakdownOverlay, setShowBreakdownOverlay] = useState(true);
  // Indicates when main chart finished rendering
  const [chartReady, setChartReady] = useState(false);
  const [gameMode, setGameMode] = useState(false);
  
  const chartContainerRef = useRef(null);
  const chartRef = useRef(null);
  const selectionOverlayRef = useRef(null);
  const priceSeriesRef = useRef(null);
  const chartDataRef = useRef(null); // Store original chart data
  const timeScaleRef = useRef(null); // Store timescale reference
  // FVG markers are now handled via series markers, no primitive ref needed
  const persistentFvgDataRef = useRef({ boxes: [], enabled: { red: false, green: false } }); // Persistent FVG data
  
  // Secondary chart refs
  const secondaryChartContainerRef = useRef(null);
  const secondaryChartRef = useRef(null);
  const secondaryPriceSeriesRef = useRef(null);
  const secondaryChartDataRef = useRef(null);

  const API_BASE_URL = 'http://localhost:8000/api/trading';

  // Helper function to get timeframe duration in minutes
  const getTimeframeMinutes = (timeframe) => {
    const timeframeMap = {
      '1m': 1,
      '5m': 5,
      '15m': 15,
      '30m': 30,
      '1h': 60,
      '4h': 240,
      '1d': 1440
    };
    return timeframeMap[timeframe] || 60;
  };

  // Helper function to determine if timeframes are compatible for correlation
  const areTimeframesCompatible = (primary, secondary) => {
    const primaryMinutes = getTimeframeMinutes(primary);
    const secondaryMinutes = getTimeframeMinutes(secondary);
    
    // Secondary should be smaller timeframe than primary
    if (secondaryMinutes >= primaryMinutes) return false;
    
    // Primary should be evenly divisible by secondary
    return primaryMinutes % secondaryMinutes === 0;
  };

  // Function to check if selected candles are sequential (allows single candle)
  const areSelectedCandlesSequential = () => {
    const currentSelection = selectedCandles.filter(c => 
      c.symbol === selectedSymbol && c.timeframe === selectedTimeframe
    );
    
    if (currentSelection.length === 0) return false;
    
    // Single candle is always considered "sequential"
    if (currentSelection.length === 1) return true;
    
    // Sort by source index
    const sortedSelection = [...currentSelection].sort((a, b) => a.sourceIndex - b.sourceIndex);
    
    // Check if indices are consecutive
    for (let i = 1; i < sortedSelection.length; i++) {
      if (sortedSelection[i].sourceIndex !== sortedSelection[i-1].sourceIndex + 1) {
        return false;
      }
    }
    
    return true;
  };

  // Function to find correlated candles for ALL selected sequential candles
  const findCorrelatedCandlesForSelection = (selectedCandlesList, secondaryData) => {
    if (!selectedCandlesList || selectedCandlesList.length === 0 || !secondaryData || secondaryData.length === 0) {
      return [];
    }
    
    const primaryMinutes = getTimeframeMinutes(selectedTimeframe);
    const allCorrelated = [];
    
    selectedCandlesList.forEach(primaryCandle => {
      const primaryTime = new Date(primaryCandle.timestamp);
      
      // Calculate the time range for this primary candle
      const primaryStart = new Date(primaryTime);
      const primaryEnd = new Date(primaryTime.getTime() + (primaryMinutes * 60 * 1000));
      
      // Find all secondary candles that fall within this time range
      const correlated = secondaryData.filter(secondaryCandle => {
        const secondaryTime = new Date(secondaryCandle.timestamp);
        return secondaryTime >= primaryStart && secondaryTime < primaryEnd;
      });
      
      allCorrelated.push(...correlated);
    });
    
    // Remove duplicates and sort by timestamp
    const uniqueCorrelated = allCorrelated.filter((candle, index, self) => 
      index === self.findIndex(c => c.timestamp === candle.timestamp)
    );
    
    return uniqueCorrelated.sort((a, b) => new Date(a.timestamp) - new Date(b.timestamp));
  };

  const updateSelectionStats = useCallback(() => {
    const currentSelection = selectedCandles.filter(c => 
      c.symbol === selectedSymbol && c.timeframe === selectedTimeframe
    );
    
    if (currentSelection.length === 0) {
      setSelectionStats(null);
      return;
    }
    
    const prices = currentSelection.map(c => c.close);
    const avgPrice = prices.reduce((a, b) => a + b, 0) / prices.length;
    const minPrice = Math.min(...prices);
    const maxPrice = Math.max(...prices);
    const totalChange = currentSelection.reduce((sum, c) => sum + c.change, 0);
    const avgChange = totalChange / currentSelection.length;
    
    const timestamps = currentSelection.map(c => new Date(c.timestamp).getTime());
    const timeRange = Math.max(...timestamps) - Math.min(...timestamps);
    const timeRangeStr = formatTimeRange(timeRange);
    
    setSelectionStats({
      count: currentSelection.length,
      avgPrice,
      minPrice,
      maxPrice,
      totalChange,
      avgChange,
      timeRange: timeRangeStr,
      bullish: currentSelection.filter(c => c.change > 0).length,
      bearish: currentSelection.filter(c => c.change < 0).length,
    });
  }, [selectedCandles, selectedSymbol, selectedTimeframe]);

  const updateChartHighlights = useCallback(() => {
    // DEBUG: Log toggle states
    const tjrLabelsElement = document.querySelector('[data-tjr-labels]');
    let tjrState = { showTjrHighs: false, showTjrLows: false, matchingLabels: [] };
    if (tjrLabelsElement) {
      try {
        tjrState = JSON.parse(tjrLabelsElement.getAttribute('data-tjr-labels'));
      } catch (e) {
        console.warn('Could not parse TJR labels state');
      }
    }
    console.log('--- updateChartHighlights DEBUG ---');
    console.log('showTjrHighs:', tjrState.showTjrHighs, 'showTjrLows:', tjrState.showTjrLows);
    
    if (!chartRef.current || !priceSeriesRef.current || !chartDataRef.current) {
      console.log('⚠️ Missing dependencies for chart highlights');
      return;
    }

    try {
      // Get selected candles for current symbol/timeframe
      const currentSymbolTimeframeSelection = selectedCandles.filter(c => 
        c.symbol === selectedSymbol && c.timeframe === selectedTimeframe
      );

      console.log('🎯 Current selection for chart:', {
        totalSelected: selectedCandles.length,
        currentSymbolTimeframe: currentSymbolTimeframeSelection.length,
        selectedCandles: currentSymbolTimeframeSelection.map(c => c.timestamp)
      });

      // Get TJR and swing labels state
      const tjrLabelsElement = document.querySelector('[data-tjr-labels]');
      let tjrState = { 
        showTjrHighs: false, 
        showTjrLows: false, 
        matchingLabels: [],
        showSwingHighs: false,
        showSwingLows: false,
        matchingSwingLabels: []
      };
      
      if (tjrLabelsElement) {
        try {
          tjrState = JSON.parse(tjrLabelsElement.getAttribute('data-tjr-labels'));
          console.log('🎛️ TJR & Swing State:', tjrState);
        } catch (e) {
          console.warn('Could not parse TJR labels state');
        }
      } else {
        console.warn('⚠️ TJR labels element not found');
      }

      // Get FVG labels state
      const fvgLabelsElement = document.querySelector('[data-fvg-labels]');
      let fvgState = { 
        showFvgGreens: false, 
        showFvgReds: false, 
        matchingFvgLabels: []
      };
      
      if (fvgLabelsElement) {
        try {
          fvgState = JSON.parse(fvgLabelsElement.getAttribute('data-fvg-labels'));
          console.log('🟢 FVG State:', fvgState);
        } catch (e) {
          console.warn('Could not parse FVG labels state');
        }
      } else {
        console.warn('⚠️ FVG labels element not found');
      }

      // Create TJR labels lookup map
      const tjrLabelsByTimestamp = {};
      if (tjrState.matchingLabels && Array.isArray(tjrState.matchingLabels)) {
        tjrState.matchingLabels.forEach(match => {
          if (match.candle && match.label) {
            const timestamp = match.candle.timestamp;
            if (!tjrLabelsByTimestamp[timestamp]) {
              tjrLabelsByTimestamp[timestamp] = [];
            }
            tjrLabelsByTimestamp[timestamp].push(match.label);
          }
        });
      }
      
      // Update candlestick data with ONLY selected candle highlighting (no TJR colors)
      const selectedTimestamps = new Set(currentSymbolTimeframeSelection.map(c => c.timestamp));
      
      const highlightedCandlestickData = chartDataRef.current.map(candle => {
        const isSelected = selectedTimestamps.has(candle.timestamp);
        const isBullish = parseFloat(candle.close) >= parseFloat(candle.open);
        
        const baseData = {
          time: Math.floor(new Date(candle.timestamp).getTime() / 1000),
          open: parseFloat(candle.open),
          high: parseFloat(candle.high),
          low: parseFloat(candle.low),
          close: parseFloat(candle.close),
        };

        // Only highlight selected candles with blue color
        if (isSelected) {
          return {
            ...baseData,
            color: '#3b82f6', // Blue body
            wickColor: isBullish ? '#22c55e' : '#ef4444', // Keep original wick colors
            borderColor: '#2563eb' // Darker blue border
          };
        }

        return baseData;
      });

      // --- MARKER & SERIES UPDATE STRATEGY ---
      // Always rebuild the price series from scratch, then set a SINGLE combined
      // marker array (highs+ lows depending on toggles). This guarantees we never
      // layer marker primitives and therefore prevents ghost markers.
      if (!chartRef.current) {
        console.warn('⚠️ Chart reference missing – cannot update markers.');
        return;
      }

      // 1️⃣ Remove existing series (if it still exists)
      try {
        if (priceSeriesRef.current) {
          console.log('🧹 Removing existing price series to ensure a clean marker state…');
          chartRef.current.removeSeries(priceSeriesRef.current);
        }
      } catch (e) {
        console.warn('⚠️ Could not remove price series (may have been removed already):', e);
      }

      // 2️⃣ Re-add a fresh series of the correct type
      let newSeries;
      if (chartType === 'candlestick') {
        newSeries = chartRef.current.addSeries(CandlestickSeries, {
          upColor: '#22c55e',
          downColor: '#ef4444',
          borderVisible: false,
          wickUpColor: '#22c55e',
          wickDownColor: '#ef4444',
        });
        newSeries.setData(highlightedCandlestickData);
      } else if (chartType === 'line') {
        newSeries = chartRef.current.addSeries(LineSeries, {
          color: isDarkMode ? '#60a5fa' : '#1d4ed8',
          lineWidth: 2,
        });
        newSeries.setData(highlightedCandlestickData.map(d => ({ time: d.time, value: d.close })));
      } else {
        newSeries = chartRef.current.addSeries(AreaSeries, {
          topColor: 'rgba(59,130,246,0.4)',
          bottomColor: 'rgba(59,130,246,0.0)',
          lineColor: isDarkMode ? '#60a5fa' : '#1d4ed8',
          lineWidth: 2,
        });
        newSeries.setData(highlightedCandlestickData.map(d => ({ time: d.time, value: d.close })));
      }
      priceSeriesRef.current = newSeries;

      // --- Build the combined marker array (TJR and swing markers) ---
      const allMarkers = [];
      
      // Convert FVG boxes to series markers for proper layering
      const fvgMarkers = [];
      
      // Check persistent state first - use it if toggles are enabled
      const shouldShowReds = persistentFvgDataRef.current.enabled.red && fvgState.showFvgReds;
      const shouldShowGreens = persistentFvgDataRef.current.enabled.green && fvgState.showFvgGreens;
      
      console.log('🔍 FVG State Check:', {
        showFvgGreens: fvgState.showFvgGreens,
        showFvgReds: fvgState.showFvgReds,
        persistentRed: persistentFvgDataRef.current.enabled.red,
        persistentGreen: persistentFvgDataRef.current.enabled.green,
        shouldShowReds,
        shouldShowGreens,
        matchingFvgLabels: fvgState.matchingFvgLabels?.length,
        persistentBoxes: persistentFvgDataRef.current.boxes?.length
      });
      
      // Process FVG data and convert to markers
      if (fvgState.showFvgGreens || fvgState.showFvgReds) {
        console.log(`🔄 Processing FVG data: ${fvgState.matchingFvgLabels?.length || 0} FVG groups, showGreens: ${fvgState.showFvgGreens}, showReds: ${fvgState.showFvgReds}`);
        if (fvgState.matchingFvgLabels && Array.isArray(fvgState.matchingFvgLabels)) {
          fvgState.matchingFvgLabels.forEach(fvgMatch => {
            const { candles, label } = fvgMatch;
            
            console.log(`🔍 Processing FVG: ${label.label}, showGreens: ${fvgState.showFvgGreens}, showReds: ${fvgState.showFvgReds}`);
            
            // Only show if the corresponding toggle is on
            if ((label.label === 'fvg_green' && !fvgState.showFvgGreens) ||
                (label.label === 'fvg_red' && !fvgState.showFvgReds)) {
              console.log(`🔍 Skipping ${label.label} due to toggle state`);
              return;
            }
            
            if (candles && candles.length === 3) {
              // Sort candles by time
              const sortedCandles = [...candles].sort((a, b) => new Date(a.timestamp) - new Date(b.timestamp));
              
              // Calculate FVG gap prices
              let gapHigh, gapLow;
              if (label.label === 'fvg_green') {
                // For bullish FVG: gap between candle 1 high and candle 3 low
                gapHigh = parseFloat(sortedCandles[2].low);
                gapLow = parseFloat(sortedCandles[0].high);
              } else {
                // For bearish FVG: gap between candle 1 low and candle 3 high  
                gapHigh = parseFloat(sortedCandles[0].low);
                gapLow = parseFloat(sortedCandles[2].high);
              }
              
              // Ensure correct order (high should be higher than low)
              if (gapHigh < gapLow) {
                [gapHigh, gapLow] = [gapLow, gapHigh];
              }
              
              // Create markers for the FVG gap
              const startTime = Math.floor(new Date(sortedCandles[0].timestamp).getTime() / 1000);
              const endTime = Math.floor(new Date(sortedCandles[2].timestamp).getTime() / 1000);
              const midTime = Math.floor((startTime + endTime) / 2);
              const midPrice = (gapHigh + gapLow) / 2;
              
              const isGreen = label.label === 'fvg_green';
              const markerColor = isGreen ? '#22c55e' : '#ef4444';
              const markerText = isGreen ? 'FVG+' : 'FVG-';
              
              // Add marker at the middle of the gap
              fvgMarkers.push({
                time: midTime,
                position: 'inBar',
                color: markerColor,
                shape: 'square',
                size: 2,
                text: markerText
              });
              
              // Add additional markers to outline the gap
              fvgMarkers.push({
                time: startTime,
                position: 'inBar',
                color: markerColor,
                shape: 'circle',
                size: 1,
                text: ''
              });
              
              fvgMarkers.push({
                time: endTime,
                position: 'inBar',
                color: markerColor,
                shape: 'circle',
                size: 1,
                text: ''
              });
              
              console.log(`✅ Added ${label.label} FVG markers for candles: ${candles.map(c => c.timestamp).join(', ')}`);
            }
          });
        }
      }
      
      // Store the markers in persistent storage for future chart updates
      if (fvgMarkers.length > 0) {
        persistentFvgDataRef.current.fvgMarkers = fvgMarkers;
        const redCount = fvgMarkers.filter(marker => marker.color === '#ef4444').length;
        const greenCount = fvgMarkers.filter(marker => marker.color === '#22c55e').length;
        console.log(`💾 Stored ${fvgMarkers.length} FVG markers: ${redCount} red, ${greenCount} green`);
      }
      
      // Add FVG markers to allMarkers array
      allMarkers.push(...fvgMarkers);
      
      // Build TJR markers
      if (tjrState.showTjrHighs || tjrState.showTjrLows) {
        // Group matching labels by their original label (not by timestamp)
        const labelGroups = {};
        if (tjrState.matchingLabels && Array.isArray(tjrState.matchingLabels)) {
          tjrState.matchingLabels.forEach(match => {
            const labelKey = `${match.label.label}_${match.label.pointer[0]}_${match.label.pointer[1]}`;
            if (!labelGroups[labelKey]) {
              labelGroups[labelKey] = {
                label: match.label,
                candles: []
              };
            }
            labelGroups[labelKey].candles.push(match.candle);
          });
        }
        // For each label group, find the actual high/low point
        Object.values(labelGroups).forEach(group => {
          const { label, candles } = group;
          if (label.label === 'tjr_high' && tjrState.showTjrHighs) {
            let highestCandle = candles[0];
            let highestPrice = parseFloat(highestCandle.high);
            candles.forEach(candle => {
              const candleHigh = parseFloat(candle.high);
              if (candleHigh > highestPrice) {
                highestPrice = candleHigh;
                highestCandle = candle;
              }
            });
            const candleTime = Math.floor(new Date(highestCandle.timestamp).getTime() / 1000);
            allMarkers.push({
              time: candleTime,
              position: 'aboveBar',
              color: '#22c55e',
              shape: 'circle',
              text: 'T',
              size: 1
            });
          }
          if (label.label === 'tjr_low' && tjrState.showTjrLows) {
            let lowestCandle = candles[0];
            let lowestPrice = parseFloat(lowestCandle.low);
            candles.forEach(candle => {
              const candleLow = parseFloat(candle.low);
              if (candleLow < lowestPrice) {
                lowestPrice = candleLow;
                lowestCandle = candle;
              }
            });
            const candleTime = Math.floor(new Date(lowestCandle.timestamp).getTime() / 1000);
            allMarkers.push({
              time: candleTime,
              position: 'belowBar',
              color: '#ef4444',
              shape: 'circle',
              text: '⊥',
              size: 1
            });
          }
        });
      }
      
      // Build swing markers (exact timestamp matches)
      if (tjrState.showSwingHighs || tjrState.showSwingLows) {
        if (tjrState.matchingSwingLabels && Array.isArray(tjrState.matchingSwingLabels)) {
          tjrState.matchingSwingLabels.forEach(match => {
            if (match.label.label === 'swing_high' && tjrState.showSwingHighs) {
              allMarkers.push({
                time: Math.floor(new Date(match.candle.timestamp).getTime() / 1000),
                position: 'aboveBar',
                color: '#2563eb', // blue
                shape: 'circle',
                text: '▲',
                size: 1
              });
            }
            if (match.label.label === 'swing_low' && tjrState.showSwingLows) {
              allMarkers.push({
                time: Math.floor(new Date(match.candle.timestamp).getTime() / 1000),
                position: 'belowBar',
                color: '#f59e0b', // orange
                shape: 'circle',
                text: '▼',
                size: 1
              });
            }
          });
        }
      }

      // 3️⃣ Apply markers (single combined set)
      if (allMarkers.length > 0) {
        createSeriesMarkers(priceSeriesRef.current, allMarkers);
        const tjrCount = allMarkers.filter(m => m.text === 'T' || m.text === '⊥').length;
        const swingCount = allMarkers.filter(m => m.text === '▲' || m.text === '▼').length;
        console.log(`✅ Applied ${allMarkers.length} total markers (${tjrCount} TJR + ${swingCount} swing) + ${fvgMarkers.length} FVG markers`);
      } else {
        createSeriesMarkers(priceSeriesRef.current, []);
        console.log('📝 No markers to display – cleared all markers');
      }



      // 5️⃣ Fit the content so the new series is rendered properly - but not in game mode or selection mode
      if (!gameMode && !isSelectionMode) {
        try {
          chartRef.current.timeScale().fitContent();
        } catch (e) {
          console.warn('⚠️ Could not fit content after series rebuild:', e);
        }
      }

      // Update selection statistics after full rebuild
      updateSelectionStats();

      console.log(`🔄 Series rebuilt & markers updated. Current markers: ${allMarkers.length}, FVG markers: ${fvgMarkers.length}`);

    } catch (error) {
      console.error('❌ Error updating chart highlights:', error);
      console.error('Error details:', error.stack);
    }
  }, [selectedCandles, selectedSymbol, selectedTimeframe, chartType, updateSelectionStats, gameMode, isSelectionMode]);

  useEffect(() => {
    if (selectedSymbol && selectedTimeframe) {
      console.log('🔄 Chart data fetch triggered:', selectedSymbol, selectedTimeframe, rowLimit, sortOrder, timeRange, fetchMode, dateRangeType, startDate, endDate); // Debug log
      fetchChartData();
    }
  }, [selectedSymbol, selectedTimeframe, rowLimit, sortOrder, timeRange, fetchMode, dateRangeType, startDate, endDate]);

  // Monitor sequential selection status
  useEffect(() => {
    const isSequential = areSelectedCandlesSequential();
    setHasSequentialSelection(isSequential);
    
    // If no longer sequential, hide secondary chart
    if (!isSequential && showSecondaryChart) {
      setShowSecondaryChart(false);
      setCorrelatedCandles([]);
    }
    
    console.log('🔄 Sequential selection status:', isSequential);
  }, [selectedCandles, selectedSymbol, selectedTimeframe]);

  // Fetch secondary chart data when secondary chart is enabled or timeframe changes
  useEffect(() => {
    if (showSecondaryChart && selectedSymbol && secondaryTimeframe && hasSequentialSelection) {
      console.log('🔄 Secondary chart data fetch triggered:', selectedSymbol, secondaryTimeframe);
      fetchSecondaryChartData();
    }
  }, [showSecondaryChart, selectedSymbol, secondaryTimeframe, hasSequentialSelection, fetchMode, dateRangeType, startDate, endDate, rowLimit, sortOrder]);

  // Initialize chart when data changes - FIXED: Ensures chart updates when symbol/timeframe changes
  useEffect(() => {
    if (chartData?.data && chartData.data.length > 0 && !loading) {
      console.log('📊 Chart initialization triggered by data change'); // Debug log
      // Small delay to ensure container is ready
      setTimeout(() => {
        initializeChart(chartData.data);
      }, 100);
    } else if (!chartData && !loading) {
      // Clear chart when no data
      if (chartRef.current) {
        console.log('🧹 Clearing chart due to no data'); // Debug log
        chartRef.current.remove();
        chartRef.current = null;
      }
    }
  }, [chartData, loading]);

  // Initialize secondary chart when data changes
  useEffect(() => {
    if (secondaryChartData?.data && secondaryChartData.data.length > 0 && !secondaryLoading && showSecondaryChart) {
      console.log('📊 Secondary chart initialization triggered by data change');
      setTimeout(() => {
        initializeSecondaryChart(secondaryChartData.data);
      }, 100);
    } else if (!showSecondaryChart && secondaryChartRef.current) {
      // Clean up secondary chart when disabled
      console.log('🧹 Cleaning up secondary chart');
      secondaryChartRef.current.remove();
      secondaryChartRef.current = null;
      secondaryPriceSeriesRef.current = null;
    }
  }, [secondaryChartData, secondaryLoading, showSecondaryChart, correlatedCandles]);

  // Reinitialize chart when theme changes or chart type changes
  useEffect(() => {
    if (chartData?.data && chartData.data.length > 0 && !loading) {
      console.log('🎨 Chart reinitialize triggered by theme/type change');
      // Small delay to ensure container is ready
      setTimeout(() => {
        initializeChart(chartData.data);
      }, 100);
    }
    
    // Also reinitialize secondary chart for theme changes
    if (secondaryChartData?.data && secondaryChartData.data.length > 0 && !secondaryLoading && showSecondaryChart) {
      console.log('🎨 Secondary chart reinitialize triggered by theme change');
      setTimeout(() => {
        initializeSecondaryChart(secondaryChartData.data);
      }, 150);
    }
  }, [isDarkMode, chartType]);

  // Cleanup chart on unmount
  useEffect(() => {
    return () => {
      if (chartRef.current) {
        chartRef.current.remove();
        chartRef.current = null;
        priceSeriesRef.current = null;
      }
      if (secondaryChartRef.current) {
        secondaryChartRef.current.remove();
        secondaryChartRef.current = null;
        secondaryPriceSeriesRef.current = null;
      }
    };
  }, []);

  // Update chart highlights when selected candles change
  useEffect(() => {
    // Add a small delay to ensure chart is fully initialized
    const timeoutId = setTimeout(() => {
      if (chartRef.current && priceSeriesRef.current && chartDataRef.current) {
        console.log('🔄 Selection changed, updating chart highlights...');
        updateChartHighlights();
      } else {
        console.log('🔄 Selection changed but chart not ready yet, skipping highlight update');
      }
    }, 50);

    return () => clearTimeout(timeoutId);
  }, [selectedCandles, selectedSymbol, selectedTimeframe, chartType]);

  // Also update highlights when chart data is initially loaded
  useEffect(() => {
    if (chartRef.current && priceSeriesRef.current && chartDataRef.current && selectedCandles.length > 0) {
      console.log('🔄 Chart data loaded, applying existing selections...');
      setTimeout(() => {
        updateChartHighlights();
      }, 200);
    }
  }, [chartData]);

  // Listen for TJR label updates
  useEffect(() => {
    let updateTimeout;
    const handleChartUpdate = () => {
      console.log('🔄 TJR labels updated, refreshing chart highlights...');
      // Clear any pending update to debounce
      if (updateTimeout) {
        clearTimeout(updateTimeout);
      }
      updateTimeout = setTimeout(() => {
        updateChartHighlights();
      }, 100);
    };

    document.addEventListener('updateChartHighlights', handleChartUpdate);
    
    return () => {
      document.removeEventListener('updateChartHighlights', handleChartUpdate);
    };
  }, [updateChartHighlights]); // Remove updateChartHighlights from dependency array since it's a stable function



  const fetchSecondaryChartData = async () => {
    if (!showSecondaryChart || !selectedSymbol || !secondaryTimeframe || !hasSequentialSelection) return;
    
    setSecondaryLoading(true);
    setSecondaryError(null);
    
    try {
      // Get the selected sequential candles
      const selectedSequentialCandles = selectedCandles.filter(c => 
        c.symbol === selectedSymbol && c.timeframe === selectedTimeframe
      ).sort((a, b) => a.sourceIndex - b.sourceIndex);
      
      if (selectedSequentialCandles.length === 0) {
        setSecondaryError('No sequential candles selected');
        return;
      }
      
      // Calculate the time range we need to fetch
      const firstCandle = selectedSequentialCandles[0];
      const lastCandle = selectedSequentialCandles[selectedSequentialCandles.length - 1];
      
      const primaryMinutes = getTimeframeMinutes(selectedTimeframe);
      const startTime = new Date(firstCandle.timestamp);
      const endTime = new Date(new Date(lastCandle.timestamp).getTime() + (primaryMinutes * 60 * 1000));
      
      console.log('📊 Fetching secondary data for time range:', startTime.toISOString(), 'to', endTime.toISOString());
      
      let url = `${API_BASE_URL}/data/${selectedSymbol}/${secondaryTimeframe}`;
      let queryParams = new URLSearchParams();

      // Use the calculated time range
      queryParams.append('start_date', startTime.toISOString());
      queryParams.append('end_date', endTime.toISOString());
      queryParams.append('limit', '10000'); // Large limit to get all data in range
      queryParams.append('order', 'asc'); // Always ascending for proper correlation
      queryParams.append('sort_by', 'timestamp');

      const response = await fetch(`${url}?${queryParams.toString()}`);
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const data = await response.json();
      console.log('📊 Secondary chart data fetched:', data.data?.length, 'candles for', selectedSequentialCandles.length, 'selected candles');
      
      // Filter to only include candles that fall within our selected primary candles' time ranges
      if (data.data && Array.isArray(data.data)) {
        const correlatedCandles = findCorrelatedCandlesForSelection(selectedSequentialCandles, data.data);
        
        // Create a new data structure with only the correlated candles
        const filteredData = {
          ...data,
          data: correlatedCandles
        };
        
        setSecondaryChartData(filteredData);
        setCorrelatedCandles(correlatedCandles);
        
        // Store secondary chart data for highlighting
        secondaryChartDataRef.current = correlatedCandles;
        
        console.log('📊 Filtered to', correlatedCandles.length, 'correlated candles');
      }
      
      setSecondaryError(null);
      
    } catch (err) {
      setSecondaryError(`Failed to fetch secondary chart data: ${err.message}`);
      console.error('❌ Secondary chart data fetch error:', err);
    } finally {
      setSecondaryLoading(false);
    }
  };

  const fetchChartData = async () => {
    setLoading(true);
    setError(null);
    
    // Clear existing chart data while loading
    setChartData(null);
    // Note: We don't clear chartDataRef.current here because we might need it for highlighting
    
    try {
      let url = `${API_BASE_URL}/data/${selectedSymbol}/${selectedTimeframe}`;
      let queryParams = new URLSearchParams();

      if (fetchMode === FETCH_MODES.DATE_RANGE) {
        // Use date range parameters
        switch (dateRangeType) {
          case DATE_RANGE_TYPES.EARLIEST_TO_DATE:
            if (endDate) {
              queryParams.append('end_date', endDate);
            }
            break;
          case DATE_RANGE_TYPES.DATE_TO_DATE:
            if (startDate) {
              queryParams.append('start_date', startDate);
            }
            if (endDate) {
              queryParams.append('end_date', endDate);
            }
            break;
          case DATE_RANGE_TYPES.DATE_TO_LATEST:
            if (startDate) {
              queryParams.append('start_date', startDate);
            }
            break;
        }
        // Set a reasonable limit to avoid overwhelming the chart
        queryParams.append('limit', '10000');
      } else {
        // Use record limit mode
        let limit = rowLimit;
        
        // Adjust limit based on time range
        switch (timeRange) {
          case '1d':
            limit = selectedTimeframe.includes('m') ? 1440 : 24; // minutes in day or hours
            break;
          case '7d':
            limit = selectedTimeframe.includes('m') ? 10080 : 168; // 7 days worth
            break;
          case '30d':
            limit = selectedTimeframe.includes('m') ? 43200 : 720; // 30 days worth
            break;
          case '90d':
            limit = Math.min(90 * 24, 2000); // Cap at reasonable limit
            break;
          default:
            limit = rowLimit;
        }
        queryParams.append('limit', limit.toString());
      }

      queryParams.append('order', sortOrder || 'desc');
      queryParams.append('sort_by', 'timestamp');

      const response = await fetch(`${url}?${queryParams.toString()}`);
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const data = await response.json();
      console.log('📊 Chart data fetched successfully:', data.data?.length, 'candles');
      
      // Charts always need data in ascending time order for proper rendering
      // The sortOrder affects which data is fetched (newest vs oldest records)
      // but the chart must display them chronologically
      if (data.data && Array.isArray(data.data)) {
        const sortedData = [...data.data];
        sortedData.sort((a, b) => {
          const aTime = new Date(a.timestamp);
          const bTime = new Date(b.timestamp);
          return aTime - bTime; // Always ascending for chart display
        });
        
        data.data = sortedData;
        console.log('📊 Chart data sorted for display - Sort preference:', sortOrder, 'Data range:', 
          data.data[0]?.timestamp, 'to', data.data[data.data.length - 1]?.timestamp);
      }
      
      setChartData(data);
      setError(null);
      
      // Store chart data for highlighting before initialization
      if (data.data && Array.isArray(data.data)) {
        chartDataRef.current = data.data;
        console.log('📊 Stored chart data for highlighting:', data.data.length, 'candles');
      }
      
      // Chart will be initialized by the useEffect that watches chartData
    } catch (err) {
      setError(`Failed to fetch chart data: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  const initializeSecondaryChart = (data) => {
    if (!secondaryChartContainerRef.current || !data || data.length === 0) {
      console.warn('⚠️ Cannot initialize secondary chart - missing container or data');
      return;
    }

    console.log('🎯 Initializing secondary chart with', data.length, 'candles');

    // Store secondary chart data for correlation
    if (!secondaryChartDataRef.current) {
      secondaryChartDataRef.current = data;
    }

    // Clear any existing secondary chart
    if (secondaryChartRef.current) {
      console.log('🧹 Cleaning up existing secondary chart');
      secondaryChartRef.current.remove();
      secondaryChartRef.current = null;
      secondaryPriceSeriesRef.current = null;
    }

    // Clear the container
    if (secondaryChartContainerRef.current) {
      secondaryChartContainerRef.current.innerHTML = '';
    }

    try {
      // Ensure container has dimensions
      const containerWidth = secondaryChartContainerRef.current.clientWidth || 800;
      const containerHeight = 200; // Smaller height for secondary chart
      
      console.log('📐 Secondary container dimensions:', { width: containerWidth, height: containerHeight });
      
      // Create the secondary chart
      const chart = createChart(secondaryChartContainerRef.current, {
        width: containerWidth,
        height: containerHeight,
        layout: {
          background: { type: ColorType.Solid, color: isDarkMode ? '#1f2937' : '#ffffff' },
          textColor: isDarkMode ? '#e5e7eb' : '#374151',
        },
        grid: {
          vertLines: { color: isDarkMode ? '#374151' : '#e5e7eb' },
          horzLines: { color: isDarkMode ? '#374151' : '#e5e7eb' },
        },
        timeScale: {
          borderColor: isDarkMode ? '#6b7280' : '#d1d5db',
          timeVisible: true,
          secondsVisible: false,
        },
        rightPriceScale: {
          borderColor: isDarkMode ? '#6b7280' : '#d1d5db',
        },
        crosshair: {
          mode: CrosshairMode.Normal,
        },
      });

      secondaryChartRef.current = chart;

      // Prepare candlestick data with conditional orange overlay
      const candlestickData = data.map(candle => {
        const baseData = {
          time: Math.floor(new Date(candle.timestamp).getTime() / 1000),
          open: parseFloat(candle.open),
          high: parseFloat(candle.high),
          low: parseFloat(candle.low),
          close: parseFloat(candle.close),
        };

        // Apply orange overlay if toggle is enabled
        if (showBreakdownOverlay) {
          const isBullish = parseFloat(candle.close) >= parseFloat(candle.open);
          return {
            ...baseData,
            color: '#f59e0b', // Orange body for breakdown candles
            wickColor: '#f59e0b', // Orange wicks too
            borderColor: '#d97706' // Darker orange border
          };
        } else {
          // Use default colors when overlay is disabled
          const isBullish = parseFloat(candle.close) >= parseFloat(candle.open);
          return {
            ...baseData,
            color: isBullish ? '#26a69a' : '#ef5350',
            wickColor: isBullish ? '#26a69a' : '#ef5350',
            borderColor: isBullish ? '#26a69a' : '#ef5350'
          };
        }
      });

      // Add the secondary price series
      const priceSeries = chart.addSeries(CandlestickSeries, {
        upColor: showBreakdownOverlay ? '#f59e0b' : '#26a69a',
        downColor: showBreakdownOverlay ? '#f59e0b' : '#ef5350',
        borderVisible: true,
        wickUpColor: showBreakdownOverlay ? '#f59e0b' : '#26a69a',
        wickDownColor: showBreakdownOverlay ? '#f59e0b' : '#ef5350',
        borderUpColor: showBreakdownOverlay ? '#d97706' : '#26a69a',
        borderDownColor: showBreakdownOverlay ? '#d97706' : '#ef5350',
      });
      
      priceSeries.setData(candlestickData);
      secondaryPriceSeriesRef.current = priceSeries;

      // Fit content to show all data - but not in game mode or selection mode
      if (!gameMode && !isSelectionMode) {
        chart.timeScale().fitContent();
      }

      console.log('✅ Secondary chart initialized successfully');

      // Handle resize
      const handleResize = () => {
        const newWidth = secondaryChartContainerRef.current?.clientWidth || 800;
        chart.applyOptions({ width: newWidth });
      };

      window.addEventListener('resize', handleResize);
      
      // Store resize handler for cleanup
      chart._resizeHandler = handleResize;

    } catch (error) {
      console.error('❌ Error initializing secondary chart:', error);
    }
  };

  // Re-initialize secondary chart when overlay toggle changes
  useEffect(() => {
    if (showSecondaryChart && secondaryChartDataRef.current) {
      initializeSecondaryChart(secondaryChartDataRef.current);
    }
  }, [showBreakdownOverlay, isDarkMode]);

  const initializeChart = (data) => {
    if (!chartContainerRef.current || !data || data.length === 0) {
      console.warn('⚠️ Cannot initialize chart - missing container or data');
      return;
    }

    console.log('🎯 Initializing chart with', data.length, 'candles');

    // Ensure chart data is stored for highlighting (should already be set from fetchChartData)
    if (!chartDataRef.current) {
      chartDataRef.current = data;
      console.log('📊 Storing chart data in initializeChart as fallback');
    }

    // Clear any existing chart
    if (chartRef.current) {
      console.log('🧹 Cleaning up existing chart');
      setChartReady(false);
      chartRef.current.remove();
      chartRef.current = null;
      priceSeriesRef.current = null;
      timeScaleRef.current = null;
      // DON'T clear chartDataRef.current here - we need it for highlighting!
    }

    // Clear the container
    if (chartContainerRef.current) {
      chartContainerRef.current.innerHTML = '';
    }

    try {
      // Ensure container has dimensions
      const containerWidth = chartContainerRef.current.clientWidth || 800;
      const containerHeight = 400;
      
      console.log('📐 Container dimensions:', { width: containerWidth, height: containerHeight });
      
      // Create the chart using v5.0 API
      const chart = createChart(chartContainerRef.current, {
        width: containerWidth,
        height: containerHeight,
        layout: {
          background: { type: ColorType.Solid, color: isDarkMode ? '#1f2937' : '#ffffff' },
          textColor: isDarkMode ? '#e5e7eb' : '#374151',
        },
        grid: {
          vertLines: { color: isDarkMode ? '#374151' : '#f3f4f6' },
          horzLines: { color: isDarkMode ? '#374151' : '#f3f4f6' },
        },
        crosshair: {
          mode: 1, // CrosshairMode.Normal
          vertLine: {
            width: 1,
            color: isDarkMode ? '#60a5fa' : '#3b82f6',
            style: 2, // LineStyle.Dashed
            labelBackgroundColor: isDarkMode ? '#1e40af' : '#3b82f6',
          },
          horzLine: {
            width: 1,
            color: isDarkMode ? '#60a5fa' : '#3b82f6',
            style: 2, // LineStyle.Dashed
            labelBackgroundColor: isDarkMode ? '#1e40af' : '#3b82f6',
          },
        },
        rightPriceScale: {
          borderColor: isDarkMode ? '#4b5563' : '#d1d5db',
        },
        timeScale: {
          borderColor: isDarkMode ? '#4b5563' : '#d1d5db',
          timeVisible: true,
          secondsVisible: false,
        },
      });

      chartRef.current = chart;
      setTimeout(() => setChartReady(true), 0);
      timeScaleRef.current = chart.timeScale();

      // Get selected candles for current symbol/timeframe for highlighting
      const currentSymbolTimeframeSelection = selectedCandles.filter(c => 
        c.symbol === selectedSymbol && c.timeframe === selectedTimeframe
      );
      const selectedTimestamps = new Set(currentSymbolTimeframeSelection.map(c => c.timestamp));

      // Prepare candlestick data with highlighting
      const candlestickData = data.map(candle => {
        const isSelected = selectedTimestamps.has(candle.timestamp);
        const isBullish = parseFloat(candle.close) >= parseFloat(candle.open);
        const baseData = {
          time: Math.floor(new Date(candle.timestamp).getTime() / 1000),
          open: parseFloat(candle.open),
          high: parseFloat(candle.high),
          low: parseFloat(candle.low),
          close: parseFloat(candle.close),
        };

        // If selected, add blue highlighting colors but preserve original wick colors
        if (isSelected) {
          return {
            ...baseData,
            color: '#3b82f6', // Blue body
            wickColor: isBullish ? '#22c55e' : '#ef4444', // Keep original wick colors (green for bull, red for bear)
            borderColor: '#2563eb' // Darker blue border
          };
        }

        return baseData;
      });

      // Render the main price series based on selected chart type using v5.0 API
      let priceSeries;
      if (chartType === 'candlestick') {
        priceSeries = chart.addSeries(CandlestickSeries, {
          upColor: '#22c55e',
          downColor: '#ef4444',
          borderVisible: false,
          wickUpColor: '#22c55e',
          wickDownColor: '#ef4444',
        });
        priceSeries.setData(candlestickData);
      } else if (chartType === 'line') {
        priceSeries = chart.addSeries(LineSeries, {
          color: isDarkMode ? '#60a5fa' : '#1d4ed8',
          lineWidth: 2,
        });
        priceSeries.setData(candlestickData.map(d => ({ time: d.time, value: d.close })));
      } else {
        // area chart
        priceSeries = chart.addSeries(AreaSeries, {
          topColor: 'rgba(59,130,246,0.4)',
          bottomColor: 'rgba(59,130,246,0.0)',
          lineColor: isDarkMode ? '#60a5fa' : '#1d4ed8',
          lineWidth: 2,
        });
        priceSeries.setData(candlestickData.map(d => ({ time: d.time, value: d.close })));
      }
      console.log('🆕 Created new price series:', priceSeries);
      // Store price series reference for highlighting
      priceSeriesRef.current = priceSeries;

      // FVG markers are now handled via series markers instead of custom primitive
      console.log('🟢 FVG markers will be handled via series markers for proper layering');

      // Auto-fit content - but not in game mode or selection mode to avoid conflicts
      if (!gameMode && !isSelectionMode) {
        chart.timeScale().fitContent();
      }

      // Update selection statistics
      updateSelectionStats();
      
      // Update highlights for any existing selected candles
      setTimeout(() => {
        if (selectedCandles.length > 0) {
          updateChartHighlights();
        }
      }, 100);

      // Handle resize
      const handleResize = () => {
        if (chartContainerRef.current && chartRef.current) {
          const newWidth = chartContainerRef.current.clientWidth || 800;
          console.log('📏 Resizing chart to width:', newWidth);
          try {
            chartRef.current.applyOptions({ width: newWidth });
          } catch (error) {
            console.warn('⚠️ Could not resize chart (might be disposed):', error);
          }
        }
      };

      window.addEventListener('resize', handleResize);

      // Cleanup function
      return () => {
        window.removeEventListener('resize', handleResize);
        
        // FVG markers are now handled via series markers, no cleanup needed
        
        try {
          chart.remove();
        } catch (error) {
          console.warn('⚠️ Could not remove chart (might be already disposed):', error);
        }
      };

      console.log('🎉 Interactive chart created successfully:', data.length, 'candles');
      
    } catch (error) {
      console.error('❌ Error initializing chart:', error);
    }
  };

  const formatPrice = (price) => {
    return price ? price.toFixed(4) : 'N/A';
  };

  const formatTimestamp = (timestamp) => {
    return new Date(timestamp).toLocaleString();
  };

  const getLatestCandle = () => {
    if (!chartData?.data || chartData.data.length === 0) return null;
    return chartData.data[chartData.data.length - 1];
  };

  const getPriceChange = () => {
    if (!chartData?.data || chartData.data.length < 2) return { change: 0, percent: 0 };
    
    const latest = chartData.data[chartData.data.length - 1];
    const previous = chartData.data[chartData.data.length - 2];
    
    const change = latest.close - previous.close;
    const percent = (change / previous.close) * 100;
    
    return { change, percent };
  };

  const getMarketStats = () => {
    if (!chartData?.data || chartData.data.length === 0) return null;
    
    const prices = chartData.data.map(d => d.close);
    const high24h = Math.max(...chartData.data.map(d => d.high));
    const low24h = Math.min(...chartData.data.map(d => d.low));
    const volume24h = chartData.data.reduce((sum, d) => sum + (d.volume || 0), 0);
    
    return {
      high24h,
      low24h,
      volume24h,
      dataPoints: chartData.data.length
    };
  };

  // Enhanced selection functions
  const toggleSelectionMode = () => {
    setIsSelectionMode(!isSelectionMode);
    setIsDragging(false);
    setDragStart(null);
    setDragEnd(null);
    setHoveredCandle(null);
    
    // Reset crosshair state when exiting selection mode
    if (isSelectionMode) {
      setShowCustomCrosshair(false);
      setCrosshairPosition(null);
    }
  };
  
  const handleKeyDown = (e) => {
    if (!isSelectionMode) return;
    
    // Escape key - exit selection mode
    if (e.key === 'Escape') {
      toggleSelectionMode();
    }
    
    // Delete key - clear selection
    if (e.key === 'Delete' && selectedCandles.length > 0) {
      // Clear only current symbol/timeframe selection
      const toRemove = selectedCandles.filter(c => 
        c.symbol === selectedSymbol && c.timeframe === selectedTimeframe
      );
      toRemove.forEach(c => removeSelectedCandle(c.id));
    }
  };
  
  // Add keyboard event listeners
  useEffect(() => {
    if (isSelectionMode) {
      window.addEventListener('keydown', handleKeyDown);
      
      return () => {
        window.removeEventListener('keydown', handleKeyDown);
      };
    }
  }, [isSelectionMode]);

  const getCandleFromPosition = (x, containerRect) => {
    if (!chartData?.data || !chartRef.current || !timeScaleRef.current) return null;
    
    try {
      // Convert screen X coordinate to logical coordinate
      const timeScale = timeScaleRef.current;
      const logical = timeScale.coordinateToLogical(x - containerRect.left);
      
      if (logical === null) return null;
      
      // Find the closest candle to this logical index
      const candleIndex = Math.round(logical);
      
      if (candleIndex >= 0 && candleIndex < chartData.data.length) {
        return {
          index: candleIndex,
          candle: chartData.data[candleIndex]
        };
      }
    } catch (error) {
      console.error('Error getting candle from position:', error);
    }
    
    return null;
  };

  const selectCandle = (candle, index) => {
    if (!candle) return;
    
    const candleId = `${selectedSymbol}_${selectedTimeframe}_${candle.timestamp}`;
    
    // Check if already selected
    const isSelected = selectedCandles.some(c => c.id === candleId);
    
    if (isSelected) {
      removeSelectedCandle(candleId);
    } else {
      const candleWithMetadata = {
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
      };
      
      addSelectedCandle(candleWithMetadata);
    }
  };

  const selectCandlesInRange = (startX, endX, containerRect) => {
    if (!chartData?.data) return;
    
    const minX = Math.min(startX, endX);
    const maxX = Math.max(startX, endX);
    
    const startCandle = getCandleFromPosition(minX, containerRect);
    const endCandle = getCandleFromPosition(maxX, containerRect);
    
    if (!startCandle || !endCandle) return;
    
    const startIndex = Math.min(startCandle.index, endCandle.index);
    const endIndex = Math.max(startCandle.index, endCandle.index);
    
    // Select all candles in range
    for (let i = startIndex; i <= endIndex; i++) {
      if (i >= 0 && i < chartData.data.length) {
        const candle = chartData.data[i];
        const candleId = `${selectedSymbol}_${selectedTimeframe}_${candle.timestamp}`;
        
        // Only add if not already selected
        if (!selectedCandles.some(c => c.id === candleId)) {
          selectCandle(candle, i);
        }
      }
    }
  };

  const handleOverlayMouseDown = (e) => {
    if (!isSelectionMode) return;
    
    e.preventDefault();
    setIsDragging(true);
    setDragStart({ x: e.clientX, y: e.clientY });
    setDragEnd({ x: e.clientX, y: e.clientY });
  };

  const handleOverlayMouseMove = (e) => {
    if (!isSelectionMode) return;
    
    const containerRect = chartContainerRef.current?.getBoundingClientRect();
    if (!containerRect) return;
    
    const mouseX = e.clientX - containerRect.left;
    const mouseY = e.clientY - containerRect.top;
    
    // Update crosshair position for selection mode
    if (enableSelectionCrosshair) {
      setCrosshairPosition({ x: mouseX, y: mouseY });
      setShowCustomCrosshair(true);
    }
    
    // Update hover info
    const candleData = getCandleFromPosition(e.clientX, containerRect);
    if (candleData) {
      setHoveredCandle({
        index: candleData.index,
        candle: candleData.candle,
        position: { x: mouseX, y: mouseY }
      });
    } else {
      setHoveredCandle(null);
    }
    
    // Update drag selection
    if (isDragging) {
      setDragEnd({ x: e.clientX, y: e.clientY });
    }
  };

  const handleOverlayMouseUp = (e) => {
    if (!isSelectionMode || !isDragging) return;
    
    const containerRect = chartContainerRef.current?.getBoundingClientRect();
    if (!containerRect || !dragStart) return;
    
    const distance = Math.abs(e.clientX - dragStart.x);
    
    if (distance < 5) {
      // Single click - select individual candle
      const candleData = getCandleFromPosition(e.clientX, containerRect);
      if (candleData) {
        selectCandle(candleData.candle, candleData.index);
      }
    } else {
      // Drag selection - select range of candles
      selectCandlesInRange(dragStart.x, e.clientX, containerRect);
    }
    
    setIsDragging(false);
    setDragStart(null);
    setDragEnd(null);
  };

  const getSelectionBoxStyle = () => {
    if (!isDragging || !dragStart || !dragEnd) return { display: 'none' };
    
    const containerRect = chartContainerRef.current?.getBoundingClientRect();
    if (!containerRect) return { display: 'none' };
    
    const left = Math.min(dragStart.x, dragEnd.x) - containerRect.left;
    const top = 0;
    const width = Math.abs(dragEnd.x - dragStart.x);
    const height = containerRect.height;
    
    return {
      position: 'absolute',
      left: `${left}px`,
      top: `${top}px`,
      width: `${width}px`,
      height: `${height}px`,
      backgroundColor: 'rgba(59, 130, 246, 0.1)',
      border: '2px solid rgba(59, 130, 246, 0.5)',
      borderRadius: '4px',
      pointerEvents: 'none',
      zIndex: 10
    };
  };
  
  const selectTimeRange = (range) => {
    if (!chartData?.data || chartData.data.length === 0) return;
    
    // Get the latest timestamp
    const latestTime = new Date(chartData.data[chartData.data.length - 1].timestamp).getTime();
    
    // Calculate the time threshold based on range
    let hoursBack = 1;
    switch (range) {
      case '1h': hoursBack = 1; break;
      case '4h': hoursBack = 4; break;
      case '1d': hoursBack = 24; break;
      default: hoursBack = 1;
    }
    
    const threshold = latestTime - (hoursBack * 60 * 60 * 1000);
    
    // Select all candles within the time range
    chartData.data.forEach((candle, index) => {
      const candleTime = new Date(candle.timestamp).getTime();
      if (candleTime >= threshold) {
        const candleId = `${selectedSymbol}_${selectedTimeframe}_${candle.timestamp}`;
        if (!selectedCandles.some(c => c.id === candleId)) {
          selectCandle(candle, index);
        }
      }
    });
    

  };

  const [iconsReady, setIconsReady] = useState(false);

  // Wait for both green 'Available' icons to appear in the DOM
  useEffect(() => {
    setIconsReady(false);
    const checkIcons = () => {
      const tjrIcon = Array.from(document.querySelectorAll('span.bg-green-100.text-green-800')).find(
        el => el.textContent && el.textContent.includes('Available') && el.closest('h3')?.textContent?.includes('TJR')
      );
      const fvgIcon = Array.from(document.querySelectorAll('span.bg-green-100.text-green-800')).find(
        el => el.textContent && el.textContent.includes('Available') && el.closest('h3')?.textContent?.includes('FVG')
      );
      if (tjrIcon && fvgIcon) {
        setIconsReady(true);
        return true;
      }
      return false;
    };
    // Poll for icons every 200ms until both are present
    const interval = setInterval(() => {
      if (checkIcons()) clearInterval(interval);
    }, 200);
    // Also check immediately in case already present
    checkIcons();
    return () => clearInterval(interval);
  }, [tables, isDarkMode]);

  // Check if game mode is active by monitoring the container style
  useEffect(() => {
    if (!chartContainerRef.current) return;
    
    const observer = new MutationObserver(() => {
      const container = chartContainerRef.current;
      if (container) {
        const isGameMode = container.style.position === 'fixed' && 
                          container.style.zIndex === '10000';
        setGameMode(isGameMode);
      }
    });
    
    observer.observe(chartContainerRef.current, {
      attributes: true,
      attributeFilter: ['style']
    });
    
    return () => observer.disconnect();
  }, [chartContainerRef.current]);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className={`animate-spin rounded-full h-12 w-12 border-b-2 ${
          isDarkMode ? 'border-blue-400' : 'border-blue-600'
        }`}></div>
        <span className={`ml-3 ${
          isDarkMode ? 'text-gray-300' : 'text-gray-600'
        }`}>Loading chart data...</span>
      </div>
    );
  }

  if (error) {
    return (
      <div className={`rounded-lg p-4 border transition-colors duration-200 ${
        isDarkMode 
          ? 'bg-red-900/20 border-red-800' 
          : 'bg-red-50 border-red-200'
      }`}>
        <div className="flex items-center">
          <div className="text-red-500 text-sm font-medium">Error loading chart data</div>
        </div>
        <div className="text-red-500 text-sm mt-1">{error}</div>
        <button 
          onClick={fetchChartData}
          className="mt-2 px-3 py-1 bg-red-600 text-white rounded text-sm hover:bg-red-700 transition-colors duration-200"
        >
          Retry
        </button>
      </div>
    );
  }

  const latestCandle = getLatestCandle();
  const priceChange = getPriceChange();
  const marketStats = getMarketStats();

  return (
    <div className="space-y-6">
      {/* Chart Dashboard Header */}
      <div className={`p-6 rounded-lg transition-colors duration-200 ${
        isDarkMode 
          ? 'bg-gradient-to-r from-green-800 via-green-700 to-blue-800 text-white' 
          : 'bg-gradient-to-r from-green-600 to-blue-600 text-white'
      }`}>
        <div className="flex items-center">
          <h2 className="text-3xl font-black tracking-tight mb-2" style={{ 
            fontFamily: 'system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
            letterSpacing: '-0.02em',
            textShadow: '0 2px 4px rgba(0,0,0,0.3)'
          }}>
            Day<span className="text-green-200">gent</span> <span className="text-xl font-semibold text-green-100">Chart Analysis</span>
          </h2>
        <InfoTooltip id="chart-dashboard" content={
          <div>
            <p className="font-semibold mb-2">📈 Chart Dashboard Overview</p>
            <p className="mb-2">Professional trading charts with advanced visualization:</p>
            <ul className="list-disc list-inside space-y-1 text-xs">
              <li><strong>Interactive Charts:</strong> Candlestick, line, and area charts</li>
              <li><strong>Technical Indicators:</strong> SMA, EMA, Bollinger Bands, Volume</li>
              <li><strong>Real-time Data:</strong> Live price updates and market stats</li>
              <li><strong>Multiple Timeframes:</strong> From 1-minute to daily charts</li>
            </ul>
            <p className="mt-2 text-xs">Perfect for technical analysis, trend identification, and trading decisions.</p>
          </div>
        } isDarkMode={isDarkMode} />
        </div>
                  <p className={isDarkMode ? 'text-green-200' : 'text-green-100'}>
            Interactive charting & technical analysis • Professional trading tools
          </p>
        <div className={`mt-4 text-sm rounded p-2 ${
          isDarkMode ? 'bg-black/30' : 'bg-white/20'
        }`}>
          📈 {selectedSymbol?.toUpperCase()} • {selectedTimeframe} • {chartData?.data?.length || 0} candles loaded
          {(() => {
            const currentSymbolTimeframeSelection = selectedCandles.filter(c => 
              c.symbol === selectedSymbol && c.timeframe === selectedTimeframe
            );
            return currentSymbolTimeframeSelection.length > 0 && (
              <span className="ml-4 px-2 py-1 rounded text-xs bg-blue-600 text-white">
                🔵 {currentSymbolTimeframeSelection.length} highlighted
              </span>
            );
          })()}
        </div>
      </div>

      {/* Data Selection & Controls - Using Consistent Component */}
      <DataSelectionControls 
        handleRefresh={fetchChartData}
        isDarkMode={isDarkMode}
        dashboardType="chart"
      />

      {/* Chart Container */}
      <div className={`rounded-lg shadow-md transition-colors duration-200 ${
        isDarkMode ? 'bg-gray-800' : 'bg-white'
      }`} style={{ position: 'relative' }}>
        {/* LOADING OVERLAY for chart+game icon */}
        {!iconsReady && (
          <div style={{
            position: 'absolute',
            zIndex: 50,
            top: 0,
            left: 0,
            width: '100%',
            height: '100%',
            background: isDarkMode ? 'rgba(17,24,39,0.85)' : 'rgba(255,255,255,0.85)',
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            pointerEvents: 'all',
            borderRadius: 'inherit',
          }}>
            <div className={`animate-spin rounded-full h-16 w-16 border-b-4 ${isDarkMode ? 'border-blue-400' : 'border-blue-600'}`}></div>
            <span className={`mt-4 text-lg font-semibold ${isDarkMode ? 'text-blue-200' : 'text-blue-700'}`}>Loading chart & game...</span>
            <span className={`mt-2 text-sm ${isDarkMode ? 'text-gray-400' : 'text-gray-600'}`}>Waiting for TJR & FVG data availability</span>
          </div>
        )}
        <div className={`px-6 py-4 border-b flex items-center justify-between transition-colors duration-200 ${
          isDarkMode ? 'border-gray-700' : 'border-gray-200'
        }`}>
          <div className="flex items-center">
            <h3 className={`text-lg font-semibold ${
              isDarkMode ? 'text-white' : 'text-gray-900'
            }`}>Price Chart</h3>
            {/* Game mode toggle button */}
            {chartReady && (
              <GameModeController chartRef={chartRef} containerRef={chartContainerRef} isDarkMode={isDarkMode} />
            )}
            <InfoTooltip id="price-chart" content={
              <div>
                <p className="font-semibold mb-2">📈 Interactive Price Chart</p>
                <p className="mb-2">Professional trading chart with the following features:</p>
                <ul className="list-disc list-inside space-y-1 text-xs">
                  <li><strong>Candlestick View:</strong> OHLC data visualization</li>
                  <li><strong>Line Chart:</strong> Clean price trend visualization</li>
                  <li><strong>Area Chart:</strong> Volume-weighted visual representation</li>
                  <li><strong>Zoom & Pan:</strong> Navigate through time periods</li>
                  <li><strong>Real-time Updates:</strong> Live data when available</li>
                </ul>
                <p className="mt-2 text-xs">Chart powered by lightweight-charts library for optimal performance.</p>
              </div>
            } isDarkMode={isDarkMode} />
          </div>
          
          <div className="flex items-center gap-3">
            {/* Chart Type Selector */}
            <div className="flex items-center gap-2">
              <label className={`text-sm font-medium ${
                isDarkMode ? 'text-gray-300' : 'text-gray-700'
              }`}>
                Chart Type:
              </label>
              <select
                value={chartType}
                onChange={(e) => setChartType(e.target.value)}
                className={`px-3 py-1 text-sm rounded-md border transition-colors duration-200 ${
                  isDarkMode 
                    ? 'border-gray-600 bg-gray-700 text-white' 
                    : 'border-gray-300 bg-white text-gray-900'
                }`}
              >
                <option value="candlestick">🕯️ Candlestick</option>
                <option value="line">📈 Line Chart</option>
                <option value="area">📊 Area Chart</option>
              </select>
            </div>

            {/* Time Range Quick Selection - Only visible in selection mode */}
            {isSelectionMode && (
              <div className="flex items-center gap-1">
                <span className={`text-xs ${isDarkMode ? 'text-gray-400' : 'text-gray-600'}`}>Quick Select:</span>
                <button
                  onClick={() => selectTimeRange('1h')}
                  className={`px-2 py-1 text-xs rounded ${
                    isDarkMode
                      ? 'bg-gray-700 hover:bg-gray-600 text-gray-300'
                      : 'bg-gray-100 hover:bg-gray-200 text-gray-700'
                  }`}
                >
                  1H
                </button>
                <button
                  onClick={() => selectTimeRange('4h')}
                  className={`px-2 py-1 text-xs rounded ${
                    isDarkMode
                      ? 'bg-gray-700 hover:bg-gray-600 text-gray-300'
                      : 'bg-gray-100 hover:bg-gray-200 text-gray-700'
                  }`}
                >
                  4H
                </button>
                <button
                  onClick={() => selectTimeRange('1d')}
                  className={`px-2 py-1 text-xs rounded ${
                    isDarkMode
                      ? 'bg-gray-700 hover:bg-gray-600 text-gray-300'
                      : 'bg-gray-100 hover:bg-gray-200 text-gray-700'
                  }`}
                >
                  1D
                </button>
              </div>
            )}

            {/* Multi-Timeframe Toggle - Only show when sequential candles are selected */}
            {hasSequentialSelection && (
              <>
                <button
                  onClick={() => setShowSecondaryChart(!showSecondaryChart)}
                  className={`px-3 py-2 rounded-md text-sm font-medium transition-all duration-200 flex items-center gap-2 ${
                    showSecondaryChart
                      ? isDarkMode
                        ? 'bg-green-600 text-white'
                        : 'bg-green-600 text-white'
                      : isDarkMode
                        ? 'bg-gray-700 text-gray-400 hover:bg-gray-600'
                        : 'bg-gray-100 text-gray-500 hover:bg-gray-200'
                  }`}
                                     title="Show breakdown of selected candle(s) in lower timeframe"
                >
                  <span>📊</span>
                  <span>{showSecondaryChart ? 'Breakdown ON' : 'Show Breakdown'}</span>
                </button>

                {/* Secondary Timeframe Selector - Only visible when secondary chart is enabled */}
                {showSecondaryChart && (
                  <div className="flex items-center gap-2">
                    <label className={`text-sm font-medium ${
                      isDarkMode ? 'text-gray-300' : 'text-gray-700'
                    }`}>
                      Breakdown:
                    </label>
                    <select
                      value={secondaryTimeframe}
                      onChange={(e) => setSecondaryTimeframe(e.target.value)}
                      className={`px-3 py-1 text-sm rounded-md border transition-colors duration-200 ${
                        isDarkMode 
                          ? 'border-gray-600 bg-gray-700 text-white' 
                          : 'border-gray-300 bg-white text-gray-900'
                      }`}
                    >
                      <option value="1m">1 Minute</option>
                      <option value="5m">5 Minutes</option>
                      <option value="15m">15 Minutes</option>
                      <option value="30m">30 Minutes</option>
                      <option value="1h">1 Hour</option>
                      <option value="4h">4 Hours</option>
                    </select>
                    
                    {/* Compatibility Warning */}
                    {!areTimeframesCompatible(selectedTimeframe, secondaryTimeframe) && (
                      <span className={`text-xs px-2 py-1 rounded ${
                        isDarkMode ? 'bg-orange-900 text-orange-300' : 'bg-orange-100 text-orange-700'
                      }`}>
                        ⚠️ Incompatible
                      </span>
                    )}
                  </div>
                )}
              </>
            )}

            {/* Crosshair Toggle - Only visible in selection mode */}
            {isSelectionMode && (
              <button
                onClick={() => setEnableSelectionCrosshair(!enableSelectionCrosshair)}
                className={`px-3 py-2 rounded-md text-sm font-medium transition-all duration-200 flex items-center gap-2 ${
                  enableSelectionCrosshair
                    ? isDarkMode
                      ? 'bg-blue-600 text-white'
                      : 'bg-blue-600 text-white'
                    : isDarkMode
                      ? 'bg-gray-700 text-gray-400 hover:bg-gray-600'
                      : 'bg-gray-100 text-gray-500 hover:bg-gray-200'
                }`}
                title="Toggle crosshair in selection mode"
              >
                <span>✛</span>
                <span>{enableSelectionCrosshair ? 'Crosshair ON' : 'Crosshair OFF'}</span>
              </button>
            )}

            {/* Selection Mode Toggle */}
            <div className="flex items-center space-x-2">
              <button
                onClick={toggleSelectionMode}
                className={`flex items-center px-3 py-2 rounded-lg transition-all duration-200 ${
                  isSelectionMode
                    ? 'bg-blue-500 text-white shadow-lg transform scale-105'
                    : 'bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-300 dark:hover:bg-gray-600'
                }`}
                title="Toggle selection mode"
              >
                {/* Modern Crosshair Icon */}
                <svg 
                  width="18" 
                  height="18" 
                  viewBox="0 0 24 24" 
                  fill="none" 
                  stroke="currentColor" 
                  strokeWidth="2.5" 
                  strokeLinecap="round" 
                  strokeLinejoin="round"
                  className="mr-2"
                >
                  {/* Crosshair lines */}
                  <line x1="12" y1="2" x2="12" y2="22"></line>
                  <line x1="2" y1="12" x2="22" y2="12"></line>
                  {/* Center circle */}
                  <circle cx="12" cy="12" r="3" fill="none"></circle>
                  {/* Corner indicators */}
                  <path d="M7 7L5 5M17 7L19 5M7 17L5 19M17 17L19 19" strokeWidth="2"></path>
                </svg>
                Select Range
              </button>

              {/* Breakdown overlay toggle - only show when in breakdown mode */}
              {showSecondaryChart && (
                <button
                  onClick={() => setShowBreakdownOverlay(!showBreakdownOverlay)}
                  className={`flex items-center px-3 py-2 rounded-lg transition-all duration-200 ${
                    showBreakdownOverlay
                      ? 'bg-orange-500 text-white shadow-lg'
                      : 'bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-300 dark:hover:bg-gray-600'
                  }`}
                  title="Toggle breakdown overlay highlighting"
                >
                  <svg 
                    width="16" 
                    height="16" 
                    viewBox="0 0 24 24" 
                    fill="none" 
                    stroke="currentColor" 
                    strokeWidth="2" 
                    strokeLinecap="round" 
                    strokeLinejoin="round"
                    className="mr-2"
                  >
                    {showBreakdownOverlay ? (
                      <>
                        <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"></path>
                        <circle cx="12" cy="12" r="3"></circle>
                      </>
                    ) : (
                      <>
                        <path d="M17.94 17.94A10.07 10.07 0 0 1 12 20c-7 0-11-8-11-8a18.45 18.45 0 0 1 5.06-5.94M9.9 4.24A9.12 9.12 0 0 1 12 4c7 0 11 8 11 8a18.5 18.5 0 0 1-2.16 3.19m-6.72-1.07a3 3 0 1 1-4.24-4.24"></path>
                        <line x1="1" y1="1" x2="23" y2="23"></line>
                      </>
                    )}
                  </svg>
                  {showBreakdownOverlay ? 'Hide' : 'Show'} Overlay
                </button>
              )}
            </div>
          </div>
        </div>

        {/* Chart Area */}
        <div className="p-6">
          {/* Enhanced Selection Instructions */}
          {isSelectionMode && (
            <div className="space-y-3 mb-4">
              {/* Selection Mode Indicator */}
              <div className={`p-3 rounded-lg border transition-colors duration-200 ${
                isDarkMode ? 'bg-blue-900/20 border-blue-800 text-blue-300' : 'bg-blue-50 border-blue-200 text-blue-700'
              }`}>
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center">
                    <span className="text-lg mr-2">🎯</span>
                    <span className="font-medium">Selection Mode: CLICK</span>
                  </div>
                  <div className="flex gap-2 text-xs">
                    <span className="px-2 py-1 rounded bg-blue-600 text-white">
                      Click
                    </span>
                  </div>
                </div>
                <div className="text-sm grid grid-cols-2 gap-2">
                  <div>
                    <p>• <strong>Click:</strong> Select/deselect individual candles</p>
                  </div>
                  <div>
                    <p>• <strong>Delete:</strong> Clear current selection</p>
                    <p>• <strong>Escape:</strong> Exit selection mode</p>
                    <p>• <strong>Hover:</strong> Preview candle data</p>
                  </div>
                </div>
              </div>
              
              {/* Selection Statistics */}
              {selectionStats && (
                <div className={`p-3 rounded-lg border transition-colors duration-200 ${
                  isDarkMode ? 'bg-gray-800 border-gray-700' : 'bg-gray-50 border-gray-200'
                }`}>
                  <div className="flex items-center mb-2">
                    <span className="text-lg mr-2">📊</span>
                    <span className="font-medium">Selection Statistics</span>
                  </div>
                  <div className="grid grid-cols-4 gap-3 text-sm">
                    <div>
                      <span className={isDarkMode ? 'text-gray-400' : 'text-gray-600'}>Count:</span>
                      <span className="ml-1 font-medium">{selectionStats.count} candles</span>
                    </div>
                    <div>
                      <span className={isDarkMode ? 'text-gray-400' : 'text-gray-600'}>Time Range:</span>
                      <span className="ml-1 font-medium">{selectionStats.timeRange}</span>
                    </div>
                    <div>
                      <span className={isDarkMode ? 'text-gray-400' : 'text-gray-600'}>Avg Price:</span>
                      <span className="ml-1 font-medium">${selectionStats.avgPrice.toFixed(2)}</span>
                    </div>
                    <div>
                      <span className={isDarkMode ? 'text-gray-400' : 'text-gray-600'}>Trend:</span>
                      <span className="ml-1">
                        <span className="text-green-500">↑{selectionStats.bullish}</span>
                        <span className="mx-1">/</span>
                        <span className="text-red-500">↓{selectionStats.bearish}</span>
                      </span>
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}
          
          <div 
            ref={chartContainerRef}
            className={`w-full h-96 rounded border transition-colors duration-200 ${
              isDarkMode 
                ? 'border-gray-600 bg-gray-900' 
                : 'border-gray-200 bg-gray-50'
            }`}
            style={{ minHeight: '400px', position: 'relative' }}
          >
            {/* Chart is rendered programmatically via lightweight-charts */}
            
            {/* Selection Overlay */}
            {/* Selection Overlay - disabled in game mode to avoid conflicts */}
            {isSelectionMode && !gameMode && (
              <div
                ref={selectionOverlayRef}
                className="absolute inset-0 z-20"
                style={{ 
                  cursor: isSelectionMode ? 'crosshair' : 'default',
                  backgroundColor: 'transparent'
                }}
                onMouseDown={handleOverlayMouseDown}
                onMouseMove={handleOverlayMouseMove}
                onMouseUp={handleOverlayMouseUp}
                onMouseLeave={() => {
                  setHoveredCandle(null);
                  setShowCustomCrosshair(false);
                  setCrosshairPosition(null);
                  if (isDragging) {
                    setIsDragging(false);
                    setDragStart(null);
                    setDragEnd(null);
                  }
                }}
              >
                {/* Selection Rectangle */}
                {isDragging && (
                  <div
                    className="absolute"
                    style={getSelectionBoxStyle()}
                  />
                )}
                
                {/* Custom Crosshair for Selection Mode */}
                {showCustomCrosshair && crosshairPosition && (
                  <>
                    {/* Vertical line */}
                    <div
                      className="absolute pointer-events-none"
                      style={{
                        left: `${crosshairPosition.x}px`,
                        top: '0px',
                        width: '1px',
                        height: '100%',
                        borderLeft: `1px dashed ${isDarkMode ? '#60a5fa' : '#3b82f6'}`,
                        opacity: 0.8,
                        zIndex: 5
                      }}
                    />
                    {/* Horizontal line */}
                    <div
                      className="absolute pointer-events-none"
                      style={{
                        left: '0px',
                        top: `${crosshairPosition.y}px`,
                        width: '100%',
                        height: '1px',
                        borderTop: `1px dashed ${isDarkMode ? '#60a5fa' : '#3b82f6'}`,
                        opacity: 0.8,
                        zIndex: 5
                      }}
                    />
                  </>
                )}
                
                {/* Hover Tooltip */}
                {hoveredCandle && (
                  <div 
                    className={`absolute z-30 pointer-events-none px-2 py-1 rounded text-xs ${
                      isDarkMode 
                        ? 'bg-gray-900/90 text-gray-200 border border-gray-600' 
                        : 'bg-white/90 text-gray-800 border border-gray-300'
                    }`}
                    style={{
                      left: `${hoveredCandle.position.x + 10}px`,
                      top: `${hoveredCandle.position.y - 30}px`,
                      transform: 'translateY(-100%)',
                    }}
                  >
                    <div className="font-medium">{formatTimestamp(hoveredCandle.candle.timestamp)}</div>
                    <div>O: ${hoveredCandle.candle.open.toFixed(2)}</div>
                    <div>H: ${hoveredCandle.candle.high.toFixed(2)}</div>
                    <div>L: ${hoveredCandle.candle.low.toFixed(2)}</div>
                    <div>C: ${hoveredCandle.candle.close.toFixed(2)}</div>
                    <div className={hoveredCandle.candle.close > hoveredCandle.candle.open ? 'text-green-500' : 'text-red-500'}>
                      {hoveredCandle.candle.close > hoveredCandle.candle.open ? '↑' : '↓'} 
                      ${Math.abs(hoveredCandle.candle.close - hoveredCandle.candle.open).toFixed(2)}
                    </div>
                  </div>
                )}
                
                {/* Selection Mode Indicator */}
                <div className={`absolute top-2 left-2 px-3 py-1 rounded text-xs font-medium ${
                  isDarkMode 
                    ? 'bg-blue-900/80 text-blue-200 border border-blue-600' 
                    : 'bg-blue-100/80 text-blue-700 border border-blue-300'
                }`}>
                  🎯 Click Mode
                </div>
                
                {/* Instructions */}
                <div className={`absolute top-2 right-2 px-2 py-1 rounded text-xs ${
                  isDarkMode 
                    ? 'bg-gray-900/80 text-gray-300 border border-gray-600' 
                    : 'bg-white/80 text-gray-600 border border-gray-300'
                }`}>
                  {isDragging ? 'Drag to select multiple candles' : 'Click or drag to select candles'}
                </div>
              </div>
            )}
          </div>

          {!chartData?.data && (
            <div className="flex items-center justify-center h-96 -mt-96">
              <div className="text-center">
                <div className="text-4xl mb-4">📊</div>
                <h3 className={`text-lg font-medium mb-2 ${
                  isDarkMode ? 'text-gray-300' : 'text-gray-900'
                }`}>No Chart Data</h3>
                <p className={`${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>
                  Select a symbol and timeframe to load chart data
                </p>
              </div>
            </div>
          )}
        </div>
      </div>

            {/* Secondary Chart Container - Only show when sequential candles are selected */}
      {showSecondaryChart && hasSequentialSelection && (
        <div className={`rounded-lg shadow-md transition-colors duration-200 ${
          isDarkMode ? 'bg-gray-800' : 'bg-white'
        }`}>
          <div className={`px-6 py-4 border-b flex items-center justify-between transition-colors duration-200 ${
            isDarkMode ? 'border-gray-700' : 'border-gray-200'
          }`}>
            <div className="flex items-center">
              <h3 className={`text-lg font-semibold ${
                isDarkMode ? 'text-white' : 'text-gray-900'
              }`}>
                Breakdown Chart ({secondaryTimeframe})
              </h3>
              <InfoTooltip id="secondary-chart" content={
                <div>
                                       <p className="font-semibold mb-2">📊 Candle Breakdown Analysis</p>
                     <p className="mb-2">Shows the lower timeframe candles that make up your selected candle(s):</p>
                     <ul className="list-disc list-inside space-y-1 text-xs">
                       <li><strong>Single or Sequential:</strong> Select 1+ consecutive candles on the main chart</li>
                       <li><strong>Breakdown Display:</strong> See all the smaller timeframe candles that compose them</li>
                       <li><strong>Timeframe Rules:</strong> Breakdown timeframe must be smaller than primary</li>
                       <li><strong>Compatible Ratios:</strong> Primary timeframe must be evenly divisible by breakdown timeframe</li>
                     </ul>
                     <p className="mt-2 text-xs">Perfect for detailed analysis of how larger candles are formed from smaller ones.</p>
                </div>
              } isDarkMode={isDarkMode} />
            </div>
            
            <div className="flex items-center gap-3">
              {/* Selection Status */}
              {(() => {
                const selectedSequentialCandles = selectedCandles.filter(c => 
                  c.symbol === selectedSymbol && c.timeframe === selectedTimeframe
                );
                return selectedSequentialCandles.length > 0 && (
                  <div className={`text-sm px-3 py-1 rounded-full ${
                    isDarkMode ? 'bg-blue-900 text-blue-300' : 'bg-blue-100 text-blue-700'
                  }`}>
                    📊 {selectedSequentialCandles.length} selected candles → {correlatedCandles.length} breakdown candles
                  </div>
                );
              })()}
              
              {/* Compatibility Status */}
              {areTimeframesCompatible(selectedTimeframe, secondaryTimeframe) ? (
                <div className={`text-sm px-3 py-1 rounded-full ${
                  isDarkMode ? 'bg-green-900 text-green-300' : 'bg-green-100 text-green-700'
                }`}>
                  ✅ Compatible
                </div>
              ) : (
                <div className={`text-sm px-3 py-1 rounded-full ${
                  isDarkMode ? 'bg-red-900 text-red-300' : 'bg-red-100 text-red-700'
                }`}>
                  ❌ Incompatible
                </div>
              )}
            </div>
          </div>
          
                     <div className="p-6">
             {/* Sequential Selection Analysis Panel */}
             {correlatedCandles.length > 0 && areTimeframesCompatible(selectedTimeframe, secondaryTimeframe) && (() => {
               const selectedSequentialCandles = selectedCandles.filter(c => 
                 c.symbol === selectedSymbol && c.timeframe === selectedTimeframe
               ).sort((a, b) => a.sourceIndex - b.sourceIndex);
               
               return selectedSequentialCandles.length > 0 && (
                 <div className={`mb-4 p-4 rounded-lg border transition-colors duration-200 ${
                   isDarkMode ? 'bg-blue-900/20 border-blue-800' : 'bg-blue-50 border-blue-200'
                 }`}>
                   <div className="flex items-center justify-between mb-3">
                                            <h4 className={`font-semibold ${
                         isDarkMode ? 'text-blue-300' : 'text-blue-700'
                       }`}>
                         🔗 Candle Breakdown Analysis
                       </h4>
                     <button
                       onClick={() => {
                         setShowSecondaryChart(false);
                         setCorrelatedCandles([]);
                       }}
                       className={`text-xs px-2 py-1 rounded ${
                         isDarkMode 
                           ? 'bg-blue-800 text-blue-200 hover:bg-blue-700' 
                           : 'bg-blue-200 text-blue-800 hover:bg-blue-300'
                       }`}
                     >
                       Close
                     </button>
                   </div>
                   
                   <div className="grid grid-cols-2 gap-4 text-sm">
                     <div>
                                                <div className={`font-medium mb-2 ${
                           isDarkMode ? 'text-blue-300' : 'text-blue-700'
                         }`}>
                           Selected Candle(s) ({selectedTimeframe})
                         </div>
                       <div className={`space-y-1 ${
                         isDarkMode ? 'text-blue-200' : 'text-blue-600'
                       }`}>
                         <div>🔢 Count: {selectedSequentialCandles.length} candles</div>
                         <div>⏰ From: {formatTimestamp(selectedSequentialCandles[0].timestamp)}</div>
                         <div>⏰ To: {formatTimestamp(selectedSequentialCandles[selectedSequentialCandles.length - 1].timestamp)}</div>
                         <div>
                           📊 Combined Range: {Math.min(...selectedSequentialCandles.map(c => c.low)).toFixed(2)} - {Math.max(...selectedSequentialCandles.map(c => c.high)).toFixed(2)}
                         </div>
                       </div>
                     </div>
                     
                     <div>
                       <div className={`font-medium mb-2 ${
                         isDarkMode ? 'text-blue-300' : 'text-blue-700'
                       }`}>
                         Breakdown Candles ({secondaryTimeframe})
                       </div>
                       <div className={`space-y-1 ${
                         isDarkMode ? 'text-blue-200' : 'text-blue-600'
                       }`}>
                         <div>🔢 Count: {correlatedCandles.length} candles</div>
                         <div>⏰ From: {correlatedCandles.length > 0 ? new Date(correlatedCandles[0].timestamp).toLocaleTimeString() : 'N/A'}</div>
                         <div>⏰ To: {correlatedCandles.length > 0 ? new Date(correlatedCandles[correlatedCandles.length - 1].timestamp).toLocaleTimeString() : 'N/A'}</div>
                         <div>
                           📊 Range: {correlatedCandles.length > 0 ? 
                             `${Math.min(...correlatedCandles.map(c => c.low)).toFixed(2)} - ${Math.max(...correlatedCandles.map(c => c.high)).toFixed(2)}` 
                             : 'N/A'}
                         </div>
                       </div>
                     </div>
                   </div>
                   
                   <div className={`mt-3 pt-3 border-t text-xs ${
                     isDarkMode ? 'border-blue-800 text-blue-400' : 'border-blue-300 text-blue-600'
                   }`}>
                     💡 Your {selectedSequentialCandles.length} {selectedTimeframe} candle{selectedSequentialCandles.length === 1 ? '' : 's'} {selectedSequentialCandles.length === 1 ? 'is' : 'are'} composed of {correlatedCandles.length} × {secondaryTimeframe} candles shown below
                   </div>
                 </div>
               );
             })()}
             
             {/* Secondary Chart Loading */}
            {secondaryLoading && (
              <div className="flex items-center justify-center h-48">
                <div className={`animate-spin rounded-full h-8 w-8 border-b-2 ${
                  isDarkMode ? 'border-green-400' : 'border-green-600'
                }`}></div>
                <span className={`ml-3 ${
                  isDarkMode ? 'text-gray-300' : 'text-gray-600'
                }`}>Loading secondary chart...</span>
              </div>
            )}
            
            {/* Secondary Chart Error */}
            {secondaryError && (
              <div className={`rounded-lg p-4 border transition-colors duration-200 ${
                isDarkMode 
                  ? 'bg-red-900/20 border-red-800' 
                  : 'bg-red-50 border-red-200'
              }`}>
                <div className="flex items-center">
                  <div className="text-red-500 text-sm font-medium">Error loading secondary chart</div>
                </div>
                <div className="text-red-500 text-sm mt-1">{secondaryError}</div>
                <button 
                  onClick={fetchSecondaryChartData}
                  className="mt-2 px-3 py-1 bg-red-600 text-white rounded text-sm hover:bg-red-700 transition-colors duration-200"
                >
                  Retry
                </button>
              </div>
            )}
            
            {/* Secondary Chart Container */}
            {!secondaryLoading && !secondaryError && (
              <div 
                ref={secondaryChartContainerRef}
                className={`w-full h-48 rounded border transition-colors duration-200 ${
                  isDarkMode 
                    ? 'border-gray-600 bg-gray-900' 
                    : 'border-gray-200 bg-gray-50'
                }`}
                style={{ minHeight: '200px' }}
              >
                {/* Secondary chart is rendered programmatically */}
              </div>
            )}
            
            {/* No Data Message */}
            {!secondaryChartData?.data && !secondaryLoading && !secondaryError && (
              <div className="flex items-center justify-center h-48">
                <div className="text-center">
                  <div className="text-4xl mb-4">📊</div>
                  <h3 className={`text-lg font-medium mb-2 ${
                    isDarkMode ? 'text-gray-300' : 'text-gray-900'
                  }`}>No Secondary Chart Data</h3>
                  <p className={`${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>
                    Secondary chart will load automatically when enabled
                  </p>
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* TJR Labels Toggle Controls */}
      <TjrLabelsControl 
        isDarkMode={isDarkMode} 
        chartRef={chartRef}
        priceSeriesRef={priceSeriesRef}
        chartDataRef={chartDataRef}
        gameMode={gameMode}
        isSelectionMode={isSelectionMode}
      />

      {/* FVG Labels Toggle Controls */}
                  <FvgLabelsControl
              isDarkMode={isDarkMode}
              chartRef={chartRef}
              priceSeriesRef={priceSeriesRef}
              chartDataRef={chartDataRef}
              persistentFvgDataRef={persistentFvgDataRef}
              gameMode={gameMode}
              isSelectionMode={isSelectionMode}
            />

      {/* Selected Candles Panel */}
      <SelectedCandlesPanel 
        isDarkMode={isDarkMode} 
        canSelectCandles={true}
      />

      {/* OHLC Data Summary Table */}
      {chartData?.data && (
        <div className={`rounded-lg shadow-md transition-colors duration-200 ${
          isDarkMode ? 'bg-gray-800' : 'bg-white'
        }`}>
          <div className={`px-6 py-4 border-b transition-colors duration-200 ${
            isDarkMode ? 'border-gray-700' : 'border-gray-200'
          }`}>
            <h3 className={`text-lg font-semibold ${
              isDarkMode ? 'text-white' : 'text-gray-900'
            }`}>Recent OHLC Data</h3>
            <p className={`text-sm ${isDarkMode ? 'text-gray-400' : 'text-gray-600'}`}>
              Last 10 candles • {chartData.data.length} total loaded
            </p>
          </div>
          
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className={`transition-colors duration-200 ${
                isDarkMode ? 'bg-gray-700' : 'bg-gray-50'
              }`}>
                <tr>
                  <th className={`px-4 py-3 text-left text-xs font-medium uppercase tracking-wider ${
                    isDarkMode ? 'text-gray-400' : 'text-gray-500'
                  }`}>Time</th>
                  <th className={`px-4 py-3 text-left text-xs font-medium uppercase tracking-wider ${
                    isDarkMode ? 'text-gray-400' : 'text-gray-500'
                  }`}>Open</th>
                  <th className={`px-4 py-3 text-left text-xs font-medium uppercase tracking-wider ${
                    isDarkMode ? 'text-gray-400' : 'text-gray-500'
                  }`}>High</th>
                  <th className={`px-4 py-3 text-left text-xs font-medium uppercase tracking-wider ${
                    isDarkMode ? 'text-gray-400' : 'text-gray-500'
                  }`}>Low</th>
                  <th className={`px-4 py-3 text-left text-xs font-medium uppercase tracking-wider ${
                    isDarkMode ? 'text-gray-400' : 'text-gray-500'
                  }`}>Close</th>
                  <th className={`px-4 py-3 text-left text-xs font-medium uppercase tracking-wider ${
                    isDarkMode ? 'text-gray-400' : 'text-gray-500'
                  }`}>Volume</th>
                  <th className={`px-4 py-3 text-left text-xs font-medium uppercase tracking-wider ${
                    isDarkMode ? 'text-gray-400' : 'text-gray-500'
                  }`}>Change</th>
                </tr>
              </thead>
              <tbody className={`divide-y transition-colors duration-200 ${
                isDarkMode ? 'bg-gray-800 divide-gray-700' : 'bg-white divide-gray-200'
              }`}>
                {chartData.data.slice(-10).reverse().map((candle, index) => {
                  const change = candle.close - candle.open;
                  const changePercent = (change / candle.open) * 100;
                  
                  return (
                    <tr key={index} className={`transition-colors duration-200 ${
                      isDarkMode ? 'hover:bg-gray-700' : 'hover:bg-gray-50'
                    }`}>
                      <td className={`px-4 py-3 whitespace-nowrap text-sm font-mono ${
                        isDarkMode ? 'text-gray-300' : 'text-gray-900'
                      }`}>
                        {new Date(candle.timestamp).toLocaleTimeString()}
                      </td>
                      <td className={`px-4 py-3 whitespace-nowrap text-sm font-mono ${
                        isDarkMode ? 'text-gray-300' : 'text-gray-900'
                      }`}>
                        {formatPrice(candle.open)}
                      </td>
                      <td className={`px-4 py-3 whitespace-nowrap text-sm font-mono font-semibold text-green-600`}>
                        {formatPrice(candle.high)}
                      </td>
                      <td className={`px-4 py-3 whitespace-nowrap text-sm font-mono font-semibold text-red-600`}>
                        {formatPrice(candle.low)}
                      </td>
                      <td className={`px-4 py-3 whitespace-nowrap text-sm font-mono font-bold ${
                        change >= 0 ? 'text-green-600' : 'text-red-600'
                      }`}>
                        {formatPrice(candle.close)}
                      </td>
                      <td className={`px-4 py-3 whitespace-nowrap text-sm font-mono ${
                        isDarkMode ? 'text-gray-400' : 'text-gray-500'
                      }`}>
                        {candle.volume ? candle.volume.toLocaleString() : 'N/A'}
                      </td>
                      <td className={`px-4 py-3 whitespace-nowrap text-sm font-mono ${
                        change >= 0 ? 'text-green-600' : 'text-red-600'
                      }`}>
                        {change >= 0 ? '+' : ''}{formatPrice(change)}
                        <br />
                        <span className="text-xs">
                          {change >= 0 ? '+' : ''}{changePercent.toFixed(2)}%
                        </span>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Chart Development Notes */}
      <details className={`rounded-lg border transition-colors duration-200 ${
        isDarkMode ? 'bg-gray-800 border-gray-700' : 'bg-white border-gray-200'
      }`}>
        <summary className={`p-6 cursor-pointer font-semibold transition-colors duration-200 ${
          isDarkMode 
            ? 'text-gray-300 hover:bg-gray-700' 
            : 'text-gray-700 hover:bg-gray-50'
        }`}>
          📋 Chart Development Roadmap (Click to expand)
        </summary>
        <div className="px-6 pb-6">
          <div className={`rounded p-4 transition-colors duration-200 ${
            isDarkMode ? 'bg-gray-900' : 'bg-gray-50'
          }`}>
            <h4 className={`font-semibold mb-3 ${isDarkMode ? 'text-gray-200' : 'text-gray-800'}`}>
              🚀 Implementation Status:
            </h4>
            <ul className={`space-y-2 text-sm ${isDarkMode ? 'text-gray-300' : 'text-gray-700'}`}>
              <li>✅ <strong>Basic Chart Dashboard:</strong> Layout and data fetching complete</li>
              <li>✅ <strong>Chart Code Implementation:</strong> Full lightweight-charts integration ready</li>
              <li>🟡 <strong>Package Installation:</strong> Run: <code className={`px-1 rounded ${isDarkMode ? 'bg-gray-800' : 'bg-gray-100'}`}>npm install lightweight-charts</code></li>
              <li>🟡 <strong>Activate Charts:</strong> Uncomment chart code in Chart.jsx</li>
              <li>✅ <strong>Technical Indicators:</strong> SMA, EMA calculations implemented</li>
              <li>✅ <strong>Volume Chart:</strong> Volume histogram ready</li>
              <li>✅ <strong>Theme Support:</strong> Dark/light mode integration</li>
              <li>⭕ <strong>Bollinger Bands:</strong> Advanced indicator (future enhancement)</li>
              <li>⭕ <strong>Pattern Recognition:</strong> Integration with vector similarity dashboard</li>
            </ul>
            
            <h4 className={`font-semibold mt-4 mb-3 ${isDarkMode ? 'text-gray-200' : 'text-gray-800'}`}>
              💡 Chart Features Ready for Development:
            </h4>
            <ul className={`space-y-1 text-sm ${isDarkMode ? 'text-gray-400' : 'text-gray-600'}`}>
              <li>• Data is loaded and formatted correctly</li>
              <li>• OHLC data structure is compatible with lightweight-charts</li>
              <li>• Controls for chart type, timeframe, and indicators are ready</li>
              <li>• Dark/light theme support is built-in</li>
              <li>• Market stats and price change calculations work</li>
            </ul>
          </div>
        </div>
      </details>
    </div>
  );
};

export default TradingChartDashboard; 