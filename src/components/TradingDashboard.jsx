import { useState, useEffect } from 'react';
import TradingChartDashboard from './Chart.jsx';
import BacktestDashboard from './backtest.jsx';
import TradingLLMDashboard from './LLMDashboard.jsx';
import PipelineDashboard from './PipelineDashboard.jsx';
import ModelTraining from './model.jsx';
import SelectedCandlesPanel from './shared/SelectedCandlesPanel.jsx';
import { useTrading } from '../context/TradingContext';
import { useDateRanges } from '../hooks/useDateRanges';
import { FETCH_MODES, DATE_RANGE_TYPES } from '../utils/constants';

// Reusable InfoTooltip component
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

// Reusable Data Selection & Controls component
const DataSelectionControls = ({ 
  selectedSymbol, 
  setSelectedSymbol, 
  selectedTimeframe, 
  setSelectedTimeframe, 
  rowLimit, 
  setRowLimit, 
  sortOrder, 
  setSortOrder, 
  searchTerm, 
  setSearchTerm, 
  selectedSearchColumn, 
  setSelectedSearchColumn, 
  showColumnFilters, 
  setShowColumnFilters, 
  showDebugInfo, 
  setShowDebugInfo, 
  handleRefresh, 
  lastFetchInfo, 
  isDarkMode,
  dashboardType = 'data', // 'data', 'vector', 'chart'
  // Date range functionality
  fetchMode,
  setFetchMode,
  dateRangeType,
  setDateRangeType,
  startDate,
  setStartDate,
  endDate,
  setEndDate
}) => {

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
          {setShowColumnFilters && (
            <button
              onClick={() => setShowColumnFilters(!showColumnFilters)}
              className={`px-3 py-2 text-sm rounded-md transition-colors flex items-center gap-1 ${
                isDarkMode 
                  ? 'bg-gray-700 hover:bg-gray-600 text-gray-300' 
                  : 'bg-gray-100 hover:bg-gray-200'
              }`}
            >
              🔍 Filters {showColumnFilters ? '▼' : '▶'}
            </button>
          )}
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
          {setShowDebugInfo && (
            <button
              onClick={() => setShowDebugInfo(!showDebugInfo)}
              className={`px-3 py-2 text-sm rounded-md transition-colors flex items-center gap-1 ${
                isDarkMode 
                  ? 'bg-yellow-800 hover:bg-yellow-700 text-yellow-300' 
                  : 'bg-yellow-100 hover:bg-yellow-200 text-yellow-700'
              }`}
            >
              🐛 Debug {showDebugInfo ? '▼' : '▶'}
            </button>
          )}
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
              onClick={() => setFetchMode && setFetchMode(FETCH_MODES.LIMIT)}
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
              onClick={() => setFetchMode && setFetchMode(FETCH_MODES.DATE_RANGE)}
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

        {(!fetchMode || fetchMode === FETCH_MODES.LIMIT) && (
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

      {/* Search and Advanced Controls */}
      {showColumnFilters && setSearchTerm && setSelectedSearchColumn && (
        <div className={`mt-4 pt-4 border-t rounded-md p-4 transition-colors duration-200 ${
          isDarkMode 
            ? 'border-gray-600 bg-gray-700' 
            : 'border-gray-200 bg-gray-50'
        }`}>
          <h4 className={`text-md font-medium mb-3 ${
            isDarkMode ? 'text-white' : 'text-gray-800'
          }`}>Advanced Filters & Controls</h4>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div>
              <div className="flex items-center mb-2">
                <label className={`block text-sm font-medium ${
                  isDarkMode ? 'text-gray-300' : 'text-gray-700'
                }`}>
                  Search Data
                </label>
                <InfoTooltip id="search-functionality" content={
                  <div>
                    <p className="font-semibold mb-2">🔍 Search Features</p>
                    <p className="mb-2">Powerful search across your trading data:</p>
                    <ul className="list-disc list-inside space-y-1 text-xs">
                      <li><strong>All Columns:</strong> Searches timestamps, prices, and volume</li>
                      <li><strong>Specific Column:</strong> Target your search to one data type</li>
                      <li><strong>Smart Formatting:</strong> Automatically handles price formatting</li>
                      <li><strong>Real-time:</strong> Results update as you type</li>
                    </ul>
                    <p className="mt-2 text-xs"><strong>Tip:</strong> Remove commas from numbers for better search results.</p>
                  </div>
                } isDarkMode={isDarkMode} />
              </div>
              <input
                type="text"
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                placeholder={selectedSearchColumn === 'all' ? 
                  "Search in all columns..." : 
                  `Search in ${selectedSearchColumn} column...`}
                className={`w-full rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500 transition-colors duration-200 ${
                  isDarkMode 
                    ? 'border-gray-600 bg-gray-800 text-white placeholder-gray-400' 
                    : 'border-gray-300 bg-white text-gray-900'
                }`}
              />
            </div>
            <div>
              <label className={`block text-sm font-medium mb-2 ${
                isDarkMode ? 'text-gray-300' : 'text-gray-700'
              }`}>
                Search Column
              </label>
              <select
                value={selectedSearchColumn}
                onChange={(e) => setSelectedSearchColumn(e.target.value)}
                className={`w-full rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500 transition-colors duration-200 ${
                  isDarkMode 
                    ? 'border-gray-600 bg-gray-800 text-white' 
                    : 'border-gray-300 bg-white text-gray-900'
                }`}
              >
                <option value="all">🔍 ALL COLUMNS</option>
                <option value="timestamp">📅 Timestamp</option>
                <option value="open">📈 Open Price</option>
                <option value="high">🔺 High Price</option>
                <option value="low">🔻 Low Price</option>
                <option value="close">💰 Close Price</option>
                <option value="volume">📊 Volume</option>
              </select>
            </div>
            <div className="flex items-end">
              <button
                onClick={() => {
                  setSearchTerm('');
                  setSelectedSearchColumn('all');
                  if (setSortOrder) setSortOrder('desc');
                  if (setShowColumnFilters) setShowColumnFilters(false);
                }}
                className={`w-full px-3 py-2 text-sm rounded-md transition-colors duration-200 ${
                  isDarkMode 
                    ? 'bg-gray-600 text-gray-200 hover:bg-gray-500' 
                    : 'bg-gray-200 text-gray-800 hover:bg-gray-300'
                }`}
              >
                🔄 Reset Filters
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Debug Information Panel */}
      {showDebugInfo && lastFetchInfo && (
        <div className={`mt-4 pt-4 border-t rounded-md p-4 transition-colors duration-200 ${
          isDarkMode 
            ? 'border-gray-600 bg-yellow-900/20' 
            : 'border-gray-200 bg-yellow-50'
        }`}>
          <h4 className={`text-md font-medium mb-3 ${
            isDarkMode ? 'text-yellow-300' : 'text-yellow-800'
          }`}>🐛 Debug Information</h4>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
            <div>
              <p><strong>Last Fetch:</strong> {lastFetchInfo.timestamp}</p>
              <p><strong>Requested Data:</strong> <span className={lastFetchInfo.requestedData === 'OLDEST' ? 'text-green-600' : 'text-blue-600'}>{lastFetchInfo.requestedData}</span> {lastFetchInfo.rowLimit} records</p>
              <p><strong>Page:</strong> {lastFetchInfo.currentPage} (offset: {lastFetchInfo.offset})</p>
              <p><strong>Sort:</strong> {lastFetchInfo.sortOrder} by {lastFetchInfo.sortColumn}</p>
              <p><strong>Data Quality:</strong> <span className={lastFetchInfo.dataQuality?.includes('GOOD') ? 'text-green-600' : 'text-red-600'}>{lastFetchInfo.dataQuality}</span></p>
            </div>
            <div>
              <p><strong>Backend Sorting:</strong> <span className={lastFetchInfo.backendHandledSorting ? 'text-green-600' : 'text-red-600'}>{lastFetchInfo.backendHandledSorting ? '✅ Working' : '❌ Not Working'}</span></p>
              <p><strong>Total Records:</strong> {lastFetchInfo.totalRecords?.toLocaleString() || 'Unknown'}</p>
              {lastFetchInfo.actualDataRange && (
                <>
                  <p><strong>Data Span:</strong> {lastFetchInfo.actualDataRange.span}</p>
                  <p><strong>First Record:</strong> {new Date(lastFetchInfo.actualDataRange.first).toLocaleString()}</p>
                  <p><strong>Last Record:</strong> {new Date(lastFetchInfo.actualDataRange.last).toLocaleString()}</p>
                </>
              )}
            </div>
          </div>
          <div className="mt-3 p-2 bg-yellow-100 rounded text-xs">
            <strong>💡 Tip:</strong> If "Data Quality" shows "POOR" for ascending order, your backend may only be returning recent data. 
            For true oldest records, you may need to modify your backend API to support historical data access.
          </div>
          <div className="mt-2 text-xs text-gray-600 break-all">
            <strong>API URL:</strong> {lastFetchInfo.apiUrl}
          </div>
        </div>
      )}

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

// Import the new Vector Dashboard component
const TradingVectorDashboard = ({ 
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
    selectedVectorType,
    setSelectedVectorType,
    vectorViewMode,
    setVectorViewMode,
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
  const [vectorData, setVectorData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [selectedRows, setSelectedRows] = useState([]);
  const [availableVectors, setAvailableVectors] = useState([]);
  const [missingVectors, setMissingVectors] = useState([]);
  const [allVectorTypes, setAllVectorTypes] = useState([]); // Now dynamic instead of hardcoded!
  
  // Shape similarity state - moved to top level to follow Rules of Hooks
  const [shapeSimilarityData, setShapeSimilarityData] = useState(null);
  const [similarityLoading, setSimilarityLoading] = useState(false);
  const [similarityError, setSimilarityError] = useState(null);
  
  // Matrix dimension controls
  const [matrixRows, setMatrixRows] = useState(20);
  const [matrixCols, setMatrixCols] = useState(20);
  
  // Calculate responsive cell size based on matrix dimensions
  const getMatrixCellSize = () => {
    const maxCols = Math.max(matrixCols, 20); // Minimum reference is 20 columns
    
    if (maxCols <= 20) return { width: 'w-6', height: 'h-6', size: '24px' }; // 24px default
    if (maxCols <= 30) return { width: 'w-5', height: 'h-5', size: '20px' }; // 20px
    if (maxCols <= 40) return { width: 'w-4', height: 'h-4', size: '16px' }; // 16px
    if (maxCols <= 60) return { width: 'w-3', height: 'h-3', size: '12px' }; // 12px
    if (maxCols <= 80) return { width: 'w-2', height: 'h-2', size: '8px' };  // 8px
    return { width: 'w-1', height: 'h-1', size: '4px' }; // 4px for very large matrices
  };
  
  const cellSize = getMatrixCellSize();

  const API_BASE_URL = 'http://localhost:8000/api/trading';

  // Function to generate vector type metadata based on the key name
  const getVectorTypeMetadata = (key) => {
    // Default metadata
    let name = key;
    let color = 'bg-gray-500';
    let description = 'Custom vector type';
    
    // Determine metadata based on key patterns
    if (key.includes('raw_ohlc_vec')) {
      name = 'Raw OHLC';
      color = 'bg-blue-500';
      description = 'Direct numerical values';
    } else if (key.includes('raw_ohlcv_vec')) {
      name = 'Raw OHLCV';
      color = 'bg-blue-600';
      description = 'With volume data';
    } else if (key === 'norm_ohlc') {
      name = 'Normalized OHLC';
      color = 'bg-green-500';
      description = 'Z-score normalized';
    } else if (key === 'norm_ohlcv') {
      name = 'Normalized OHLCV';
      color = 'bg-green-600';
      description = 'Z-score with volume';
    } else if (key.includes('BERT_ohlc') && !key.includes('ohlcv')) {
      name = 'BERT OHLC';
      color = 'bg-purple-500';
      description = 'Semantic embeddings';
    } else if (key.includes('BERT_ohlcv')) {
      name = 'BERT OHLCV';
      color = 'bg-purple-600';
      description = 'Semantic with volume';
    } else if (key === 'iso_ohlc') {
      name = 'ISO OHLC';
      color = 'bg-orange-500';
      description = 'Isolation forest features';
    } else if (key === 'iso_ohlcv') {
      name = 'ISO OHLCV';
      color = 'bg-orange-600';
      description = 'Isolation forest with volume';
    } else if (key.includes('iso_')) {
      // Generic ISO handling
      name = key.replace(/_/g, ' ').toUpperCase();
      color = 'bg-orange-500';
      description = 'Isolation forest based vector';
    }
    
    return { key, name, color, description };
  };

  // Get available vector types based on actual data
  const getAvailableVectorTypes = () => {
    if (!allVectorTypes || allVectorTypes.length === 0) return [];
    if (!vectorData?.data || vectorData.data.length === 0) return allVectorTypes;
    
    const sampleRow = vectorData.data[0];
    return allVectorTypes.filter(type => 
      sampleRow.hasOwnProperty(type.key) && sampleRow[type.key] !== null && sampleRow[type.key] !== undefined
    );
  };

  const vectorTypes = getAvailableVectorTypes();

  useEffect(() => {
    if (selectedSymbol && selectedTimeframe) {
      fetchVectorData();
    }
  }, [selectedSymbol, selectedTimeframe, rowLimit, sortOrder, fetchMode, dateRangeType, startDate, endDate]);

  // Handle view mode switching based on vector type
  useEffect(() => {
    if (selectedVectorType.includes('iso_')) {
      // For ISO vectors, default to shape-similarity mode
      if (vectorViewMode !== 'shape-similarity' && vectorViewMode !== 'comparison') {
        setVectorViewMode('shape-similarity');
      }
    } else {
      // For non-ISO vectors, default to heatmap mode
      if (vectorViewMode === 'shape-similarity') {
        setVectorViewMode('heatmap');
      }
    }
  }, [selectedVectorType]);

  // Fetch shape similarity data when ISO vector is selected
  useEffect(() => {
    if (selectedVectorType.includes('iso_') && vectorData?.data?.length > 0) {
      fetchShapeSimilarityData();
    } else {
      // Clear similarity data when switching away from ISO vectors
      setShapeSimilarityData(null);
      setSimilarityError(null);
    }
  }, [selectedVectorType, selectedSymbol, selectedTimeframe, matrixRows, matrixCols]);

  const fetchShapeSimilarityData = async () => {
    setSimilarityLoading(true);
    setSimilarityError(null);
    try {
      const params = new URLSearchParams({
        vector_type: selectedVectorType,
        limit: Math.max(matrixRows, matrixCols).toString()
      });

      const response = await fetch(`${API_BASE_URL}/shape-similarity/${selectedSymbol}/${selectedTimeframe}?${params}`);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data = await response.json();
      setShapeSimilarityData(data);
    } catch (err) {
      console.error('Error fetching shape similarity:', err);
      setSimilarityError(err.message);
      // Fallback to client-side calculation
      calculateClientSideSimilarity();
    } finally {
      setSimilarityLoading(false);
    }
  };

  const calculateClientSideSimilarity = () => {
    if (!vectorData?.data) return;
    
    const dataToShow = vectorData.data.slice(0, Math.max(matrixRows, matrixCols));
    const vectors = dataToShow.map(row => row[selectedVectorType]).filter(v => v && Array.isArray(v));
    
    if (vectors.length === 0) return;

    // Calculate pairwise similarities (client-side fallback)
    const shapeSimilarityMatrix = [];
    for (let i = 0; i < vectors.length; i++) {
      shapeSimilarityMatrix[i] = [];
      for (let j = 0; j < vectors.length; j++) {
        if (i === j) {
          shapeSimilarityMatrix[i][j] = 1.0;
        } else {
          shapeSimilarityMatrix[i][j] = calculateVectorSimilarity(vectors[i], vectors[j]);
        }
      }
    }

    // Create fallback data structure
    setShapeSimilarityData({
      similarity_matrix: {
        matrix: shapeSimilarityMatrix,
        candles: dataToShow,
        statistics: {
          average_similarity: 0,
          pattern_diversity: "Unknown"
        }
      },
      count: dataToShow.length
    });
  };

  const fetchVectorData = async () => {
    setLoading(true);
    console.log(`🚀 fetchVectorData called - Symbol: ${selectedSymbol}, Timeframe: ${selectedTimeframe}, RowLimit: ${rowLimit}, SortOrder: ${sortOrder}, FetchMode: ${fetchMode}, DateRangeType: ${dateRangeType}`);
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
        // Set a reasonable limit to avoid overwhelming the UI
        queryParams.append('limit', '10000');
      } else {
        // Use record limit mode
        queryParams.append('limit', rowLimit.toString());
      }

      // Add sorting parameters (this was missing!)
      queryParams.append('order', sortOrder);
      queryParams.append('sort_by', 'timestamp');
      
      // Include vectors
      queryParams.append('include_vectors', 'true');

      const finalUrl = `${url}?${queryParams.toString()}`;
      console.log('Vector API URL:', finalUrl);
      
      const response = await fetch(finalUrl);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data = await response.json();
      
      // Apply client-side sorting to ensure proper ordering (same as main data fetch)
      if (data.data && Array.isArray(data.data)) {
        let sortedData = [...data.data];
        
        console.log(`🔀 Applying client-side sorting to vector data by timestamp (${sortOrder})...`);
        
        sortedData.sort((a, b) => {
          const aVal = new Date(a.timestamp);
          const bVal = new Date(b.timestamp);
          
          if (aVal < bVal) return sortOrder === 'asc' ? -1 : 1;
          if (aVal > bVal) return sortOrder === 'asc' ? 1 : -1;
          return 0;
        });
        
        data.data = sortedData;
        console.log(`✅ Vector data sorting complete. Data range: ${data.data.length > 0 ? `${new Date(data.data[0].timestamp).toLocaleDateString()} to ${new Date(data.data[data.data.length - 1].timestamp).toLocaleDateString()}` : 'No data'}`);
      }
      
      setVectorData(data);
      
      // Dynamically build vector types from actual data
      if (data.data && data.data.length > 0) {
        const sampleRow = data.data[0];
        
        // Extract all potential vector columns from the data
        const vectorKeys = Object.keys(sampleRow).filter(key => {
          // Check if this looks like a vector column
          const value = sampleRow[key];
          const isVector = Array.isArray(value) && value.length > 0 && typeof value[0] === 'number';
          const looksLikeVector = key.includes('_vec') || key.includes('BERT') || 
                                 key.includes('norm_') || key.includes('raw_ohlc') || 
                                 key.includes('iso_');
          return isVector || looksLikeVector;
        });
        
        // Build allVectorTypes dynamically
        const dynamicVectorTypes = vectorKeys.map(key => getVectorTypeMetadata(key));
        setAllVectorTypes(dynamicVectorTypes);
        
        // Now check which are available vs missing
        const available = dynamicVectorTypes.filter(type => 
          sampleRow.hasOwnProperty(type.key) && sampleRow[type.key] !== null && sampleRow[type.key] !== undefined
        );
        const missing = dynamicVectorTypes.filter(type => 
          !sampleRow.hasOwnProperty(type.key) || sampleRow[type.key] === null || sampleRow[type.key] === undefined
        );
        
        setAvailableVectors(available);
        setMissingVectors(missing);
        
        // Auto-select first available vector type if current selection is not available
        if (available.length > 0 && !available.find(v => v.key === selectedVectorType)) {
          setSelectedVectorType(available[0].key);
        }
      }
      
      setError(null);
    } catch (err) {
      setError(`Failed to fetch vector data: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  const getVectorStats = (vectors) => {
    if (!vectors || vectors.length === 0) return null;
    
    const flatVectors = vectors.filter(v => v && Array.isArray(v)).flat();
    if (flatVectors.length === 0) return null;

    // Use iterative approach to avoid call stack overflow with large arrays
    let min = flatVectors[0];
    let max = flatVectors[0];
    let sum = 0;
    
    for (let i = 0; i < flatVectors.length; i++) {
      const val = flatVectors[i];
      if (val < min) min = val;
      if (val > max) max = val;
      sum += val;
    }
    
    const avg = sum / flatVectors.length;
    
    // Calculate standard deviation
    let sumSquaredDiffs = 0;
    for (let i = 0; i < flatVectors.length; i++) {
      sumSquaredDiffs += Math.pow(flatVectors[i] - avg, 2);
    }
    const std = Math.sqrt(sumSquaredDiffs / flatVectors.length);

    return { min, max, avg, std, count: vectors.length, dimensions: vectors[0]?.length || 0 };
  };

  const getColorFromValue = (value, min, max) => {
    // Normalize value to 0-1 range
    const normalized = (value - min) / (max - min);
    
    // Create a color scale from blue (low) to red (high)
    if (normalized < 0.5) {
      // Blue to green
      const factor = normalized * 2;
      const r = Math.floor(0 * (1 - factor) + 0 * factor);
      const g = Math.floor(100 * (1 - factor) + 255 * factor);
      const b = Math.floor(255 * (1 - factor) + 0 * factor);
      return `rgb(${r}, ${g}, ${b})`;
    } else {
      // Green to red
      const factor = (normalized - 0.5) * 2;
      const r = Math.floor(0 * (1 - factor) + 255 * factor);
      const g = Math.floor(255 * (1 - factor) + 0 * factor);
      const b = Math.floor(0 * (1 - factor) + 0 * factor);
      return `rgb(${r}, ${g}, ${b})`;
    }
  };

  const renderVectorHeatmap = () => {
    if (!vectorData?.data) return null;

    try {
      // Use the full data up to the row limit instead of hardcoding 20
      const dataToShow = vectorData.data.slice(0, rowLimit);
      const vectors = dataToShow
        .map(row => row[selectedVectorType])
        .filter(v => v && Array.isArray(v));

      if (vectors.length === 0) return <div className={`${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>No vector data available</div>;

      const maxDimensions = Math.min(vectors[0]?.length || 0, 20); // Limit to 20 dimensions for display
      
      // Safety check for very large vectors
      if (vectors[0]?.length > 1000) {
        return (
          <div className={`p-4 rounded-lg border transition-colors duration-200 ${
            isDarkMode 
              ? 'bg-yellow-900/20 border-yellow-800 text-yellow-300' 
              : 'bg-yellow-50 border-yellow-200 text-yellow-800'
          }`}>
            <div className="text-sm font-medium">Large Vector Detected</div>
            <div className="text-xs mt-1">
              This vector type has {vectors[0].length} dimensions (likely BERT embeddings). 
              Heatmap display is limited to smaller vectors for performance.
            </div>
            <div className="text-xs mt-2">
              Try using the "Comparison" view mode instead, or switch to Raw OHLC/OHLCV vectors for heatmap visualization.
            </div>
          </div>
        );
      }
      
      // Calculate global min/max for proper color scaling
      const allValues = vectors.flat();
      
      // Additional safety for very large flattened arrays
      if (allValues.length > 10000) {
        return (
          <div className={`p-4 rounded-lg border transition-colors duration-200 ${
            isDarkMode 
              ? 'bg-yellow-900/20 border-yellow-800 text-yellow-300' 
              : 'bg-yellow-50 border-yellow-200 text-yellow-800'
          }`}>
            <div className="text-sm font-medium">Too Much Data for Heatmap</div>
            <div className="text-xs mt-1">
              {allValues.length.toLocaleString()} total values detected. Heatmap visualization is optimized for smaller datasets.
            </div>
            <div className="text-xs mt-2">
              Try reducing the vector limit or using a different vector type.
            </div>
          </div>
        );
      }
      
      const globalMin = Math.min(...allValues);
      const globalMax = Math.max(...allValues);
      
      // Helper function to handle candle selection
      const handleCandleSelect = (candle, rowIndex) => {
        const candleId = `${selectedSymbol}_${selectedTimeframe}_${candle.timestamp}`;
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
            sourceIndex: rowIndex
          };
          addSelectedCandle(candleWithMetadata);
        }
      };
      
      return (
        <div className="space-y-2">
          {/* Color scale legend with info tooltip */}
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4 text-xs">
              <div className="flex items-center">
              <span className={isDarkMode ? 'text-gray-300' : 'text-gray-600'}>Color Scale:</span>
                <InfoTooltip id="color-scale" content={
                  <div>
                    <p className="font-semibold mb-2">🎨 Color Scale</p>
                    <p className="mb-2">Visual encoding of vector values using color intensity:</p>
                    <ul className="list-disc list-inside space-y-1 text-xs">
                      <li><strong>🔵 Blue (Low):</strong> Values in the bottom 33% of range</li>
                      <li><strong>🟢 Green (Mid):</strong> Values in the middle 33% of range</li>
                      <li><strong>🔴 Red (High):</strong> Values in the top 33% of range</li>
                    </ul>
                    <p className="mt-2 text-xs">Colors are normalized across all values in the current dataset for consistent comparison.</p>
                  </div>
                } isDarkMode={isDarkMode} />
              </div>
              <div className="flex items-center space-x-2">
                <div className="w-4 h-4 rounded" style={{ backgroundColor: 'rgb(0, 100, 255)' }}></div>
                <span className={isDarkMode ? 'text-gray-400' : 'text-gray-500'}>Low ({globalMin.toFixed(2)})</span>
                <div className="w-4 h-4 rounded" style={{ backgroundColor: 'rgb(0, 255, 0)' }}></div>
                <span className={isDarkMode ? 'text-gray-400' : 'text-gray-500'}>Mid</span>
                <div className="w-4 h-4 rounded" style={{ backgroundColor: 'rgb(255, 0, 0)' }}></div>
                <span className={isDarkMode ? 'text-gray-400' : 'text-gray-500'}>High ({globalMax.toFixed(2)})</span>
              </div>
            </div>
            <div className="text-xs">
              <span className={isDarkMode ? 'text-gray-400' : 'text-gray-500'}>
                Showing {dataToShow.length} of {vectorData.data.length} candles • Click rows to select
              </span>
            </div>
          </div>
          
          {/* Dimension headers */}
          <div className="flex items-center space-x-1 text-xs">
            <div className="flex items-center">
            <div className={`w-24 ${isDarkMode ? 'text-gray-400' : 'text-gray-600'}`}>Candle Info</div>
              <InfoTooltip id="rows-explanation" content={
                <div>
                  <p className="font-semibold mb-2">📋 Candle Information</p>
                  <p className="mb-2">Each row represents one trading period (candle):</p>
                  <ul className="list-disc list-inside space-y-1 text-xs">
                    <li><strong>Timestamp:</strong> When the candle occurred</li>
                    <li><strong>Close Price:</strong> Final price for that period</li>
                    <li><strong>Selection:</strong> Click any row to add/remove from selected candles</li>
                    <li><strong>Order:</strong> Depends on your sort setting in controls</li>
                  </ul>
                </div>
              } isDarkMode={isDarkMode} />
            </div>
            <div className="flex items-center space-x-0.5">
            <div className="flex space-x-0.5">
              {Array.from({ length: maxDimensions }, (_, i) => (
                <div key={i} className={`w-8 text-center ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>
                  D{i}
                </div>
              ))}
              </div>
              <InfoTooltip id="dimensions" content={
                <div>
                  <p className="font-semibold mb-2">📊 Vector Dimensions</p>
                  <p className="mb-2">Each column represents a dimension of your trading vector:</p>
                  <ul className="list-disc list-inside space-y-1 text-xs">
                    <li><strong>Raw OHLC:</strong> D0=Open, D1=High, D2=Low, D3=Close</li>
                    <li><strong>Raw OHLCV:</strong> D0=Open, D1=High, D2=Low, D3=Close, D4=Volume</li>
                    <li><strong>Normalized:</strong> Same mapping but z-score scaled</li>
                    <li><strong>BERT:</strong> D0-D383 are semantic feature dimensions</li>
                  </ul>
                  <p className="mt-2 text-xs">Hover over any cell to see exact values. Each dimension captures different aspects of market behavior.</p>
                </div>
              } isDarkMode={isDarkMode} />
            </div>
          </div>
          
          {/* Vector heatmap */}
          <div className="space-y-1">
            {dataToShow.map((candle, rowIndex) => {
              const vector = candle[selectedVectorType];
              if (!vector || !Array.isArray(vector)) return null;
              
              const candleId = `${selectedSymbol}_${selectedTimeframe}_${candle.timestamp}`;
              const isSelected = selectedCandles.some(c => c.id === candleId);
              
              return (
                <div 
                  key={rowIndex} 
                  className={`flex items-center space-x-1 cursor-pointer rounded transition-all duration-200 hover:shadow-md ${
                    isSelected 
                      ? isDarkMode 
                        ? 'bg-blue-900/40 border-l-4 border-blue-400 pl-2' 
                        : 'bg-blue-50 border-l-4 border-blue-500 pl-2'
                      : isDarkMode 
                        ? 'hover:bg-gray-700 pl-2' 
                        : 'hover:bg-gray-50 pl-2'
                  }`}
                  onClick={() => handleCandleSelect(candle, rowIndex)}
                  title={`Click to ${isSelected ? 'deselect' : 'select'} this candle for analysis`}
                >
                  <div className={`w-24 text-xs font-mono ${isDarkMode ? 'text-gray-300' : 'text-gray-700'}`}>
                    <div className="truncate">{new Date(candle.timestamp).toLocaleDateString()}</div>
                    <div className="text-xs opacity-75">${candle.close?.toFixed(2)}</div>
                  </div>
                  <div className="flex space-x-0.5">
                    {vector.slice(0, maxDimensions).map((value, dimIndex) => {
                      const color = getColorFromValue(value, globalMin, globalMax);
                      return (
                        <div
                          key={dimIndex}
                          className="w-8 h-8 rounded-sm border border-gray-300 hover:scale-110 transition-transform"
                          style={{ backgroundColor: color }}
                          title={`${new Date(candle.timestamp).toLocaleString()}\nClose: $${candle.close?.toFixed(2)}\nDim ${dimIndex}: ${value.toFixed(4)}\nRange: ${globalMin.toFixed(2)} to ${globalMax.toFixed(2)}`}
                        />
                      );
                    })}
                  </div>
                  {isSelected && (
                    <div className="ml-2 text-blue-500 text-xs">✓ Selected</div>
                  )}
                </div>
              );
            })}
          </div>
          
          {dataToShow.length < vectorData.data.length && (
            <div className={`text-xs ${isDarkMode ? 'text-gray-400' : 'text-gray-500'} text-center mt-2`}>
              Showing {dataToShow.length} of {vectorData.data.length} vectors (adjust "Data Limit" to show more)
            </div>
          )}
        </div>
      );
    } catch (error) {
      console.error('Error rendering vector heatmap:', error);
      return (
        <div className={`p-4 rounded-lg border transition-colors duration-200 ${
          isDarkMode 
            ? 'bg-red-900/20 border-red-800 text-red-300' 
            : 'bg-red-50 border-red-200 text-red-800'
        }`}>
          <div className="text-sm font-medium">Error rendering heatmap</div>
          <div className="text-xs mt-1">
            {error.message || 'An unknown error occurred while rendering the vector heatmap'}
          </div>
          <div className="text-xs mt-2 opacity-75">
            This often happens with very large vector datasets. Try reducing the vector limit or using a simpler vector type.
          </div>
        </div>
      );
    }
  };

  const [compareIndices, setCompareIndices] = useState([0, 1]);

  // Vector statistics explanations
  const vectorStatsInfo = {
    count: {
      title: "Vector Count",
      content: (
        <div>
          <p className="font-semibold mb-2">📊 Vector Count</p>
          <p className="mb-2">This shows the total number of trading vectors (time periods) loaded from your database.</p>
          <p className="mb-2"><strong>What it means:</strong></p>
          <ul className="list-disc list-inside space-y-1 text-xs">
            <li>Each vector represents one candle/time period (e.g., 1 minute, 1 hour)</li>
            <li>More vectors = more historical data for pattern analysis</li>
            <li>Typical ranges: 25-500 vectors depending on your settings</li>
            <li>Higher counts improve ML model training but increase processing time</li>
          </ul>
        </div>
      )
    },
    dimensions: {
      title: "Dimensions",
      content: (
        <div>
          <p className="font-semibold mb-2">🔢 Vector Dimensions</p>
          <p className="mb-2">The number of features/values in each trading vector.</p>
          <p className="mb-2"><strong>Common dimension sizes:</strong></p>
          <ul className="list-disc list-inside space-y-1 text-xs">
            <li><strong>Raw OHLC:</strong> 4 dimensions (Open, High, Low, Close)</li>
            <li><strong>Raw OHLCV:</strong> 5 dimensions (+ Volume)</li>
            <li><strong>Normalized:</strong> Same as raw but z-score scaled</li>
            <li><strong>BERT vectors:</strong> 384 dimensions (semantic embeddings)</li>
          </ul>
          <p className="mt-2 text-xs">Higher dimensions capture more complexity but require more computational resources.</p>
        </div>
      )
    },
    range: {
      title: "Value Range",
      content: (
        <div>
          <p className="font-semibold mb-2">📈 Value Range (Min to Max)</p>
          <p className="mb-2">The lowest and highest values across all vector dimensions.</p>
          <p className="mb-2"><strong>Range interpretation:</strong></p>
          <ul className="list-disc list-inside space-y-1 text-xs">
            <li><strong>Raw data:</strong> Actual price values (e.g., $4,200 - $4,800)</li>
            <li><strong>Normalized data:</strong> Usually -3 to +3 (z-scores)</li>
            <li><strong>BERT embeddings:</strong> Typically -1 to +1 range</li>
          </ul>
          <p className="mt-2 text-xs">Large ranges in raw data suggest high price volatility. Normalized data should have consistent ranges for better ML performance.</p>
        </div>
      )
    },
    std: {
      title: "Standard Deviation",
      content: (
        <div>
          <p className="font-semibold mb-2">📊 Standard Deviation</p>
          <p className="mb-2">Measures how spread out the vector values are from their average.</p>
          <p className="mb-2"><strong>What different values mean:</strong></p>
          <ul className="list-disc list-inside space-y-1 text-xs">
            <li><strong>Low Std Dev (&lt; 1.0):</strong> Values are clustered, low volatility</li>
            <li><strong>Medium Std Dev (1.0-5.0):</strong> Moderate spread, normal market</li>
            <li><strong>High Std Dev (&gt; 5.0):</strong> High volatility, trending market</li>
          </ul>
          <p className="mt-2 text-xs">For normalized data, std dev close to 1.0 indicates proper scaling. For raw price data, higher values suggest more volatile trading periods.</p>
        </div>
      )
    }
  };

  // Heatmap explanation
  const heatmapInfo = {
    content: (
      <div>
        <p className="font-semibold mb-2">🔥 Vector Heatmap Visualization</p>
        <p className="mb-3">Interactive color-coded representation of your trading vector data.</p>
        
        <p className="font-medium mb-2">📖 How to read the heatmap:</p>
        <ul className="list-disc list-inside space-y-1 text-xs mb-3">
          <li><strong>Rows:</strong> Individual candles/time periods (newest to oldest)</li>
          <li><strong>Columns:</strong> Vector dimensions (D0, D1, D2, etc.)</li>
          <li><strong>Colors:</strong> Value intensity from blue (low) to green (mid) to red (high)</li>
          <li><strong>Hover:</strong> See exact values and ranges for each cell</li>
        </ul>

        <p className="font-medium mb-2">🎨 Color coding:</p>
        <ul className="list-disc list-inside space-y-1 text-xs mb-3">
          <li><span className="inline-block w-3 h-3 bg-blue-500 rounded mr-1"></span><strong>Blue:</strong> Values in lower range (bearish/low activity)</li>
          <li><span className="inline-block w-3 h-3 bg-green-500 rounded mr-1"></span><strong>Green:</strong> Values in middle range (neutral/balanced)</li>
          <li><span className="inline-block w-3 h-3 bg-red-500 rounded mr-1"></span><strong>Red:</strong> Values in upper range (bullish/high activity)</li>
        </ul>

        <p className="font-medium mb-2">🔍 What to look for:</p>
        <ul className="list-disc list-inside space-y-1 text-xs">
          <li><strong>Patterns:</strong> Vertical bands indicate correlated dimensions</li>
          <li><strong>Clusters:</strong> Similar colors in rows show market patterns</li>
          <li><strong>Anomalies:</strong> Isolated extreme colors may indicate significant events</li>
          <li><strong>Trends:</strong> Color shifts across rows show market direction changes</li>
        </ul>
      </div>
    )
  };

  const calculateVectorSimilarity = (vec1, vec2) => {
    if (!vec1 || !vec2 || vec1.length !== vec2.length) return 0;
    
    // Calculate cosine similarity
    let dotProduct = 0;
    let norm1 = 0;
    let norm2 = 0;
    
    for (let i = 0; i < vec1.length; i++) {
      dotProduct += vec1[i] * vec2[i];
      norm1 += vec1[i] * vec1[i];
      norm2 += vec2[i] * vec2[i];
    }
    
    if (norm1 === 0 || norm2 === 0) return 0;
    
    const cosineSim = dotProduct / (Math.sqrt(norm1) * Math.sqrt(norm2));
    
    // Add Euclidean distance component for better discrimination
    norm1 = Math.sqrt(norm1);
    norm2 = Math.sqrt(norm2);
    
    let euclideanDist = 0;
    for (let i = 0; i < vec1.length; i++) {
      const diff = (vec1[i] / norm1) - (vec2[i] / norm2);
      euclideanDist += diff * diff;
    }
    euclideanDist = Math.sqrt(euclideanDist);
    
    // Convert distance to similarity
    const maxDistance = Math.sqrt(2);
    const euclideanSim = 1.0 - (euclideanDist / maxDistance);
    
    // Combine similarities (70% cosine, 30% Euclidean)
    const combinedSimilarity = 0.7 * cosineSim + 0.3 * euclideanSim;
    
    // Apply sigmoid-like transformation to enhance differences
    const enhancedSimilarity = Math.tanh(combinedSimilarity * 2) * 0.8;
    
    return Math.max(-1, Math.min(1, enhancedSimilarity));
  };

  const renderVectorComparison = () => {
    if (!vectorData?.data || vectorData.data.length < 2) {
      return <div className={`${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>Need at least 2 vectors for comparison</div>;
    }

    const vectors = compareIndices.map(i => vectorData.data[i]?.[selectedVectorType]).filter(v => v);
    if (vectors.length < 2) return <div className={`${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>Invalid comparison indices</div>;

    const maxDimensions = Math.min(vectors[0]?.length || 0, 15);
    const similarity = calculateVectorSimilarity(vectors[0], vectors[1]);

    return (
      <div className="space-y-4">
        {/* Comparison Controls */}
        <div className="flex items-center justify-between">
          <div className="flex items-center">
          <h4 className={`text-sm font-semibold ${
            isDarkMode ? 'text-gray-300' : 'text-gray-700'
          }`}>Vector Comparison</h4>
            <InfoTooltip id="vector-comparison" content={
              <div>
                <p className="font-semibold mb-2">⚖️ Vector Comparison</p>
                <p className="mb-2">Compare two trading periods side-by-side:</p>
                <ul className="list-disc list-inside space-y-1 text-xs">
                  <li><strong>Select Candles:</strong> Choose any two time periods to compare</li>
                  <li><strong>OHLC Data:</strong> See actual price values for each period</li>
                  <li><strong>Vector Values:</strong> Compare mathematical representations</li>
                  <li><strong>Differences:</strong> Red highlighting shows significant variations</li>
                </ul>
                <p className="mt-2 text-xs">Useful for finding similar market conditions and understanding pattern variations.</p>
              </div>
            } isDarkMode={isDarkMode} />
          </div>
          <div className="flex items-center space-x-4">
            <div className="flex items-center space-x-2">
              <label className={`text-xs ${isDarkMode ? 'text-gray-400' : 'text-gray-600'}`}>Compare:</label>
              <select
                value={compareIndices[0]}
                onChange={(e) => setCompareIndices([parseInt(e.target.value), compareIndices[1]])}
                className={`text-xs px-2 py-1 rounded border ${
                  isDarkMode 
                    ? 'bg-gray-700 border-gray-600 text-gray-300' 
                    : 'bg-white border-gray-300'
                }`}
              >
                {vectorData.data.slice(0, Math.min(rowLimit, vectorData.data.length)).map((candle, i) => (
                  <option key={i} value={i}>
                    {new Date(candle.timestamp).toLocaleDateString()} (${candle.close?.toFixed(2)})
                  </option>
                ))}
              </select>
              <span className={`text-xs ${isDarkMode ? 'text-gray-400' : 'text-gray-600'}`}>vs</span>
              <select
                value={compareIndices[1]}
                onChange={(e) => setCompareIndices([compareIndices[0], parseInt(e.target.value)])}
                className={`text-xs px-2 py-1 rounded border ${
                  isDarkMode 
                    ? 'bg-gray-700 border-gray-600 text-gray-300' 
                    : 'bg-white border-gray-300'
                }`}
              >
                {vectorData.data.slice(0, Math.min(rowLimit, vectorData.data.length)).map((candle, i) => (
                  <option key={i} value={i}>
                    {new Date(candle.timestamp).toLocaleDateString()} (${candle.close?.toFixed(2)})
                  </option>
                ))}
              </select>
            </div>
            <div className="flex items-center">
            <div className={`text-sm px-3 py-1 rounded transition-colors duration-200 ${
              similarity > 0.8 
                ? isDarkMode ? 'bg-green-900/30 text-green-300' : 'bg-green-100 text-green-800'
                : similarity > 0.6 
                ? isDarkMode ? 'bg-yellow-900/30 text-yellow-300' : 'bg-yellow-100 text-yellow-800'
                : isDarkMode ? 'bg-red-900/30 text-red-300' : 'bg-red-100 text-red-800'
            }`}>
              Similarity: {(similarity * 100).toFixed(1)}%
            </div>
              <InfoTooltip id="similarity-score" content={
                <div>
                  <p className="font-semibold mb-2">📈 Similarity Score</p>
                  <p className="mb-2">Cosine similarity between the two selected vectors:</p>
                  <ul className="list-disc list-inside space-y-1 text-xs">
                    <li><strong>80-100%:</strong> <span className="text-green-600">Very Similar</span> - Strong pattern match</li>
                    <li><strong>60-79%:</strong> <span className="text-yellow-600">Moderately Similar</span> - Some correlation</li>
                    <li><strong>0-59%:</strong> <span className="text-red-600">Different</span> - Weak or no correlation</li>
                  </ul>
                  <p className="mt-2 text-xs">Higher scores suggest similar market conditions that might lead to similar future behavior.</p>
                </div>
              } isDarkMode={isDarkMode} />
            </div>
          </div>
        </div>

        {/* Vector Comparison Display */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          {vectors.map((vector, vectorIndex) => {
            const candleData = vectorData.data[compareIndices[vectorIndex]];
            return (
              <div key={vectorIndex} className={`p-4 rounded-lg border transition-colors duration-200 ${
                isDarkMode ? 'bg-gray-700 border-gray-600' : 'bg-white border-gray-200'
              }`}>
                <div className={`text-sm font-medium mb-2 ${
                  isDarkMode ? 'text-gray-200' : 'text-gray-800'
                }`}>
                  Candle #{compareIndices[vectorIndex] + 1}
                </div>
                <div className={`text-xs mb-3 ${
                  isDarkMode ? 'text-gray-400' : 'text-gray-600'
                }`}>
                  {new Date(candleData.timestamp).toLocaleString()}
                </div>
                
                {/* OHLC Info if available */}
                {candleData.open && (
                  <div className={`text-xs mb-3 grid grid-cols-4 gap-2 ${
                    isDarkMode ? 'text-gray-300' : 'text-gray-700'
                  }`}>
                    <div>O: {candleData.open.toFixed(2)}</div>
                    <div>H: {candleData.high.toFixed(2)}</div>
                    <div>L: {candleData.low.toFixed(2)}</div>
                    <div>C: {candleData.close.toFixed(2)}</div>
                  </div>
                )}
                
                {/* Vector Values */}
                <div className="grid grid-cols-5 gap-1 text-xs">
                  {vector.slice(0, maxDimensions).map((value, dimIndex) => {
                    const otherValue = vectors[1 - vectorIndex]?.[dimIndex];
                    const diff = otherValue ? Math.abs(value - otherValue) : 0;
                    const isHighDiff = diff > (Math.abs(value) * 0.1); // 10% difference threshold
                    
                    return (
                      <div key={dimIndex} className={`p-1 rounded text-center font-mono transition-colors duration-200 ${
                        isHighDiff 
                          ? isDarkMode ? 'bg-red-900/30 border border-red-700' : 'bg-red-50 border border-red-200'
                          : isDarkMode ? 'bg-gray-800' : 'bg-gray-50'
                      }`}>
                        <div className={`${isDarkMode ? 'text-gray-400' : 'text-gray-500'} text-xs`}>D{dimIndex}</div>
                        <div className={`font-bold ${
                          value >= 0 ? 'text-green-500' : 'text-red-500'
                        }`}>
                          {value.toFixed(3)}
                        </div>
                        {otherValue && (
                          <div className={`text-xs ${isDarkMode ? 'text-gray-500' : 'text-gray-400'}`}>
                            Δ{(value - otherValue).toFixed(3)}
                          </div>
                        )}
                      </div>
                    );
                  })}
                </div>
                
                {maxDimensions < vector.length && (
                  <div className={`text-xs mt-2 text-center ${isDarkMode ? 'text-gray-500' : 'text-gray-400'}`}>
                    Showing {maxDimensions} of {vector.length} dimensions
                  </div>
                )}
              </div>
            );
          })}
        </div>

        {/* Similarity Analysis */}
        <div className={`p-3 rounded-lg border transition-colors duration-200 ${
          isDarkMode ? 'bg-gray-700 border-gray-600' : 'bg-gray-50 border-gray-200'
        }`}>
          <div className="flex items-center mb-2">
            <h5 className={`text-sm font-medium ${
            isDarkMode ? 'text-gray-300' : 'text-gray-700'
          }`}>Similarity Analysis</h5>
            <InfoTooltip id="similarity-analysis" content={
              <div>
                <p className="font-semibold mb-2">🔍 Vector Similarity</p>
                <p className="mb-2">Cosine similarity measures how similar two trading periods are:</p>
                <ul className="list-disc list-inside space-y-1 text-xs">
                  <li><strong>90-100%:</strong> Very Similar - Almost identical patterns</li>
                  <li><strong>70-89%:</strong> Similar - Strong pattern match</li>
                  <li><strong>50-69%:</strong> Somewhat Similar - Weak correlation</li>
                  <li><strong>0-49%:</strong> Different - No significant pattern match</li>
                </ul>
                <p className="mt-2 text-xs">High similarity suggests similar market conditions and potential future behavior.</p>
              </div>
            } isDarkMode={isDarkMode} />
          </div>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-xs">
            <div>
              <span className={isDarkMode ? 'text-gray-400' : 'text-gray-600'}>Cosine Similarity:</span>
              <div className="font-bold text-blue-500">{(similarity * 100).toFixed(2)}%</div>
            </div>
            <div>
              <span className={isDarkMode ? 'text-gray-400' : 'text-gray-600'}>Interpretation:</span>
              <div className={`font-medium ${
                similarity > 0.9 ? 'text-green-600' :
                similarity > 0.7 ? 'text-yellow-600' :
                similarity > 0.5 ? 'text-orange-600' : 'text-red-600'
              }`}>
                {similarity > 0.9 ? 'Very Similar' :
                 similarity > 0.7 ? 'Similar' :
                 similarity > 0.5 ? 'Somewhat Similar' : 'Different'}
              </div>
            </div>
            <div>
              <span className={isDarkMode ? 'text-gray-400' : 'text-gray-600'}>Pattern Match:</span>
              <div className={`font-medium ${
                similarity > 0.8 ? 'text-green-600' : 'text-gray-500'
              }`}>
                {similarity > 0.8 ? 'Strong pattern match detected' : 'No strong pattern match'}
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  };

  const renderShapeSimilarity = () => {
    if (!vectorData?.data) return null;
    if (!selectedVectorType.includes('iso_')) {
      return (
        <div className={`p-4 rounded-lg border transition-colors duration-200 ${
          isDarkMode 
            ? 'bg-yellow-900/20 border-yellow-800 text-yellow-300' 
            : 'bg-yellow-50 border-yellow-200 text-yellow-800'
        }`}>
          <div className="text-sm font-medium">Shape Similarity Analysis</div>
          <div className="text-xs mt-1">
            Shape similarity analysis is only available for ISO (Isolation Forest) vectors.
          </div>
          <div className="text-xs mt-2">
            Please select an ISO vector type (iso_ohlc or iso_ohlcv) to use this visualization.
          </div>
        </div>
      );
    }

    if (similarityLoading) {
      return (
        <div className="flex items-center justify-center h-32">
          <div className={`animate-spin rounded-full h-8 w-8 border-b-2 ${
            isDarkMode ? 'border-blue-400' : 'border-blue-600'
          }`}></div>
          <span className={`ml-3 ${
            isDarkMode ? 'text-gray-300' : 'text-gray-600'
          }`}>Calculating shape similarities...</span>
        </div>
      );
    }

    if (similarityError && !shapeSimilarityData) {
      return (
        <div className={`p-4 rounded-lg border transition-colors duration-200 ${
          isDarkMode 
            ? 'bg-red-900/20 border-red-800 text-red-300' 
            : 'bg-red-50 border-red-200 text-red-800'
        }`}>
          <div className="text-sm font-medium">Error calculating shape similarity</div>
          <div className="text-xs mt-1">{similarityError}</div>
          <button 
            onClick={fetchShapeSimilarityData}
            className="mt-2 px-3 py-1 bg-red-600 text-white rounded text-sm hover:bg-red-700 transition-colors duration-200"
          >
            Retry
          </button>
        </div>
      );
    }

    if (!shapeSimilarityData) {
      return <div className={`${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>No shape similarity data available</div>;
    }

    const { similarity_matrix } = shapeSimilarityData;
    const { matrix: shapeSimilarityMatrix, candles: dataToShow, statistics } = similarity_matrix;

    // Helper function to get similarity color with proper full-range mapping
    const getSimilarityColor = (similarity) => {
      // Clamp similarity to expected range
      const clampedSim = Math.max(-1, Math.min(1, similarity));
      
      // Special case: 100% similarity (perfect match) - black
      if (clampedSim >= 0.999) {
        return '#000000';
      }
      
      // Create a more intuitive color mapping for the full -1 to +1 range
      if (clampedSim >= 0.7) {
        // High positive similarity: Bright Green
        const intensity = (clampedSim - 0.7) / 0.3; // 0-1 for range 0.7-1.0
        const green = Math.floor(200 + intensity * 55); // 200-255
        return `rgb(0, ${green}, 0)`;
      } else if (clampedSim >= 0.4) {
        // Medium-high positive similarity: Yellow-Green
        const intensity = (clampedSim - 0.4) / 0.3; // 0-1 for range 0.4-0.7
        const red = Math.floor(150 * (1 - intensity)); // 150-0
        const green = 255;
        return `rgb(${red}, ${green}, 0)`;
      } else if (clampedSim >= 0.1) {
        // Low positive similarity: Yellow to Orange
        const intensity = (clampedSim - 0.1) / 0.3; // 0-1 for range 0.1-0.4
        const red = 255;
        const green = Math.floor(100 + intensity * 155); // 100-255
        return `rgb(${red}, ${green}, 0)`;
      } else if (clampedSim >= -0.2) {
        // Near zero similarity: Orange to Red-Orange
        const intensity = (clampedSim + 0.2) / 0.3; // 0-1 for range -0.2-0.1
        const red = 255;
        const green = Math.floor(intensity * 100); // 0-100
        return `rgb(${red}, ${green}, 0)`;
      } else if (clampedSim >= -0.5) {
        // Medium negative similarity: Red to Dark Red
        const intensity = (clampedSim + 0.5) / 0.3; // 0-1 for range -0.5 to -0.2
        const red = Math.floor(180 + intensity * 75); // 180-255
        const green = 0;
        const blue = 0;
        return `rgb(${red}, ${green}, ${blue})`;
      } else {
        // High negative similarity: Very Dark Red to Purple-Red
        const intensity = (clampedSim + 1) / 0.5; // 0-1 for range -1.0 to -0.5
        const red = Math.floor(100 + intensity * 80); // 100-180
        const green = 0;
        const blue = Math.floor(intensity * 50); // 0-50 (slight purple tint)
        return `rgb(${red}, ${green}, ${blue})`;
      }
    };

    // Helper function to handle candle selection
    const handleCandleSelect = (candle, rowIndex) => {
      const candleId = `${selectedSymbol}_${selectedTimeframe}_${candle.timestamp}`;
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
          sourceIndex: rowIndex
        };
        addSelectedCandle(candleWithMetadata);
      }
    };



    // Handle matrix cell click - toggle selection of both candles being compared
    const handleMatrixCellClick = (rowCandle, rowIndex, colIndex) => {
      const colCandle = dataToShow[colIndex];
      
      // Create candle IDs for both candles
      const rowCandleId = `${selectedSymbol}_${selectedTimeframe}_${rowCandle.timestamp}`;
      const colCandleId = `${selectedSymbol}_${selectedTimeframe}_${colCandle.timestamp}`;
      
      // Check if both candles are currently selected
      const rowSelected = selectedCandles.some(c => c.id === rowCandleId);
      const colSelected = selectedCandles.some(c => c.id === colCandleId);
      
      // If both are selected, deselect both
      if (rowSelected && colSelected) {
        removeSelectedCandle(rowCandleId);
        removeSelectedCandle(colCandleId);
      } else {
        // Otherwise, ensure both are selected
        if (!rowSelected) {
          const rowCandleWithMetadata = {
            id: rowCandleId,
            symbol: selectedSymbol,
            timeframe: selectedTimeframe,
            timestamp: rowCandle.timestamp,
            open: rowCandle.open,
            high: rowCandle.high,
            low: rowCandle.low,
            close: rowCandle.close,
            volume: rowCandle.volume,
            change: rowCandle.close - rowCandle.open,
            changePercent: ((rowCandle.close - rowCandle.open) / rowCandle.open) * 100,
            sourceIndex: rowIndex
          };
          addSelectedCandle(rowCandleWithMetadata);
        }
        
        if (!colSelected) {
          const colCandleWithMetadata = {
            id: colCandleId,
            symbol: selectedSymbol,
            timeframe: selectedTimeframe,
            timestamp: colCandle.timestamp,
            open: colCandle.open,
            high: colCandle.high,
            low: colCandle.low,
            close: colCandle.close,
            volume: colCandle.volume,
            change: colCandle.close - colCandle.open,
            changePercent: ((colCandle.close - colCandle.open) / colCandle.open) * 100,
            sourceIndex: colIndex
          };
          addSelectedCandle(colCandleWithMetadata);
        }
      }
    };

    return (
      <div className="space-y-4">
        {/* Shape Similarity Explanation */}
        <div className={`p-4 rounded-lg border transition-colors duration-200 ${
          isDarkMode ? 'bg-blue-900/20 border-blue-800' : 'bg-blue-50 border-blue-200'
        }`}>
          <div className="flex items-center mb-2">
            <h4 className={`text-sm font-semibold ${
              isDarkMode ? 'text-blue-300' : 'text-blue-700'
            }`}>🔍 Shape Similarity Analysis</h4>
            <InfoTooltip id="shape-similarity-explanation" content={
              <div>
                <p className="font-semibold mb-2">🔍 Shape Similarity Matrix</p>
                <p className="mb-2">Visual analysis of candlestick pattern similarity using ISO vectors:</p>
                <ul className="list-disc list-inside space-y-1 text-xs">
                  <li><strong>🟢 Green:</strong> High similarity (70-100%) - Very similar patterns</li>
                  <li><strong>🟡 Yellow-Green:</strong> Good similarity (40-70%) - Notable pattern match</li>
                  <li><strong>🟠 Orange:</strong> Low similarity (10-40%) - Some differences</li>
                  <li><strong>🔴 Red:</strong> Poor similarity (-20-10%) - Different shapes</li>
                  <li><strong>🔴 Dark Red:</strong> Very poor similarity (-100 to -20%) - Opposite patterns</li>
                  <li><strong>Matrix View:</strong> Each cell shows similarity between two candles</li>
                </ul>
                <p className="mt-2 text-xs">ISO vectors capture the geometric shape characteristics of candlesticks, making this perfect for pattern recognition.</p>
              </div>
            } isDarkMode={isDarkMode} />
          </div>
          <div className={`text-xs ${
            isDarkMode ? 'text-blue-200' : 'text-blue-600'
          }`}>
            This matrix shows how similar each candlestick shape is to every other candlestick in the dataset. 
            ISO vectors are specifically designed to capture the geometric properties of price movements.
          </div>
        </div>

        {/* Matrix Dimension Controls */}
        <div className={`p-4 rounded-lg border transition-colors duration-200 ${
          isDarkMode ? 'bg-gray-700 border-gray-600' : 'bg-gray-50 border-gray-200'
        }`}>
          <div className="flex items-center justify-between mb-3">
            <h4 className={`text-sm font-semibold ${
              isDarkMode ? 'text-gray-300' : 'text-gray-700'
            }`}>Matrix Dimensions</h4>
            <div className="flex items-center">
              <InfoTooltip id="matrix-dimensions" content={
                <div>
                  <p className="font-semibold mb-2">📐 Matrix Dimension Controls</p>
                  <p className="mb-2">Customize the size and layout of your similarity matrix:</p>
                  <ul className="list-disc list-inside space-y-1 text-xs">
                    <li><strong>Rows:</strong> Number of candles shown vertically (Y-axis)</li>
                    <li><strong>Columns:</strong> Number of candles shown horizontally (X-axis)</li>
                    <li><strong>Square Matrix:</strong> Rows = Columns (e.g., 50x50)</li>
                    <li><strong>Rectangular Matrix:</strong> Different dimensions (e.g., 10x30)</li>
                    <li><strong>Vector Comparison:</strong> Single row/column (e.g., 1x20)</li>
                  </ul>
                  <p className="mt-2 text-xs">Larger matrices provide more data but may impact performance. Consider your analysis needs.</p>
                </div>
              } isDarkMode={isDarkMode} />
            </div>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <div>
              <label className={`block text-sm font-medium mb-2 ${
                isDarkMode ? 'text-gray-300' : 'text-gray-700'
              }`}>
                Matrix Rows (Y-axis)
              </label>
              <select
                value={matrixRows}
                onChange={(e) => setMatrixRows(Number(e.target.value))}
                className={`w-full rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500 transition-colors duration-200 ${
                  isDarkMode 
                    ? 'border-gray-600 bg-gray-700 text-white' 
                    : 'border-gray-300 bg-white text-gray-900'
                }`}
              >
                <option value={1}>1 row</option>
                <option value={5}>5 rows</option>
                <option value={10}>10 rows</option>
                <option value={15}>15 rows</option>
                <option value={20}>20 rows</option>
                <option value={25}>25 rows</option>
                <option value={30}>30 rows</option>
                <option value={40}>40 rows</option>
                <option value={50}>50 rows</option>
                <option value={75}>75 rows</option>
                <option value={100}>100 rows</option>
              </select>
            </div>
            
            <div>
              <label className={`block text-sm font-medium mb-2 ${
                isDarkMode ? 'text-gray-300' : 'text-gray-700'
              }`}>
                Matrix Columns (X-axis)
              </label>
              <select
                value={matrixCols}
                onChange={(e) => setMatrixCols(Number(e.target.value))}
                className={`w-full rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500 transition-colors duration-200 ${
                  isDarkMode 
                    ? 'border-gray-600 bg-gray-700 text-white' 
                    : 'border-gray-300 bg-white text-gray-900'
                }`}
              >
                <option value={1}>1 column</option>
                <option value={5}>5 columns</option>
                <option value={10}>10 columns</option>
                <option value={15}>15 columns</option>
                <option value={20}>20 columns</option>
                <option value={25}>25 columns</option>
                <option value={30}>30 columns</option>
                <option value={40}>40 columns</option>
                <option value={50}>50 columns</option>
                <option value={75}>75 columns</option>
                <option value={100}>100 columns</option>
              </select>
            </div>
            
            <div>
              <label className={`block text-sm font-medium mb-2 ${
                isDarkMode ? 'text-gray-300' : 'text-gray-700'
              }`}>
                Quick Presets
              </label>
              <select
                onChange={(e) => {
                  const [rows, cols] = e.target.value.split('x').map(Number);
                  if (rows && cols) {
                    setMatrixRows(rows);
                    setMatrixCols(cols);
                  }
                }}
                className={`w-full rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500 transition-colors duration-200 ${
                  isDarkMode 
                    ? 'border-gray-600 bg-gray-700 text-white' 
                    : 'border-gray-300 bg-white text-gray-900'
                }`}
              >
                <option value="">Select preset...</option>
                <option value="1x5">1x5 (Vector)</option>
                <option value="1x10">1x10 (Vector)</option>
                <option value="1x20">1x20 (Vector)</option>
                <option value="5x5">5x5 (Small)</option>
                <option value="10x10">10x10 (Medium)</option>
                <option value="20x20">20x20 (Default)</option>
                <option value="30x30">30x30 (Large)</option>
                <option value="50x50">50x50 (XL)</option>
                <option value="100x100">100x100 (XXL)</option>
                <option value="10x50">10x50 (Wide)</option>
                <option value="50x10">50x10 (Tall)</option>
              </select>
            </div>
            
            <div className="flex items-end">
              <button
                onClick={() => {
                  setMatrixRows(20);
                  setMatrixCols(20);
                }}
                className={`w-full px-3 py-2 text-sm rounded-md transition-colors duration-200 ${
                  isDarkMode 
                    ? 'bg-gray-600 text-gray-200 hover:bg-gray-500' 
                    : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                }`}
              >
                Reset to 20x20
              </button>
            </div>
          </div>
          
          <div className={`mt-3 p-2 rounded text-xs ${
            isDarkMode ? 'bg-gray-800 text-gray-300' : 'bg-white text-gray-600'
          }`}>
            <div className="flex items-center justify-between">
              <div>
                <strong>Current Matrix:</strong> {matrixRows} × {matrixCols} = {matrixRows * matrixCols} cells
                {matrixRows === 1 || matrixCols === 1 ? ' (Vector comparison mode)' : ' (Full similarity matrix)'}
                {matrixRows * matrixCols > 2500 && (
                  <span className="text-orange-500 font-medium"> • Large matrix - may impact performance</span>
                )}
              </div>
              <div className="flex items-center space-x-2">
                <span className="text-gray-500">Cell size:</span>
                <div 
                  className={`${cellSize.width} ${cellSize.height} border-2 border-gray-400 rounded-sm`}
                  style={{ backgroundColor: '#4ade80' }}
                  title={`Matrix cells will be ${cellSize.size} × ${cellSize.size}`}
                />
                <span className="font-mono">{cellSize.size}</span>
              </div>
            </div>
          </div>
        </div>

        {/* Similarity Matrix */}
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4 text-xs">
              <span className={isDarkMode ? 'text-gray-300' : 'text-gray-600'}>Similarity Scale:</span>
              <div className="flex items-center space-x-2 text-xs flex-wrap">
                <div className="w-4 h-4 rounded" style={{ backgroundColor: 'rgb(150, 0, 25)' }}></div>
                <span className={isDarkMode ? 'text-gray-400' : 'text-gray-500'}>Very Poor (-100 to -20%)</span>
                <div className="w-4 h-4 rounded" style={{ backgroundColor: 'rgb(255, 0, 0)' }}></div>
                <span className={isDarkMode ? 'text-gray-400' : 'text-gray-500'}>Poor (-20 to 10%)</span>
                <div className="w-4 h-4 rounded" style={{ backgroundColor: 'rgb(255, 180, 0)' }}></div>
                <span className={isDarkMode ? 'text-gray-400' : 'text-gray-500'}>Low (10-40%)</span>
                <div className="w-4 h-4 rounded" style={{ backgroundColor: 'rgb(100, 255, 0)' }}></div>
                <span className={isDarkMode ? 'text-gray-400' : 'text-gray-500'}>Good (40-70%)</span>
                <div className="w-4 h-4 rounded" style={{ backgroundColor: 'rgb(0, 220, 0)' }}></div>
                <span className={isDarkMode ? 'text-gray-400' : 'text-gray-500'}>High (70-99%)</span>
                <div className="w-4 h-4 rounded border-2 border-white" style={{ backgroundColor: '#000000' }}></div>
                <span className={`font-semibold ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>Perfect (100%)</span>
              </div>
            </div>
            <div className="text-xs">
              <span className={isDarkMode ? 'text-gray-400' : 'text-gray-500'}>
                Showing {Math.min(matrixRows, dataToShow.length)} × {Math.min(matrixCols, dataToShow.length)} matrix • Click cells to select candles
              </span>
            </div>
          </div>

          {/* Matrix Headers */}
          <div className="flex items-center space-x-1 text-xs">
            <div className={`${matrixCols > 50 ? 'w-12' : 'w-20'} ${isDarkMode ? 'text-gray-400' : 'text-gray-600'}`}>
              {matrixCols > 50 ? 'C#' : 'Candle'}
            </div>
            <div className={`flex ${matrixCols > 80 ? 'space-x-0' : 'space-x-0.5'} flex-wrap`}>
              {dataToShow.slice(0, Math.min(matrixCols, dataToShow.length)).map((candle, i) => (
                <div key={i} className={`${cellSize.width} text-center text-xs ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>
                  {matrixCols <= 20 ? i + 1 : 
                   matrixCols <= 40 ? (i % 5 === 0 ? i + 1 : '•') :
                   matrixCols <= 80 ? (i % 10 === 0 ? i + 1 : '') : ''}
                </div>
              ))}
            </div>
          </div>

          {/* Similarity Matrix Grid - NO SCROLLING */}
          <div className="space-y-1">
            {dataToShow.slice(0, Math.min(matrixRows, dataToShow.length)).map((candle, rowIndex) => {
              const candleId = `${selectedSymbol}_${selectedTimeframe}_${candle.timestamp}`;
              const isSelected = selectedCandles.some(c => c.id === candleId);
              
              return (
                <div key={rowIndex} className="flex items-center space-x-1">
                  <div 
                    className={`${matrixCols > 50 ? 'w-12' : 'w-20'} text-xs font-mono cursor-pointer rounded transition-all duration-200 ${
                      isSelected 
                        ? isDarkMode 
                          ? 'bg-blue-900/40 text-blue-300' 
                          : 'bg-blue-50 text-blue-700'
                        : isDarkMode 
                          ? 'text-gray-300 hover:bg-gray-700' 
                          : 'text-gray-700 hover:bg-gray-50'
                    }`}
                    onClick={() => handleCandleSelect(candle, rowIndex)}
                    title={`Candle ${rowIndex + 1}: ${new Date(candle.timestamp).toLocaleDateString()}`}
                  >
                    {matrixCols > 50 ? (
                      <div className="truncate">#{rowIndex + 1}</div>
                    ) : (
                      <>
                        <div className="truncate">#{rowIndex + 1}</div>
                        <div className="text-xs opacity-75">${candle.close?.toFixed(2)}</div>
                      </>
                    )}
                  </div>
                  <div className={`flex ${matrixCols > 80 ? 'space-x-0' : 'space-x-0.5'} flex-wrap`}>
                    {shapeSimilarityMatrix[rowIndex]?.slice(0, Math.min(matrixCols, dataToShow.length)).map((similarity, colIndex) => {
                      const color = getSimilarityColor(similarity);
                      const isDiagonal = rowIndex === colIndex;
                      const isPerfectMatch = similarity >= 0.999;
                      
                      return (
                        <div
                          key={colIndex}
                          className={`${cellSize.width} ${cellSize.height} rounded-sm transition-transform hover:scale-110 cursor-pointer ${
                            isPerfectMatch 
                              ? 'border-2 border-white' 
                              : isDiagonal 
                                ? 'border border-gray-600' 
                                : 'border border-gray-300'
                          }`}
                          style={{ 
                            backgroundColor: color,
                            minWidth: cellSize.size,
                            minHeight: cellSize.size
                          }}
                          title={`Similarity: ${(similarity * 100).toFixed(1)}%${isPerfectMatch ? ' (Perfect Match!)' : ''}\nCandle ${rowIndex + 1} vs Candle ${colIndex + 1}\n${new Date(candle.timestamp).toLocaleDateString()} vs ${new Date(dataToShow[colIndex].timestamp).toLocaleDateString()}`}
                          onClick={() => handleMatrixCellClick(candle, rowIndex, colIndex)}
                        />
                      );
                    })}
                  </div>
                  {isSelected && (
                    <div className="ml-2 text-blue-500 text-xs">✓</div>
                  )}
                </div>
              );
            })}
          </div>

          {dataToShow.length < Math.max(matrixRows, matrixCols) && (
            <div className={`text-xs ${isDarkMode ? 'text-orange-400' : 'text-orange-600'} text-center mt-2 p-2 rounded ${
              isDarkMode ? 'bg-orange-900/20' : 'bg-orange-50'
            }`}>
              ⚠️ Only {dataToShow.length} candles available, but matrix is set to {matrixRows}×{matrixCols}. 
              Consider reducing matrix size or increasing data limit.
            </div>
          )}
        </div>

        {/* Shape Similarity Statistics */}
        <div className={`p-4 rounded-lg border transition-colors duration-200 ${
          isDarkMode ? 'bg-gray-700 border-gray-600' : 'bg-gray-50 border-gray-200'
        }`}>
          <h5 className={`text-sm font-medium mb-3 ${
            isDarkMode ? 'text-gray-300' : 'text-gray-700'
          }`}>Shape Similarity Statistics</h5>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-xs">
            <div>
              <span className={isDarkMode ? 'text-gray-400' : 'text-gray-600'}>Average Similarity:</span>
              <div className="font-bold text-blue-500">
                {statistics.average_similarity ? (statistics.average_similarity * 100).toFixed(1) + '%' : 'N/A'}
              </div>
            </div>
            <div>
              <span className={isDarkMode ? 'text-gray-400' : 'text-gray-600'}>Max Similarity:</span>
              <div className={`font-medium ${
                statistics.max_similarity > 0.8 ? 'text-green-600' : 'text-yellow-600'
              }`}>
                {statistics.max_similarity ? (statistics.max_similarity * 100).toFixed(1) + '%' : 'N/A'}
              </div>
            </div>
            <div>
              <span className={isDarkMode ? 'text-gray-400' : 'text-gray-600'}>Pattern Diversity:</span>
              <div className={`font-medium ${
                statistics.pattern_diversity === 'High Diversity' ? 'text-green-600' : 
                statistics.pattern_diversity === 'Medium Diversity' ? 'text-yellow-600' : 'text-red-600'
              }`}>
                {statistics.pattern_diversity || 'Unknown'}
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  };

  const renderVectorStats = () => {
    if (!vectorData?.data) return null;

    try {
      const vectors = vectorData.data.map(row => row[selectedVectorType]).filter(v => v);
      if (vectors.length === 0) {
        return <div className={`${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>No vector data available for {selectedVectorType}</div>;
      }

      const stats = getVectorStats(vectors);
      if (!stats) return <div className={`${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>No vector statistics available</div>;
    
      // Add some validation for the stats
      if (typeof stats.min !== 'number' || typeof stats.max !== 'number' || 
          typeof stats.avg !== 'number' || typeof stats.std !== 'number') {
        return <div className={`${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>Invalid vector statistics calculated</div>;
      }

      return (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className={`p-4 rounded-lg border transition-colors duration-200 ${
            isDarkMode ? 'bg-gray-700 border-gray-600' : 'bg-white border-gray-200'
          }`}>
            <div className={`flex items-center text-xs mb-1 ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>
              Vector Count
              <InfoTooltip id="vector-count" content={vectorStatsInfo.count.content} isDarkMode={isDarkMode} />
            </div>
            <div className="text-2xl font-bold text-blue-500">{stats.count}</div>
          </div>
          <div className={`p-4 rounded-lg border transition-colors duration-200 ${
            isDarkMode ? 'bg-gray-700 border-gray-600' : 'bg-white border-gray-200'
          }`}>
            <div className={`flex items-center text-xs mb-1 ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>
              Dimensions
              <InfoTooltip id="vector-dimensions" content={vectorStatsInfo.dimensions.content} isDarkMode={isDarkMode} />
            </div>
            <div className="text-2xl font-bold text-green-500">{stats.dimensions}</div>
          </div>
          <div className={`p-4 rounded-lg border transition-colors duration-200 ${
            isDarkMode ? 'bg-gray-700 border-gray-600' : 'bg-white border-gray-200'
          }`}>
            <div className={`flex items-center text-xs mb-1 ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>
              Range
              <InfoTooltip id="vector-range" content={vectorStatsInfo.range.content} isDarkMode={isDarkMode} />
            </div>
            <div className="text-sm font-bold text-purple-500">
              {stats.min.toFixed(3)} to {stats.max.toFixed(3)}
            </div>
          </div>
          <div className={`p-4 rounded-lg border transition-colors duration-200 ${
            isDarkMode ? 'bg-gray-700 border-gray-600' : 'bg-white border-gray-200'
          }`}>
            <div className={`flex items-center text-xs mb-1 ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>
              Std Dev
              <InfoTooltip id="vector-std" content={vectorStatsInfo.std.content} isDarkMode={isDarkMode} />
            </div>
            <div className="text-2xl font-bold text-orange-500">{stats.std.toFixed(4)}</div>
          </div>
        </div>
      );
    } catch (error) {
      console.error('Error rendering vector stats:', error);
      return (
        <div className={`p-4 rounded-lg border transition-colors duration-200 ${
          isDarkMode 
            ? 'bg-red-900/20 border-red-800 text-red-300' 
            : 'bg-red-50 border-red-200 text-red-800'
        }`}>
          <div className="text-sm font-medium">Error calculating vector statistics</div>
          <div className="text-xs mt-1">
            {error.message || 'An unknown error occurred while processing vector data'}
          </div>
          <div className="text-xs mt-2 opacity-75">
            This may happen with very large vectors (like BERT embeddings). Try reducing the vector limit or switching to a different vector type.
          </div>
        </div>
      );
    }
  };

  if (loading) {
      return (
      <div className="flex items-center justify-center h-64">
        <div className={`animate-spin rounded-full h-12 w-12 border-b-2 ${
          isDarkMode ? 'border-blue-400' : 'border-blue-600'
        }`}></div>
        <span className={`ml-3 ${
          isDarkMode ? 'text-gray-300' : 'text-gray-600'
        }`}>Loading vector data...</span>
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
          <div className="text-red-500 text-sm font-medium">Error loading vector data</div>
        </div>
        <div className="text-red-500 text-sm mt-1">{error}</div>
        <button 
          onClick={fetchVectorData}
          className="mt-2 px-3 py-1 bg-red-600 text-white rounded text-sm hover:bg-red-700 transition-colors duration-200"
        >
          Retry
        </button>
      </div>
    );
  }

      return (
      <div className="space-y-6">
        {/* Vector Dashboard Header */}
        <div className={`p-6 rounded-lg transition-colors duration-200 ${
          isDarkMode 
            ? 'bg-gradient-to-r from-purple-800 via-purple-700 to-blue-800 text-white' 
            : 'bg-gradient-to-r from-purple-600 to-blue-600 text-white'
        }`}>
          <div className="flex items-center">
          <h2 className="text-3xl font-black tracking-tight mb-2" style={{ 
            fontFamily: 'system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
            letterSpacing: '-0.02em',
            textShadow: '0 2px 4px rgba(0,0,0,0.3)'
          }}>
            Day<span className="text-purple-200">gent</span> <span className="text-xl font-semibold text-purple-100">Vector Intelligence</span>
          </h2>
            <InfoTooltip id="vector-dashboard" content={
              <div>
                <p className="font-semibold mb-2">🧠 Vector Dashboard Overview</p>
                <p className="mb-2">Advanced AI-powered analysis of your trading data:</p>
                <ul className="list-disc list-inside space-y-1 text-xs">
                  <li><strong>Mathematical Vectors:</strong> Convert price data into mathematical representations</li>
                  <li><strong>Pattern Recognition:</strong> Find similar market conditions using AI</li>
                  <li><strong>Multiple Types:</strong> Raw prices, normalized data, and BERT embeddings</li>
                  <li><strong>Visual Analysis:</strong> Heatmaps and comparisons for deeper insights</li>
                </ul>
                <p className="mt-2 text-xs">Perfect for quantitative analysis, backtesting, and discovering hidden market patterns.</p>
              </div>
            } isDarkMode={isDarkMode} />
          </div>
          <p className={isDarkMode ? 'text-purple-200' : 'text-purple-100'}>
            Advanced pattern recognition & ML analysis • AI-powered market intelligence
          </p>
          <div className={`mt-4 text-sm rounded p-2 ${
            isDarkMode ? 'bg-black/30' : 'bg-white/20'
          }`}>
            📊 {selectedSymbol?.toUpperCase()} • {selectedTimeframe} • {vectorData?.data?.length || 0} vectors loaded
          </div>
        </div>

        {/* Data Selection & Controls - Using Consistent Component */}
        <DataSelectionControls 
          selectedSymbol={selectedSymbol}
          setSelectedSymbol={setSelectedSymbol}
          selectedTimeframe={selectedTimeframe}
          setSelectedTimeframe={setSelectedTimeframe}
          rowLimit={rowLimit}
          setRowLimit={setRowLimit}
          sortOrder={sortOrder}
          setSortOrder={setSortOrder}
          handleRefresh={fetchVectorData}
          isDarkMode={isDarkMode}
          dashboardType="vector"
          // Date range props
          fetchMode={fetchMode}
          setFetchMode={setFetchMode}
          dateRangeType={dateRangeType}
          setDateRangeType={setDateRangeType}
          startDate={startDate}
          setStartDate={setStartDate}
          endDate={endDate}
          setEndDate={setEndDate}
        />

        {/* Selected Candles Panel */}
        <SelectedCandlesPanel 
          isDarkMode={isDarkMode} 
          canSelectCandles={true}
        />

      {/* Vector Type Selector */}
      <div className={`p-6 rounded-lg border transition-colors duration-200 ${
        isDarkMode 
          ? 'bg-gray-800 border-gray-700 text-white' 
          : 'bg-white border-gray-200'
      }`}>
        <div className="flex items-center mb-4">
          <h3 className={`text-lg font-semibold ${
          isDarkMode ? 'text-white' : 'text-gray-900'
        }`}>Vector Type Selection</h3>
          <InfoTooltip id="vector-types" content={
            <div>
              <p className="font-semibold mb-2">🧠 Vector Types Explained</p>
              <p className="mb-2">Different ways to represent trading data as mathematical vectors:</p>
              <ul className="list-disc list-inside space-y-1 text-xs">
                <li><strong>Raw OHLC:</strong> Direct price values (4 dimensions)</li>
                <li><strong>Raw OHLCV:</strong> Price + volume (5 dimensions)</li>
                <li><strong>Normalized:</strong> Z-score scaled for pattern matching</li>
                <li><strong>BERT:</strong> AI semantic embeddings (384 dimensions)</li>
              </ul>
              <p className="mt-2 text-xs">Choose based on your analysis needs - raw for price analysis, normalized for pattern matching, BERT for semantic similarity.</p>
            </div>
          } isDarkMode={isDarkMode} />
        </div>
        
        {/* Status Information */}
        {(availableVectors.length > 0 || missingVectors.length > 0) && (
          <div className={`mb-4 p-3 rounded-lg transition-colors duration-200 ${
            isDarkMode ? 'bg-gray-700' : 'bg-gray-50'
          }`}>
            <div className="flex items-center justify-between">
            <div className="text-sm">
              <span className="text-green-500 font-medium">✅ Available: {availableVectors.length}</span>
              {missingVectors.length > 0 && (
                <span className="text-orange-500 font-medium ml-4">⚠️ Missing: {missingVectors.length}</span>
              )}
              </div>
              <InfoTooltip id="vector-status" content={
                <div>
                  <p className="font-semibold mb-2">📋 Vector Status</p>
                  <p className="mb-2">Availability of different vector types in your database:</p>
                  <ul className="list-disc list-inside space-y-1 text-xs">
                    <li><strong>✅ Available:</strong> Vector types ready for analysis</li>
                    <li><strong>⚠️ Missing:</strong> Vector types not yet computed</li>
                    <li><strong>Generate Missing:</strong> Use compute.py script to create missing vectors</li>
                  </ul>
                  <p className="mt-2 text-xs">Missing vectors don't affect available ones - you can still analyze what's available.</p>
                </div>
              } isDarkMode={isDarkMode} />
            </div>
            {missingVectors.length > 0 && (
              <div className={`text-xs mt-1 ${
                isDarkMode ? 'text-gray-300' : 'text-gray-600'
              }`}>
                Missing vectors can be generated using the compute.py script
              </div>
            )}
            
            {/* Debug Info - Show actual available columns */}
            {vectorData?.data?.[0] && (
              <details className="mt-2">
                <summary className={`text-xs cursor-pointer transition-colors duration-200 ${
                  isDarkMode ? 'text-blue-400 hover:text-blue-300' : 'text-blue-600 hover:text-blue-800'
                }`}>
                  🔍 Debug: Show all available columns
                </summary>
                <div className={`mt-2 p-2 rounded text-xs font-mono transition-colors duration-200 ${
                  isDarkMode ? 'bg-gray-900 text-gray-300' : 'bg-white text-gray-700'
                }`}>
                  <div className="text-green-500 mb-1">✅ Available columns:</div>
                  {Object.keys(vectorData.data[0])
                    .filter(key => vectorData.data[0][key] !== null && vectorData.data[0][key] !== undefined)
                    .map(key => (
                      <div key={key} className="ml-2">
                        • {key} ({Array.isArray(vectorData.data[0][key]) ? `array[${vectorData.data[0][key].length}]` : typeof vectorData.data[0][key]})
                      </div>
                    ))
                  }
                </div>
              </details>
            )}
          </div>
        )}
        
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Without Volume Column */}
          <div>
            <div className={`text-center mb-4 pb-2 border-b ${
              isDarkMode ? 'border-gray-600' : 'border-gray-200'
            }`}>
              <div className="flex items-center justify-center">
              <h4 className={`text-lg font-semibold ${
                isDarkMode ? 'text-blue-300' : 'text-blue-700'
              }`}>📊 Without Volume</h4>
                <InfoTooltip id="without-volume" content={
                  <div>
                    <p className="font-semibold mb-2">📊 OHLC Vectors (Without Volume)</p>
                    <p className="mb-2">Price-only vectors focusing on market movement patterns:</p>
                    <ul className="list-disc list-inside space-y-1 text-xs">
                      <li><strong>Best for:</strong> Price pattern analysis, trend detection</li>
                      <li><strong>Faster processing:</strong> Fewer dimensions (4 vs 5)</li>
                      <li><strong>Ideal when:</strong> Volume data is unreliable or unavailable</li>
                      <li><strong>Use cases:</strong> Technical analysis, candlestick patterns</li>
                    </ul>
                  </div>
                } isDarkMode={isDarkMode} />
              </div>
              <p className={`text-xs mt-1 ${
                isDarkMode ? 'text-gray-400' : 'text-gray-600'
              }`}>OHLC (Open, High, Low, Close) only</p>
            </div>
            <div className="space-y-3">
              {allVectorTypes.filter(type => !type.key.includes('ohlcv')).map(type => {
                const isAvailable = availableVectors.find(v => v.key === type.key);
                const isMissing = missingVectors.find(v => v.key === type.key);
                
                return (
                  <div key={type.key} className="relative">
                  <button
                    onClick={() => isAvailable && setSelectedVectorType(type.key)}
                    disabled={!isAvailable}
                    className={`w-full p-3 rounded-lg border text-left transition-all duration-200 ${
                      selectedVectorType === type.key 
                        ? isDarkMode
                          ? 'border-blue-400 bg-blue-900/30 ring-2 ring-blue-400/50 text-white'
                          : 'border-blue-500 bg-blue-50 ring-2 ring-blue-200'
                        : isAvailable
                        ? isDarkMode
                          ? 'border-gray-600 hover:border-gray-500 hover:bg-gray-700 text-white'
                          : 'border-gray-200 hover:border-gray-300 hover:bg-gray-50'
                        : isDarkMode
                          ? 'border-gray-700 bg-gray-800 opacity-60 cursor-not-allowed text-gray-500'
                          : 'border-gray-100 bg-gray-50 opacity-60 cursor-not-allowed'
                    }`}
                  >
                      <div className="flex items-center justify-between mb-1">
                        <div className="flex items-center space-x-2">
                      <div className={`w-3 h-3 rounded-full ${type.color} ${!isAvailable ? 'opacity-50' : ''}`}></div>
                      <div className={`font-medium text-sm ${
                        !isAvailable 
                          ? isDarkMode ? 'text-gray-500' : 'text-gray-400'
                          : isDarkMode ? 'text-white' : 'text-gray-900'
                      }`}>
                        {type.name}
                        {isAvailable && <span className="text-green-500 ml-1">✓</span>}
                        {isMissing && <span className="text-orange-500 ml-1">⚠</span>}
                      </div>
                        </div>
                        {isAvailable && (
                          <InfoTooltip id={`vector-type-${type.key}`} content={
                            <div>
                              <p className="font-semibold mb-2">🎯 {type.name}</p>
                              <p className="mb-2">{type.description}</p>
                              <ul className="list-disc list-inside space-y-1 text-xs">
                                {type.key.includes('raw') && (
                                  <>
                                    <li><strong>Raw Values:</strong> Actual price numbers from market data</li>
                                    <li><strong>Best for:</strong> Price analysis, absolute value comparisons</li>
                                    <li><strong>Dimensions:</strong> 4 (Open, High, Low, Close)</li>
                                  </>
                                )}
                                {type.key.includes('norm') && (
                                  <>
                                    <li><strong>Normalized:</strong> Z-score standardized values (mean=0, std=1)</li>
                                    <li><strong>Best for:</strong> Pattern recognition across different price levels</li>
                                    <li><strong>Dimensions:</strong> 4 (Open, High, Low, Close)</li>
                                  </>
                                )}
                                {type.key.includes('BERT') && (
                                  <>
                                    <li><strong>BERT Embeddings:</strong> AI semantic representation of price patterns</li>
                                    <li><strong>Best for:</strong> Advanced pattern matching and similarity analysis</li>
                                    <li><strong>Dimensions:</strong> 384 (semantic features)</li>
                                  </>
                                )}
                                {type.key.includes('iso_') && (
                                  <>
                                    <li><strong>Isolation Forest:</strong> Anomaly detection feature vectors</li>
                                    <li><strong>Best for:</strong> Detecting unusual market conditions and outliers</li>
                                    <li><strong>Dimensions:</strong> Varies based on isolation tree features</li>
                                  </>
                                )}
                              </ul>
                              <p className="mt-2 text-xs">Click to select this vector type for analysis.</p>
                            </div>
                          } isDarkMode={isDarkMode} asSpan={true} />
                        )}
                    </div>
                    <div className={`text-xs ${
                      !isAvailable 
                        ? isDarkMode ? 'text-gray-500' : 'text-gray-400'
                        : isDarkMode ? 'text-gray-300' : 'text-gray-600'
                    }`}>
                      {type.description}
                      {isMissing && <div className="text-orange-500 mt-1">Not available in database</div>}
                    </div>
                  </button>
                  </div>
                );
              })}
            </div>
          </div>

          {/* With Volume Column */}
          <div>
            <div className={`text-center mb-4 pb-2 border-b ${
              isDarkMode ? 'border-gray-600' : 'border-gray-200'
            }`}>
              <div className="flex items-center justify-center">
              <h4 className={`text-lg font-semibold ${
                isDarkMode ? 'text-green-300' : 'text-green-700'
              }`}>📈 With Volume</h4>
                <InfoTooltip id="with-volume" content={
                  <div>
                    <p className="font-semibold mb-2">📈 OHLCV Vectors (With Volume)</p>
                    <p className="mb-2">Complete market data including volume for deeper insights:</p>
                    <ul className="list-disc list-inside space-y-1 text-xs">
                      <li><strong>Best for:</strong> Volume analysis, market strength detection</li>
                      <li><strong>More complete:</strong> Includes trading activity (volume)</li>
                      <li><strong>Ideal for:</strong> Institutional analysis, liquidity assessment</li>
                      <li><strong>Use cases:</strong> Volume-price analysis, market making, trend confirmation</li>
                    </ul>
                    <p className="mt-2 text-xs">Volume data helps confirm price movements and detect market manipulation.</p>
                  </div>
                } isDarkMode={isDarkMode} />
              </div>
              <p className={`text-xs mt-1 ${
                isDarkMode ? 'text-gray-400' : 'text-gray-600'
              }`}>OHLCV (Open, High, Low, Close, Volume)</p>
            </div>
                         <div className="space-y-3">
               {allVectorTypes.filter(type => type.key.includes('ohlcv')).map(type => {
                const isAvailable = availableVectors.find(v => v.key === type.key);
                const isMissing = missingVectors.find(v => v.key === type.key);
                
                return (
                  <div key={type.key} className="relative">
                  <button
                    onClick={() => isAvailable && setSelectedVectorType(type.key)}
                    disabled={!isAvailable}
                    className={`w-full p-3 rounded-lg border text-left transition-all duration-200 ${
                      selectedVectorType === type.key 
                        ? isDarkMode
                          ? 'border-blue-400 bg-blue-900/30 ring-2 ring-blue-400/50 text-white'
                          : 'border-blue-500 bg-blue-50 ring-2 ring-blue-200'
                        : isAvailable
                        ? isDarkMode
                          ? 'border-gray-600 hover:border-gray-500 hover:bg-gray-700 text-white'
                          : 'border-gray-200 hover:border-gray-300 hover:bg-gray-50'
                        : isDarkMode
                          ? 'border-gray-700 bg-gray-800 opacity-60 cursor-not-allowed text-gray-500'
                          : 'border-gray-100 bg-gray-50 opacity-60 cursor-not-allowed'
                    }`}
                  >
                      <div className="flex items-center justify-between mb-1">
                        <div className="flex items-center space-x-2">
                      <div className={`w-3 h-3 rounded-full ${type.color} ${!isAvailable ? 'opacity-50' : ''}`}></div>
                      <div className={`font-medium text-sm ${
                        !isAvailable 
                          ? isDarkMode ? 'text-gray-500' : 'text-gray-400'
                          : isDarkMode ? 'text-white' : 'text-gray-900'
                      }`}>
                        {type.name}
                        {isAvailable && <span className="text-green-500 ml-1">✓</span>}
                        {isMissing && <span className="text-orange-500 ml-1">⚠</span>}
                      </div>
                        </div>
                        {isAvailable && (
                          <InfoTooltip id={`vector-type-volume-${type.key}`} content={
                            <div>
                              <p className="font-semibold mb-2">🎯 {type.name}</p>
                              <p className="mb-2">{type.description}</p>
                              <ul className="list-disc list-inside space-y-1 text-xs">
                                {type.key.includes('raw') && (
                                  <>
                                    <li><strong>Raw Values:</strong> Actual price + volume numbers from market data</li>
                                    <li><strong>Best for:</strong> Volume analysis, institutional flow detection</li>
                                    <li><strong>Dimensions:</strong> 5 (Open, High, Low, Close, Volume)</li>
                                  </>
                                )}
                                {type.key.includes('norm') && (
                                  <>
                                    <li><strong>Normalized:</strong> Z-score standardized OHLCV values</li>
                                    <li><strong>Best for:</strong> Volume-weighted pattern recognition</li>
                                    <li><strong>Dimensions:</strong> 5 (Open, High, Low, Close, Volume)</li>
                                  </>
                                )}
                                {type.key.includes('BERT') && (
                                  <>
                                    <li><strong>BERT Embeddings:</strong> AI semantic representation including volume</li>
                                    <li><strong>Best for:</strong> Advanced pattern matching with volume context</li>
                                    <li><strong>Dimensions:</strong> 384 (semantic features)</li>
                                  </>
                                )}
                                {type.key.includes('iso_') && (
                                  <>
                                    <li><strong>Isolation Forest:</strong> Anomaly detection with volume features</li>
                                    <li><strong>Best for:</strong> Detecting unusual volume patterns and market anomalies</li>
                                    <li><strong>Dimensions:</strong> Varies based on isolation tree features</li>
                                  </>
                                )}
                              </ul>
                              <p className="mt-2 text-xs">Volume data adds depth for institutional analysis and market strength detection.</p>
                            </div>
                          } isDarkMode={isDarkMode} asSpan={true} />
                        )}
                    </div>
                    <div className={`text-xs ${
                      !isAvailable 
                        ? isDarkMode ? 'text-gray-500' : 'text-gray-400'
                        : isDarkMode ? 'text-gray-300' : 'text-gray-600'
                    }`}>
                      {type.description}
                      {isMissing && <div className="text-orange-500 mt-1">Not available in database</div>}
                    </div>
                  </button>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      </div>

      {/* Vector Statistics */}
      <div className={`p-6 rounded-lg border transition-colors duration-200 ${
        isDarkMode 
          ? 'bg-gray-800 border-gray-700' 
          : 'bg-white border-gray-200'
      }`}>
        <div className="flex items-center justify-between mb-4">
          <h3 className={`text-lg font-semibold ${
            isDarkMode ? 'text-white' : 'text-gray-900'
          }`}>Vector Statistics</h3>
          <div className={`text-xs px-3 py-1 rounded-full ${
            isDarkMode ? 'bg-blue-900/30 text-blue-300' : 'bg-blue-100 text-blue-700'
          }`}>
            {selectedVectorType.replace(/_/g, ' ').toUpperCase()}
          </div>
        </div>
        
        {/* Quick Explanation */}
        <div className={`text-xs mb-4 p-3 rounded ${
          isDarkMode ? 'bg-gray-700 text-gray-300' : 'bg-gray-50 text-gray-600'
        }`}>
          <strong>What this shows:</strong> Statistical analysis of {vectorData?.data?.length || 0} trading vectors of type "{selectedVectorType}". 
          {selectedVectorType.includes('raw') && ' Raw vectors contain direct OHLC price values.'}
          {selectedVectorType.includes('norm') && ' Normalized vectors use z-score scaling for pattern matching across different price levels.'}
          {selectedVectorType.includes('BERT') && ' BERT vectors are semantic embeddings (384 dimensions) created by converting price data to natural language.'}
          {selectedVectorType.includes('iso_') && ' ISO vectors are isolation forest features designed to detect anomalies and unusual market conditions.'}
        </div>
        
        {renderVectorStats()}
      </div>

      {/* View Mode Selector */}
      <div className={`p-6 rounded-lg border transition-colors duration-200 ${
        isDarkMode 
          ? 'bg-gray-800 border-gray-700' 
          : 'bg-white border-gray-200'
      }`}>
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center">
          <h3 className={`text-lg font-semibold ${
            isDarkMode ? 'text-white' : 'text-gray-900'
          }`}>Vector Visualization</h3>
            <InfoTooltip id="visualization-modes" content={
              <div>
                <p className="font-semibold mb-2">👁️ Visualization Modes</p>
                <p className="mb-2">Different ways to analyze your vector data:</p>
                <ul className="list-disc list-inside space-y-1 text-xs">
                  <li><strong>Heatmap:</strong> Color-coded matrix showing all vector values at once (not for ISO vectors)</li>
                  <li><strong>Comparison:</strong> Side-by-side analysis of two specific time periods</li>
                  <li><strong>Shape Similarity:</strong> Matrix showing candlestick shape similarity (ISO vectors only)</li>
                </ul>
                <p className="mt-2 text-xs"><strong>Tip:</strong> Use heatmap for numerical analysis, shape similarity for pattern recognition.</p>
              </div>
            } isDarkMode={isDarkMode} />
          </div>
          <div className="flex space-x-2">
            {['heatmap', 'comparison', 'shape-similarity'].map(mode => {
              // Only show shape-similarity for ISO vectors, hide heatmap for ISO vectors
              if (mode === 'shape-similarity' && !selectedVectorType.includes('iso_')) return null;
              if (mode === 'heatmap' && selectedVectorType.includes('iso_')) return null;
              
              return (
                <button
                  key={mode}
                  onClick={() => setVectorViewMode(mode)}
                  className={`px-3 py-1 rounded text-sm font-medium transition-all duration-200 ${
                    vectorViewMode === mode 
                      ? isDarkMode
                        ? 'bg-blue-600 text-white' 
                        : 'bg-blue-600 text-white'
                      : isDarkMode
                        ? 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                        : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                  }`}
                >
                  {mode === 'shape-similarity' ? 'Shape Similarity' : mode.charAt(0).toUpperCase() + mode.slice(1)}
                </button>
              );
            })}
          </div>
        </div>

        <div className="min-h-64">
          {vectorViewMode === 'heatmap' && renderVectorHeatmap()}
          {vectorViewMode === 'comparison' && renderVectorComparison()}
          {vectorViewMode === 'shape-similarity' && renderShapeSimilarity()}
        </div>
      </div>

      {/* Raw Vector Data (Collapsible) */}
      <details className={`rounded-lg border transition-colors duration-200 ${
        isDarkMode ? 'bg-gray-800 border-gray-700' : 'bg-white border-gray-200'
      }`}>
        <summary className={`p-6 cursor-pointer font-semibold transition-colors duration-200 ${
          isDarkMode 
            ? 'text-gray-300 hover:bg-gray-700' 
            : 'text-gray-700 hover:bg-gray-50'
        }`}>
          Raw Vector Data (Click to expand)
        </summary>
        <div className="px-6 pb-6">
          <div className={`rounded p-4 max-h-64 overflow-auto transition-colors duration-200 ${
            isDarkMode ? 'bg-gray-900' : 'bg-gray-50'
          }`}>
            <pre className={`text-xs font-mono ${
              isDarkMode ? 'text-gray-300' : 'text-gray-700'
            }`}>
              {vectorData?.data?.slice(0, 3).map((row, i) => 
                `Row ${i + 1}: ${JSON.stringify(row[selectedVectorType]?.slice(0, 5), null, 2)}...\n`
              ).join('')}
            </pre>
          </div>
        </div>
      </details>
    </div>
  );
};

const TradingDashboard = () => {
  // Use shared trading context for data settings
  const {
    selectedSymbol,
    setSelectedSymbol,
    selectedTimeframe,
    setSelectedTimeframe,
    rowLimit,
    setRowLimit,
    sortOrder,
    setSortOrder,
    sortColumn,
    setSortColumn,
    currentPage,
    setCurrentPage,
    searchTerm,
    setSearchTerm,
    selectedSearchColumn,
    setSelectedSearchColumn,
    showColumnFilters,
    setShowColumnFilters,
    showDebugInfo,
    setShowDebugInfo,
    resetFilters,
    selectedCandles,
    addSelectedCandle,
    removeSelectedCandle,
    clearSelectedCandles,
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

  // Debug: Log context values to check synchronization
  useEffect(() => {
    console.log('📊 Data Dashboard - Context state:', {
      selectedSymbol,
      selectedTimeframe,
      rowLimit,
      sortOrder,
      currentPage
    });
  }, [selectedSymbol, selectedTimeframe, rowLimit, sortOrder, currentPage]);

  // Dashboard mode is local to each tab (not shared)
  const [dashboardMode, setDashboardMode] = useState('data'); // 'data', 'vector', 'chart', 'backtest', 'llm', 'pipeline', 'model'

  // Local state that doesn't need to be shared across tabs
  const [stats, setStats] = useState(null);
  const [tables, setTables] = useState([]);
  const [tradingData, setTradingData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  // Local selection state is now managed by TradingContext as selectedCandles
  // Keep track of local indices for UI purposes
  const [localSelectedIndices, setLocalSelectedIndices] = useState(new Set());
  const [lastFetchInfo, setLastFetchInfo] = useState(null);
  
  // Theme management (separate from trading state)
  const [isDarkMode, setIsDarkMode] = useState(() => {
    const saved = localStorage.getItem('tradingDashboard-theme');
    return saved ? saved === 'dark' : false;
  });

  useEffect(() => {
    localStorage.setItem('tradingDashboard-theme', isDarkMode ? 'dark' : 'light');
    // Update document class for global theme
    if (isDarkMode) {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
  }, [isDarkMode]);

  const toggleTheme = () => {
    setIsDarkMode(!isDarkMode);
  };

  const API_BASE_URL = 'http://localhost:8000/api/trading';

  useEffect(() => {
    fetchStats();
    fetchTables();
  }, []);

  useEffect(() => {
    if (selectedSymbol && selectedTimeframe) {
      fetchTradingData();
    }
  }, [selectedSymbol, selectedTimeframe, rowLimit, sortOrder, sortColumn, currentPage, fetchMode, dateRangeType, startDate, endDate]);

  useEffect(() => {
    if (selectedSymbol && selectedTimeframe) {
      setCurrentPage(1); // Reset to first page when selection changes
    }
  }, [selectedSymbol, selectedTimeframe, rowLimit, sortOrder, sortColumn]);

  // Clear local selection when changing symbol, timeframe, or page
  useEffect(() => {
    setLocalSelectedIndices(new Set());
  }, [selectedSymbol, selectedTimeframe, currentPage]);

  // Synchronize local selection indices with global selected candles
  useEffect(() => {
    if (!tradingData?.data) return;
    
    const newLocalIndices = new Set();
    
    // Check each row in current data to see if it's in global selection
    tradingData.data.forEach((candle, index) => {
      const candleId = `${selectedSymbol}_${selectedTimeframe}_${candle.timestamp}`;
      const isGloballySelected = selectedCandles.some(c => c.id === candleId);
      
      if (isGloballySelected) {
        newLocalIndices.add(index);
      }
    });
    
    setLocalSelectedIndices(newLocalIndices);
  }, [selectedCandles, tradingData?.data, selectedSymbol, selectedTimeframe]);

  const fetchStats = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/stats`);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const text = await response.text();
      if (!text) {
        throw new Error('Empty response from server');
      }
      const data = JSON.parse(text);
      setStats(data);
    } catch (err) {
      console.error('Error fetching stats:', err);
      setError('Failed to fetch database stats');
    }
  };

  const fetchTables = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/tables`);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const text = await response.text();
      if (!text) {
        throw new Error('Empty response from server');
      }
      const data = JSON.parse(text);
      setTables(data);
    } catch (err) {
      console.error('Error fetching tables:', err);
      setError('Failed to fetch tables');
    }
  };

  const fetchTradingData = async () => {
    setLoading(true);
    console.log(`🚀 fetchTradingData called - Page: ${currentPage}, Symbol: ${selectedSymbol}, Timeframe: ${selectedTimeframe}, RowLimit: ${rowLimit}, SortOrder: ${sortOrder}, FetchMode: ${fetchMode}, DateRangeType: ${dateRangeType}`);
    
    // Helper that performs the original /data fetch with pagination parameters
    const fetchFromStandardEndpoint = async () => {
      let url = `${API_BASE_URL}/data/${selectedSymbol}/${selectedTimeframe}`;
      let queryParams = new URLSearchParams();
      let localOffset = 0;

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
        // Set a reasonable limit to avoid overwhelming the UI
        queryParams.append('limit', '10000');
        // For date range mode, we don't use pagination offset
        localOffset = 0;
      } else {
        // Use record limit mode (original logic)
        queryParams.append('limit', rowLimit.toString());
        
        // Strategy: For oldest data, we need to get the very first records from the database
        // For newest data, we get the most recent records
        localOffset = (currentPage - 1) * rowLimit;
      }

      // Add offset for pagination (only in record limit mode)
      if (fetchMode !== FETCH_MODES.DATE_RANGE) {
        queryParams.append('offset', localOffset.toString());
      }

      // Add sorting parameters
      queryParams.append('order', sortOrder);
      queryParams.append('sort_by', sortColumn);

      const finalUrl = `${url}?${queryParams.toString()}`;
      
      if (sortOrder === 'asc') {
        console.log(`🟢 ASCENDING: Requesting OLDEST records`);
      } else {
        console.log(`🔵 DESCENDING: Requesting NEWEST records`);
      }
      console.log('API URL:', finalUrl);
      
      const response = await fetch(finalUrl);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const text = await response.text();
      if (!text) {
        throw new Error('Empty response from server');
      }
      let standardData = JSON.parse(text);
      let standardFetchUrl = finalUrl;
     
      // If we're on ascending and still getting recent data, try a different approach
      if (sortOrder === 'asc' && standardData.data && standardData.data.length > 0) {
        const firstTimestamp = new Date(standardData.data[0].timestamp);
        const currentYear = new Date().getFullYear();
        const dataYear = firstTimestamp.getFullYear();
        
        // If the data is from recent times (same year), try to get older data
        if (dataYear >= currentYear - 1) {
          console.log('⚠️ Still getting recent data for ascending, trying alternative approach...');
          
          // Try to get data with a much larger offset to reach older records
          const largeOffset = Math.max(10000, (currentPage - 1) * rowLimit);
          const alternativeUrl = `${API_BASE_URL}/data/${selectedSymbol}/${selectedTimeframe}?limit=${rowLimit}&offset=${largeOffset}&order=asc&sort_by=timestamp`;
          
          try {
            console.log('Trying large offset approach:', alternativeUrl);
            const altResponse = await fetch(alternativeUrl);
            if (!altResponse.ok) {
              throw new Error(`HTTP error! status: ${altResponse.status}`);
            }
            const altText = await altResponse.text();
            if (!altText) {
              throw new Error('Empty response from server');
            }
            const altData = JSON.parse(altText);
            
            if (altData.data && altData.data.length > 0) {
              const altFirstTimestamp = new Date(altData.data[0].timestamp);
              // If we got older data, use it
              if (altFirstTimestamp < firstTimestamp) {
                console.log('✅ Found older data with large offset approach');
                standardData = altData;
                standardFetchUrl = alternativeUrl;
              }
            }
          } catch (e) {
            console.log('Large offset approach failed, using original data');
          }
        }
      }
      
      // Estimate total count
      if (!standardData.total_count) {
        // Try to get a better estimate of total records
        try {
          const countResponse = await fetch(`${API_BASE_URL}/data/${selectedSymbol}/${selectedTimeframe}?limit=1&count_only=true`);
          if (countResponse.ok) {
            const countText = await countResponse.text();
            if (countText) {
              const countData = JSON.parse(countText);
              standardData.total_count = countData.total_count || countData.count || standardData.count || 50000; // Better fallback
            }
          }
        } catch (e) {
          console.warn('Failed to get count estimate:', e);
          standardData.total_count = standardData.count || 50000; // Fallback estimate
        }
      }
      
      // Check if backend properly handled our sort order
      let backendHandledSorting = false;
      
      if (standardData.data && Array.isArray(standardData.data) && standardData.data.length > 1) {
        const firstTimestamp = new Date(standardData.data[0].timestamp);
        const lastTimestamp = new Date(standardData.data[standardData.data.length - 1].timestamp);
        
        if (sortOrder === 'asc') {
          // For ascending, first should be older than last
          backendHandledSorting = firstTimestamp <= lastTimestamp;
        } else {
          // For descending, first should be newer than last
          backendHandledSorting = firstTimestamp >= lastTimestamp;
        }
      }

      console.log(`Backend handled sorting properly (standard endpoint): ${backendHandledSorting}`);

      return {
        data: standardData,
        fetchUrl: standardFetchUrl,
        offset: localOffset,
        backendHandledSorting
      };
    };

    try {
      let data = null;
      let fetchUrl = null;
      let offset = 0;
      let backendHandledSorting = false;
      let usedHybrid = false;

      // --- 1) Try hybrid backtest+fronttest endpoint first ---
      try {
        const hybridParams = new URLSearchParams();

        if (fetchMode === FETCH_MODES.DATE_RANGE) {
          switch (dateRangeType) {
            case DATE_RANGE_TYPES.EARLIEST_TO_DATE:
              if (endDate) {
                hybridParams.append('end_date', endDate);
              }
              break;
            case DATE_RANGE_TYPES.DATE_TO_DATE:
              if (startDate) {
                hybridParams.append('start_date', startDate);
              }
              if (endDate) {
                hybridParams.append('end_date', endDate);
              }
              break;
            case DATE_RANGE_TYPES.DATE_TO_LATEST:
              if (startDate) {
                hybridParams.append('start_date', startDate);
              }
              break;
          }
        }

        // Request a generous slice so we can paginate client-side while still
        // spanning the backtest→fronttest boundary.
        const HYBRID_LIMIT = 100000;
        hybridParams.append('limit', HYBRID_LIMIT.toString());

        const hybridUrl = `${API_BASE_URL}/hybrid/${selectedSymbol}/${selectedTimeframe}?${hybridParams.toString()}`;
        console.log('🧪 Attempting hybrid trading data fetch:', hybridUrl);

        const hybridResponse = await fetch(hybridUrl);
        if (hybridResponse.ok) {
          const text = await hybridResponse.text();
          if (text) {
            data = JSON.parse(text);
            fetchUrl = hybridUrl;
            usedHybrid = true;
            offset = 0; // pagination is handled client-side for hybrid
            console.log('✅ Hybrid trading data fetched:', data.data?.length, 'rows');
          }
        } else {
          console.warn(`Hybrid endpoint returned ${hybridResponse.status}, falling back to /data`);
        }
      } catch (hybridErr) {
        console.warn('Hybrid fetch failed, falling back to /data:', hybridErr);
      }

      // --- 2) Fallback to original /data endpoint if hybrid not available ---
      if (!data) {
        const standardResult = await fetchFromStandardEndpoint();
        data = standardResult.data;
        fetchUrl = standardResult.fetchUrl;
        offset = standardResult.offset;
        backendHandledSorting = standardResult.backendHandledSorting;
      } else {
        // Hybrid endpoint always returns ascending by timestamp
        backendHandledSorting = true;
      }

      if (data.data && Array.isArray(data.data)) {
        let sortedData = [...data.data];
        
        // Always apply client-side sorting to ensure proper ordering
        console.log(`🔀 Applying client-side sorting by ${sortColumn} (${sortOrder})...`);
        console.log(`📊 Sample data before sort:`, sortedData.slice(0, 3).map(row => ({ [sortColumn]: row[sortColumn] })));
        
        sortedData.sort((a, b) => {
          let aVal = a[sortColumn];
          let bVal = b[sortColumn];
          
          // Handle different data types
          if (sortColumn === 'timestamp') {
            aVal = new Date(aVal);
            bVal = new Date(bVal);
          } else if (typeof aVal === 'string') {
            aVal = aVal.toLowerCase();
            bVal = bVal.toLowerCase();
          } else if (typeof aVal === 'number') {
            // Numbers - handle null/undefined
            aVal = aVal || 0;
            bVal = bVal || 0;
          }
          
          if (aVal < bVal) return sortOrder === 'asc' ? -1 : 1;
          if (aVal > bVal) return sortOrder === 'asc' ? 1 : -1;
          return 0;
        });

        // For hybrid we keep the full sorted array for range/debug, but page
        // the visible rows on the client side.
        let rangeSource = sortedData;

        if (usedHybrid) {
          const totalCount = sortedData.length;
          const startIndex = (currentPage - 1) * rowLimit;
          const endIndex = startIndex + rowLimit;
          const pageData = sortedData.slice(startIndex, endIndex);

          data.data = pageData;
          data.count = pageData.length;
          data.total_count = totalCount;
        } else {
          data.data = sortedData;
        }
        
        console.log(`✅ Sorting complete. Sample data after sort:`, sortedData.slice(0, 3).map(row => ({ [sortColumn]: row[sortColumn] })));
        
        // Enhanced debug information
        setLastFetchInfo({
          timestamp: new Date().toLocaleTimeString(),
          sortOrder: sortOrder,
          requestedData: sortOrder === 'asc' ? 'OLDEST' : 'NEWEST',
          rowLimit: rowLimit,
          currentPage: currentPage,
          offset: usedHybrid ? (currentPage - 1) * rowLimit : offset,
          apiUrl: fetchUrl,
          backendHandledSorting: backendHandledSorting,
          actualDataRange: rangeSource.length > 0 ? {
            first: rangeSource[0].timestamp,
            last: rangeSource[rangeSource.length - 1].timestamp,
            firstYear: new Date(rangeSource[0].timestamp).getFullYear(),
            lastYear: new Date(rangeSource[rangeSource.length - 1].timestamp).getFullYear(),
            span: `${new Date(rangeSource[0].timestamp).toLocaleDateString()} to ${new Date(rangeSource[rangeSource.length - 1].timestamp).toLocaleDateString()}`
          } : null,
          totalRecords: data.total_count || data.count,
          dataQuality: (() => {
            if (sortOrder === 'desc') return 'GOOD (Recent data)';
            
            // For ascending, check if we're actually getting properly sorted data
            if (rangeSource.length > 1) {
              const firstTime = new Date(rangeSource[0].timestamp);
              const lastTime = new Date(rangeSource[rangeSource.length - 1].timestamp);
              
              // If properly sorted ascending and we're on page 1, this should be good
              if (firstTime <= lastTime && currentPage === 1) {
                return 'GOOD (Historical data - oldest first)';
              } else if (firstTime <= lastTime) {
                return 'GOOD (Historical data - properly sorted)';
              } else {
                return 'POOR (Sorting may be incorrect)';
              }
            }
            
            // Single record or no data
            return currentPage === 1 ? 'GOOD (Historical data)' : 'OK (Paginated data)';
          })()
        });
      }
      
      console.log(`📊 Setting trading data - Total Count: ${data.total_count}, Data Length: ${data.data?.length}, Current Page: ${currentPage}, UsedHybrid: ${usedHybrid}`);
      setTradingData(data);
      setError(null);
    } catch (err) {
      setError(`Failed to fetch trading data for ${selectedSymbol}_${selectedTimeframe}: ${err.message}`);
      console.error('Fetch error:', err);
    } finally {
      setLoading(false);
    }
  };

  const formatPrice = (price) => {
    return price ? price.toFixed(4) : 'N/A';
  };

  const formatTimestamp = (timestamp) => {
    return new Date(timestamp).toLocaleString();
  };

  const getSymbolColor = (symbol) => {
    switch (symbol) {
      case 'es': return 'text-blue-600';
      case 'eurusd': return 'text-green-600';
      case 'spy': return 'text-purple-600';
      default: return 'text-gray-600';
    }
  };

  const getFilteredData = () => {
    if (!tradingData?.data) return [];
    if (!searchTerm) return tradingData.data;
    
    // Remove commas from search term for better functionality
    const cleanSearchTerm = searchTerm.replace(/,/g, '');
    const searchLower = cleanSearchTerm.toLowerCase();
    
    return tradingData.data.filter(row => {
      // If searching all columns (default behavior)
      if (selectedSearchColumn === 'all') {
        return (
          row.timestamp?.toString().toLowerCase().includes(searchLower) ||
          row.open?.toString().includes(cleanSearchTerm) ||
          row.high?.toString().includes(cleanSearchTerm) ||
          row.low?.toString().includes(cleanSearchTerm) ||
          row.close?.toString().includes(cleanSearchTerm) ||
          row.volume?.toString().includes(cleanSearchTerm) ||
          formatTimestamp(row.timestamp).toLowerCase().includes(searchLower) ||
          formatPrice(row.open).includes(cleanSearchTerm) ||
          formatPrice(row.high).includes(cleanSearchTerm) ||
          formatPrice(row.low).includes(cleanSearchTerm) ||
          formatPrice(row.close).includes(cleanSearchTerm)
        );
      }
      
      // Column-specific search
      switch (selectedSearchColumn) {
        case 'timestamp':
          return (
            row.timestamp?.toString().toLowerCase().includes(searchLower) ||
            formatTimestamp(row.timestamp).toLowerCase().includes(searchLower)
          );
        case 'open':
          return (
            row.open?.toString().includes(cleanSearchTerm) ||
            formatPrice(row.open).includes(cleanSearchTerm)
          );
        case 'high':
          return (
            row.high?.toString().includes(cleanSearchTerm) ||
            formatPrice(row.high).includes(cleanSearchTerm)
          );
        case 'low':
          return (
            row.low?.toString().includes(cleanSearchTerm) ||
            formatPrice(row.low).includes(cleanSearchTerm)
          );
        case 'close':
          return (
            row.close?.toString().includes(cleanSearchTerm) ||
            formatPrice(row.close).includes(cleanSearchTerm)
          );
        case 'volume':
          return row.volume?.toString().includes(cleanSearchTerm);
        default:
          return true;
      }
    });
  };

  const getFilteredIndices = () => {
    const filteredData = getFilteredData();
    const indices = [];
    tradingData?.data?.forEach((row, index) => {
      if (filteredData.includes(row)) {
        indices.push(index);
      }
    });
    return indices;
  };

  const handleColumnSort = (column) => {
    console.log(`🔄 Column sort clicked: ${column}, current sort: ${sortColumn} ${sortOrder}`);
    
    if (sortColumn === column) {
      // Toggle sort order if clicking the same column
      setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc');
    } else {
      // Set new column and default to descending
      setSortColumn(column);
      setSortOrder('desc');
    }
    setCurrentPage(1);
  };

  const handleRowSelect = (index, candle) => {
    // Create a unique ID for the candle based on symbol, timeframe, and timestamp
    const candleId = `${selectedSymbol}_${selectedTimeframe}_${candle.timestamp}`;
    
    // Check if this candle is already selected
    const isSelected = selectedCandles.some(c => c.id === candleId);
    
    if (isSelected) {
      // Remove from selection
      removeSelectedCandle(candleId);
      setLocalSelectedIndices(prev => {
        const newSet = new Set(prev);
        newSet.delete(index);
        return newSet;
      });
    } else {
      // Add to selection with full candle data
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
        // Add calculated fields
        change: candle.close - candle.open,
        changePercent: ((candle.close - candle.open) / candle.open) * 100,
        // Add index for reference
        sourceIndex: index
      };
      
      addSelectedCandle(candleWithMetadata);
      setLocalSelectedIndices(prev => new Set([...prev, index]));
    }
  };

  const handleSelectAll = () => {
    const filteredData = getFilteredData();
    
    // Check if all filtered candles are currently selected
    const allSelected = filteredData.every((candle) => {
      const candleId = `${selectedSymbol}_${selectedTimeframe}_${candle.timestamp}`;
      return selectedCandles.some(c => c.id === candleId);
    });

    if (allSelected) {
      // Deselect all filtered candles
      const currentPageCandleIds = filteredData.map(candle => 
        `${selectedSymbol}_${selectedTimeframe}_${candle.timestamp}`
      );
      
      // Remove current page candles from global selection
      currentPageCandleIds.forEach(id => removeSelectedCandle(id));
    } else {
      // Select all filtered candles that aren't already selected
      filteredData.forEach((candle, filteredIndex) => {
        const candleId = `${selectedSymbol}_${selectedTimeframe}_${candle.timestamp}`;
        
        if (!selectedCandles.some(c => c.id === candleId)) {
          // Find the actual index in the full data array
          const actualIndex = tradingData.data.indexOf(candle);
          
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
            sourceIndex: actualIndex
          };
          
          addSelectedCandle(candleWithMetadata);
        }
      });
    }
    
    // Note: localSelectedIndices will be updated automatically by the synchronization useEffect
  };

  const exportToCSV = () => {
    if (!tradingData?.data) return;
    
    const headers = ['Row', 'Timestamp', 'Open', 'High', 'Low', 'Close', 'Candle Type', 'Volume'];
    const csvContent = [
      headers.join(','),
      ...tradingData.data.map((row, index) => {
        const isGreen = row.close > row.open;
        const isEqual = row.close === row.open;
        const candleType = isEqual ? 'DOJI' : (isGreen ? 'BULL' : 'BEAR');
        
        return [
          ((currentPage - 1) * rowLimit) + index + 1,
          row.timestamp,
          row.open,
          row.high,
          row.low,
          row.close,
          candleType,
          row.volume
        ].join(',');
      })
    ].join('\n');

    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${selectedSymbol}_${selectedTimeframe}_data.csv`;
    a.click();
    window.URL.revokeObjectURL(url);
  };

  const quickSelectTable = (table) => {
    const [symbol, timeframe] = table.table_name.split('_');
    setSelectedSymbol(symbol);
    setSelectedTimeframe(timeframe);
  };

  const handleBulkDelete = () => {
    const currentSymbolTimeframeSelection = selectedCandles.filter(c => 
      c.symbol === selectedSymbol && c.timeframe === selectedTimeframe
    );
    
    if (currentSymbolTimeframeSelection.length === 0) return;
    
    if (confirm(`Are you sure you want to delete ${currentSymbolTimeframeSelection.length} selected candles?`)) {
      // TODO: Implement bulk delete API call
      alert('Bulk delete functionality would be implemented here');
      
      // Clear the selected candles from global state
      currentSymbolTimeframeSelection.forEach(candle => {
        removeSelectedCandle(candle.id);
      });
    }
  };

  const handleRefresh = () => {
    // Clear only local selection state, global selection will be re-synchronized
    setLocalSelectedIndices(new Set());
    fetchTradingData();
  };

  const totalPages = Math.max(1, Math.ceil((tradingData?.total_count || tradingData?.count || tradingData?.data?.length || 0) / rowLimit));
  
  // Debug pagination calculations
  if (tradingData) {
    console.log(`📄 Pagination Debug - Total Count: ${tradingData.total_count}, Count: ${tradingData.count}, Data Length: ${tradingData.data?.length}, Row Limit: ${rowLimit}, Total Pages: ${totalPages}, Current Page: ${currentPage}`);
  }

  // Helper function to check if a candle is selected globally
  const isCandleSelected = (candle, index) => {
    const candleId = `${selectedSymbol}_${selectedTimeframe}_${candle.timestamp}`;
    return localSelectedIndices.has(index) || selectedCandles.some(c => c.id === candleId);
  };

  return (
    <div className={`min-h-screen transition-colors duration-200 p-6 ${
      isDarkMode ? 'bg-gray-900' : 'bg-gray-50'
    }`}>
      <div className="max-w-7xl mx-auto">
        {/* Header with Dashboard Toggle */}
        <div className="mb-8">
          <div className="flex items-center justify-between">
            <div>
              <div className="flex items-center mb-2">
                <h1 className={`text-5xl font-black tracking-tight transition-colors duration-200 ${
                  isDarkMode ? 'text-white' : 'text-gray-900'
                }`} style={{ 
                  fontFamily: 'system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
                  letterSpacing: '-0.02em',
                  textShadow: isDarkMode ? '0 2px 4px rgba(0,0,0,0.5)' : '0 2px 4px rgba(0,0,0,0.1)'
                }}>
                  Day<span className={`${isDarkMode ? 'text-blue-400' : 'text-blue-600'}`}>gent</span>
                </h1>
                <div className={`ml-4 px-3 py-1 rounded-lg text-sm font-semibold transition-colors duration-200 ${
                  dashboardMode === 'data' ? 
                    (isDarkMode ? 'bg-blue-900/30 text-blue-300' : 'bg-blue-100 text-blue-700') :
                  dashboardMode === 'vector' ? 
                    (isDarkMode ? 'bg-purple-900/30 text-purple-300' : 'bg-purple-100 text-purple-700') :
                  dashboardMode === 'chart' ? 
                    (isDarkMode ? 'bg-green-900/30 text-green-300' : 'bg-green-100 text-green-700') :
                  dashboardMode === 'backtest' ? 
                    (isDarkMode ? 'bg-indigo-900/30 text-indigo-300' : 'bg-indigo-100 text-indigo-700') :
                    (isDarkMode ? 'bg-orange-900/30 text-orange-300' : 'bg-orange-100 text-orange-700')
                }`}>
                  {dashboardMode === 'data' ? 'Data Analytics' : 
                   dashboardMode === 'vector' ? 'Vector Intelligence' : 
                   dashboardMode === 'chart' ? 'Chart Analysis' :
                   dashboardMode === 'backtest' ? 'Backtest Analysis' :
                   'AI Assistant'}
                </div>
              </div>
                              <p className={`text-lg transition-colors duration-200 ${
                  isDarkMode ? 'text-gray-300' : 'text-gray-600'
                }`}>
                  {dashboardMode === 'data' 
                    ? 'Professional trading database management • Real-time market data access'
                    : dashboardMode === 'vector' 
                    ? 'Advanced pattern recognition & ML analysis • AI-powered market insights'
                    : dashboardMode === 'chart'
                    ? 'Interactive charting & technical analysis • Professional trading tools'
                    : dashboardMode === 'backtest'
                    ? 'Historical replay & strategy evaluation • Candle-by-candle simulation'
                    : 'Intelligent trading assistant • Natural language market analysis'}
                </p>
            </div>
            
            {/* Professional Dashboard Toggle and Theme Toggle */}
            <div className="flex items-center space-x-4">
              {/* Theme Toggle */}
              <button
                onClick={toggleTheme}
                className={`p-2 rounded-lg transition-all duration-200 ${
                  isDarkMode 
                    ? 'bg-gray-800 text-yellow-400 hover:bg-gray-700' 
                    : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                }`}
                title={isDarkMode ? 'Switch to Light Mode' : 'Switch to Dark Mode'}
              >
                {isDarkMode ? '🌞' : '🌙'}
              </button>
              
              {/* Dashboard Mode Toggle */}
              <div className={`flex rounded-lg p-1 transition-colors duration-200 ${
                isDarkMode ? 'bg-gray-800' : 'bg-gray-100'
              }`}>
                <button
                  onClick={() => setDashboardMode('data')}
                  className={`px-4 py-2 rounded-md text-sm font-medium transition-all duration-200 flex items-center space-x-2 ${
                    dashboardMode === 'data'
                      ? isDarkMode 
                        ? 'bg-gray-700 text-blue-400 shadow-sm ring-1 ring-blue-500/50'
                        : 'bg-white text-blue-600 shadow-sm ring-1 ring-blue-200'
                      : isDarkMode
                        ? 'text-gray-300 hover:text-white hover:bg-gray-700'
                        : 'text-gray-600 hover:text-gray-800 hover:bg-gray-50'
                  }`}
                >
                  <span>📊</span>
                  <span>Data Dashboard</span>
                </button>
                <button
                  onClick={() => setDashboardMode('vector')}
                  className={`px-4 py-2 rounded-md text-sm font-medium transition-all duration-200 flex items-center space-x-2 ${
                    dashboardMode === 'vector'
                      ? isDarkMode
                        ? 'bg-gray-700 text-purple-400 shadow-sm ring-1 ring-purple-500/50'
                        : 'bg-white text-purple-600 shadow-sm ring-1 ring-purple-200'
                      : isDarkMode
                        ? 'text-gray-300 hover:text-white hover:bg-gray-700'
                        : 'text-gray-600 hover:text-gray-800 hover:bg-gray-50'
                  }`}
                >
                  <span>🧠</span>
                  <span>Vector Dashboard</span>
                </button>
                <button
                  onClick={() => setDashboardMode('chart')}
                  className={`px-4 py-2 rounded-md text-sm font-medium transition-all duration-200 flex items-center space-x-2 ${
                    dashboardMode === 'chart'
                      ? isDarkMode
                        ? 'bg-gray-700 text-green-400 shadow-sm ring-1 ring-green-500/50'
                        : 'bg-white text-green-600 shadow-sm ring-1 ring-green-200'
                      : isDarkMode
                        ? 'text-gray-300 hover:text-white hover:bg-gray-700'
                        : 'text-gray-600 hover:text-gray-800 hover:bg-gray-50'
                  }`}
                >
                  <span>📈</span>
                  <span>Chart Dashboard</span>
                </button>
                <button
                  onClick={() => setDashboardMode('backtest')}
                  className={`px-4 py-2 rounded-md text-sm font-medium transition-all duration-200 flex items-center space-x-2 ${
                    dashboardMode === 'backtest'
                      ? (isDarkMode
                        ? 'bg-gray-700 text-indigo-400 shadow-sm ring-1 ring-indigo-500/50'
                        : 'bg-white text-indigo-600 shadow-sm ring-1 ring-indigo-200')
                      : (isDarkMode
                        ? 'text-gray-300 hover:text-white hover:bg-gray-700'
                        : 'text-gray-600 hover:text-gray-800 hover:bg-gray-50')
                  }`}
                >
                  <span>🧪</span>
                  <span>Backtest Dashboard</span>
                </button>
                <button
                  onClick={() => setDashboardMode('llm')}
                  className={`px-4 py-2 rounded-md text-sm font-medium transition-all duration-200 flex items-center space-x-2 ${
                    dashboardMode === 'llm'
                      ? isDarkMode
                        ? 'bg-gray-700 text-orange-400 shadow-sm ring-1 ring-orange-500/50'
                        : 'bg-white text-orange-600 shadow-sm ring-1 ring-orange-200'
                      : isDarkMode
                        ? 'text-gray-300 hover:text-white hover:bg-gray-700'
                        : 'text-gray-600 hover:text-gray-800 hover:bg-gray-50'
                  }`}
                >
                  <span>🤖</span>
                  <span>LLM Dashboard</span>
                </button>
                <button
                  onClick={() => setDashboardMode('pipeline')}
                  className={`px-4 py-2 rounded-md text-sm font-medium transition-all duration-200 flex items-center space-x-2 ${
                    dashboardMode === 'pipeline'
                      ? isDarkMode
                        ? 'bg-gray-700 text-blue-400 shadow-sm ring-1 ring-blue-500/50'
                        : 'bg-white text-blue-600 shadow-sm ring-1 ring-blue-200'
                      : isDarkMode
                        ? 'text-gray-300 hover:text-white hover:bg-gray-700'
                        : 'text-gray-600 hover:text-gray-800 hover:bg-gray-50'
                  }`}
                >
                  <span>⚙️</span>
                  <span>Pipeline Dashboard</span>
                </button>
                <button
                  onClick={() => setDashboardMode('model')}
                  className={`px-4 py-2 rounded-md text-sm font-medium transition-all duration-200 flex items-center space-x-2 ${
                    dashboardMode === 'model'
                      ? isDarkMode
                        ? 'bg-gray-700 text-blue-400 shadow-sm ring-1 ring-blue-500/50'
                        : 'bg-white text-blue-600 shadow-sm ring-1 ring-blue-200'
                      : isDarkMode
                        ? 'text-gray-300 hover:text-white hover:bg-gray-700'
                        : 'text-gray-600 hover:text-gray-800 hover:bg-gray-50'
                  }`}
                >
                  <span>🧪</span>
                  <span>Model Training</span>
                </button>
              </div>
            </div>
          </div>
        </div>

        {/* Error Display */}
        {error && (
          <div className={`mb-6 rounded-lg p-4 border transition-colors duration-200 ${
          isDarkMode 
            ? 'bg-red-900/20 border-red-800 text-red-300' 
            : 'bg-red-50 border-red-200 text-red-800'
        }`}>
            <div className="flex items-center">
              <div className="text-red-600 mr-2">⚠️</div>
              <div>{error}</div>
          </div>
          </div>
        )}

        {/* Conditional Dashboard Rendering */}
        {dashboardMode === 'vector' ? (
          <TradingVectorDashboard
            tables={tables}
            isDarkMode={isDarkMode}
          />
        ) : dashboardMode === 'chart' ? (
          <TradingChartDashboard
            tables={tables}
            isDarkMode={isDarkMode}
          />
        ) : dashboardMode === 'backtest' ? (
          <BacktestDashboard
            isDarkMode={isDarkMode}
          />
        ) : dashboardMode === 'llm' ? (
          <TradingLLMDashboard
            tables={tables}
            isDarkMode={isDarkMode}
          />
        ) : dashboardMode === 'pipeline' ? (
          <PipelineDashboard isDarkMode={isDarkMode} />
        ) : dashboardMode === 'model' ? (
          <ModelTraining isDarkMode={isDarkMode} />
        ) : (
          <>
            {/* Original Data Dashboard Content */}

        {/* Database Stats */}
        {stats && (
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
            <div className={`rounded-lg shadow-md p-6 transition-colors duration-200 ${
              isDarkMode ? 'bg-gray-800' : 'bg-white'
            }`}>
              <div className="flex items-center">
                <div className="text-3xl mr-4">📊</div>
                <div>
                  <h3 className={`text-lg font-semibold ${
                    isDarkMode ? 'text-white' : 'text-gray-900'
                  }`}>Total Tables</h3>
                  <p className="text-2xl font-bold text-blue-500">{stats.total_tables}</p>
                </div>
              </div>
            </div>
            
            <div className={`rounded-lg shadow-md p-6 transition-colors duration-200 ${
              isDarkMode ? 'bg-gray-800' : 'bg-white'
            }`}>
              <div className="flex items-center">
                <div className="text-3xl mr-4">💾</div>
                <div>
                  <h3 className={`text-lg font-semibold ${
                    isDarkMode ? 'text-white' : 'text-gray-900'
                  }`}>Total Records</h3>
                  <p className="text-2xl font-bold text-green-500">
                    {stats.total_rows.toLocaleString()}
                  </p>
                </div>
              </div>
            </div>
            
            <div className={`rounded-lg shadow-md p-6 transition-colors duration-200 ${
              isDarkMode ? 'bg-gray-800' : 'bg-white'
            }`}>
              <div className="flex items-center">
                <div className="text-3xl mr-4">🔄</div>
                <div>
                  <h3 className={`text-lg font-semibold ${
                    isDarkMode ? 'text-white' : 'text-gray-900'
                  }`}>Status</h3>
                  <p className="text-2xl font-bold text-green-500">Connected</p>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Data Selection & Controls - Using Consistent Component */}
        <DataSelectionControls 
          selectedSymbol={selectedSymbol}
          setSelectedSymbol={setSelectedSymbol}
          selectedTimeframe={selectedTimeframe}
          setSelectedTimeframe={setSelectedTimeframe}
          rowLimit={rowLimit}
          setRowLimit={setRowLimit}
          sortOrder={sortOrder}
          setSortOrder={setSortOrder}
          searchTerm={searchTerm}
          setSearchTerm={setSearchTerm}
          selectedSearchColumn={selectedSearchColumn}
          setSelectedSearchColumn={setSelectedSearchColumn}
          showColumnFilters={showColumnFilters}
          setShowColumnFilters={setShowColumnFilters}
          showDebugInfo={showDebugInfo}
          setShowDebugInfo={setShowDebugInfo}
          handleRefresh={handleRefresh}
          lastFetchInfo={lastFetchInfo}
          isDarkMode={isDarkMode}
          dashboardType="data"
          // Date range props
          fetchMode={fetchMode}
          setFetchMode={setFetchMode}
          dateRangeType={dateRangeType}
          setDateRangeType={setDateRangeType}
          startDate={startDate}
          setStartDate={setStartDate}
          endDate={endDate}
          setEndDate={setEndDate}
        />

        {/* Selected Candles Panel */}
        <SelectedCandlesPanel 
          isDarkMode={isDarkMode} 
          canSelectCandles={true}
        />

        {/* Trading Data Table */}
        <div className={`rounded-lg shadow-md overflow-hidden transition-colors duration-200 ${
          isDarkMode ? 'bg-gray-800' : 'bg-white'
        }`}>
          <div className={`px-6 py-4 border-b flex items-center justify-between transition-colors duration-200 ${
            isDarkMode ? 'border-gray-700' : 'border-gray-200'
          }`}>
            <div>
              <h3 className={`text-lg font-semibold ${
                isDarkMode ? 'text-white' : 'text-gray-900'
              }`}>
                {selectedSymbol.toUpperCase()} - {selectedTimeframe} Data
                {tradingData && (
                  <span className={`ml-2 text-sm ${
                    isDarkMode ? 'text-gray-400' : 'text-gray-500'
                  }`}>
                    ({(() => {
                      if (!searchTerm) return tradingData.count || tradingData.data?.length || 0;
                      const filteredCount = getFilteredData().length;
                      return `${filteredCount} filtered / ${tradingData.count || tradingData.data?.length || 0} total`;
                    })()} records)
                  </span>
                )}
                <br />
                <div className="flex items-center">
                <span className={`text-sm font-medium ${sortOrder === 'asc' ? 'text-green-600' : 'text-blue-600'}`}>
                  {sortOrder === 'asc' ? '🟢 Showing OLDEST data first' : '🔵 Showing NEWEST data first'} 
                  {lastFetchInfo && (
                    <span className={`ml-2 ${
                      isDarkMode ? 'text-gray-400' : 'text-gray-500'
                    }`}>
                      (Backend sorting: {lastFetchInfo.backendHandledSorting ? '✅' : '❌ using client-side fallback'})
                    </span>
                  )}
                </span>
                  <InfoTooltip id="data-order" content={
                    <div>
                      <p className="font-semibold mb-2">📊 Data Order & Quality</p>
                      <p className="mb-2">Understanding your data display:</p>
                      <ul className="list-disc list-inside space-y-1 text-xs">
                        <li><strong>🟢 OLDEST first:</strong> Historical analysis, pattern backtesting</li>
                        <li><strong>🔵 NEWEST first:</strong> Current market monitoring, recent activity</li>
                        <li><strong>✅ Backend sorting:</strong> Server handles ordering efficiently</li>
                        <li><strong>❌ Client fallback:</strong> Browser sorts data (may be slower)</li>
                      </ul>
                      <p className="mt-2 text-xs">Data quality indicators help you understand if you're getting the expected time range.</p>
                    </div>
                  } isDarkMode={isDarkMode} />
                </div>
              </h3>
              {(() => {
                const currentSymbolTimeframeSelection = selectedCandles.filter(c => 
                  c.symbol === selectedSymbol && c.timeframe === selectedTimeframe
                );
                return currentSymbolTimeframeSelection.length > 0 && (
                  <p className="text-sm text-blue-600 mt-1">
                    {currentSymbolTimeframeSelection.length} candle(s) selected for analysis
                  </p>
                );
              })()}
            </div>
            <div className="flex gap-2">
              <button
                onClick={exportToCSV}
                disabled={!tradingData?.data?.length}
                className={`px-3 py-2 text-sm rounded-md transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-1 ${
                  isDarkMode 
                    ? 'bg-green-800 hover:bg-green-700 text-green-300' 
                    : 'bg-green-100 hover:bg-green-200 text-green-700'
                }`}
              >
                📊 Export CSV
              </button>
              {(() => {
                const currentSymbolTimeframeSelection = selectedCandles.filter(c => 
                  c.symbol === selectedSymbol && c.timeframe === selectedTimeframe
                );
                return currentSymbolTimeframeSelection.length > 0 && (
                  <>
                    <button
                      onClick={handleBulkDelete}
                      className={`px-3 py-2 text-sm rounded-md transition-colors flex items-center gap-1 ${
                        isDarkMode 
                          ? 'bg-red-800 hover:bg-red-700 text-red-300' 
                          : 'bg-red-100 hover:bg-red-200 text-red-700'
                      }`}
                    >
                      🗑️ Delete Selected ({currentSymbolTimeframeSelection.length})
                    </button>
                    <button
                      onClick={() => {
                        // Clear selections for current symbol/timeframe from global state
                        currentSymbolTimeframeSelection.forEach(candle => {
                          removeSelectedCandle(candle.id);
                        });
                      }}
                      className={`px-3 py-2 text-sm rounded-md transition-colors ${
                        isDarkMode 
                          ? 'bg-gray-700 hover:bg-gray-600 text-gray-300' 
                          : 'bg-gray-100 hover:bg-gray-200 text-gray-700'
                      }`}
                    >
                      ✖️ Clear Selection
                    </button>
                  </>
                );
              })()}
            </div>
          </div>
          
          {loading ? (
            <div className="p-8 text-center">
              <div className={`inline-block animate-spin rounded-full h-8 w-8 border-b-2 ${
                isDarkMode ? 'border-blue-400' : 'border-blue-600'
              }`}></div>
              <p className={`mt-2 ${
                isDarkMode ? 'text-gray-300' : 'text-gray-600'
              }`}>Loading trading data...</p>
            </div>
          ) : tradingData && tradingData.data && tradingData.data.length > 0 ? (
            <>
              {/* Enhanced Pagination - TOP */}
              {totalPages > 1 && (
                <div className={`px-6 py-4 border-b flex items-center justify-between transition-colors duration-200 ${
                  isDarkMode ? 'bg-gray-800 border-gray-700' : 'bg-white border-gray-200'
                }`}>
                  <div className={`flex items-center text-sm ${
                    isDarkMode ? 'text-gray-300' : 'text-gray-700'
                  }`}>
                    <span>
                      Showing {((currentPage - 1) * rowLimit) + 1} to {Math.min(currentPage * rowLimit, tradingData.total_count || tradingData.count || tradingData.data.length)} 
                      {' '}of {(tradingData.total_count || tradingData.count || tradingData.data.length)?.toLocaleString()} results
                    </span>
                  </div>
                  <div className="flex items-center space-x-2">
                    <button
                      onClick={() => {
                        console.log(`🔸 First button clicked - setting page to 1 (was ${currentPage}), totalPages: ${totalPages}`);
                        setCurrentPage(1);
                      }}
                      disabled={currentPage === 1}
                      className={`px-3 py-2 text-sm border rounded-md disabled:opacity-50 disabled:cursor-not-allowed transition-colors duration-200 ${
                        isDarkMode 
                          ? 'border-gray-600 text-gray-300 hover:bg-gray-700' 
                          : 'border-gray-300 text-gray-700 hover:bg-gray-50'
                      }`}
                    >
                      ⏮ First
                    </button>
                    <button
                      onClick={() => {
                        const newPage = Math.max(1, currentPage - 1);
                        console.log(`🔸 Previous button clicked - setting page to ${newPage} (was ${currentPage}), totalPages: ${totalPages}`);
                        setCurrentPage(newPage);
                      }}
                      disabled={currentPage === 1}
                      className={`px-3 py-2 text-sm border rounded-md disabled:opacity-50 disabled:cursor-not-allowed transition-colors duration-200 ${
                        isDarkMode 
                          ? 'border-gray-600 text-gray-300 hover:bg-gray-700' 
                          : 'border-gray-300 text-gray-700 hover:bg-gray-50'
                      }`}
                    >
                      ⬅ Previous
                    </button>
                    <span className={`px-3 py-2 text-sm rounded-md transition-colors duration-200 ${
                      isDarkMode ? 'bg-gray-700 text-gray-300' : 'bg-gray-100 text-gray-700'
                    }`}>
                      Page {currentPage} of {totalPages} ({(tradingData.total_count || tradingData.count || 0).toLocaleString()} total records)
                    </span>
                    <button
                      onClick={() => {
                        const newPage = Math.min(totalPages, currentPage + 1);
                        console.log(`🔸 Next button clicked - setting page to ${newPage} (was ${currentPage}), totalPages: ${totalPages}`);
                        setCurrentPage(newPage);
                      }}
                      disabled={currentPage >= totalPages}
                      className={`px-3 py-2 text-sm border rounded-md disabled:opacity-50 disabled:cursor-not-allowed transition-colors duration-200 ${
                        isDarkMode 
                          ? 'border-gray-600 text-gray-300 hover:bg-gray-700' 
                          : 'border-gray-300 text-gray-700 hover:bg-gray-50'
                      }`}
                    >
                      Next ➡
                    </button>
                    <button
                      onClick={() => {
                        console.log(`🔸 Last button clicked - setting page to ${totalPages} (was ${currentPage}), totalPages: ${totalPages}`);
                        setCurrentPage(totalPages);
                      }}
                      disabled={currentPage >= totalPages}
                      className={`px-3 py-2 text-sm border rounded-md disabled:opacity-50 disabled:cursor-not-allowed transition-colors duration-200 ${
                        isDarkMode 
                          ? 'border-gray-600 text-gray-300 hover:bg-gray-700' 
                          : 'border-gray-300 text-gray-700 hover:bg-gray-50'
                      }`}
                    >
                      Last ⏭
                    </button>
                  </div>
                </div>
              )}

              <div className="overflow-x-auto">
                <table className="min-w-full divide-y divide-gray-200">
                  <thead className={`transition-colors duration-200 ${
                    isDarkMode ? 'bg-gray-700' : 'bg-gray-50'
                  }`}>
                    <tr>
                      <th className="px-4 py-3 text-left">
                        <input
                          type="checkbox"
                          checked={(() => {
                            if (!tradingData?.data.length) return false;
                            const filteredData = getFilteredData();
                            if (filteredData.length === 0) return false;
                            
                            // Check if all filtered candles are selected (either locally or globally)
                            return filteredData.every((candle, index) => {
                              const actualIndex = tradingData.data.indexOf(candle);
                              return isCandleSelected(candle, actualIndex);
                            });
                          })()}
                          onChange={handleSelectAll}
                          className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                        />
                      </th>
                      <th className={`px-4 py-3 text-left text-xs font-medium uppercase tracking-wider ${
                        isDarkMode ? 'text-gray-400' : 'text-gray-500'
                      }`}>
                        #
                      </th>
                      {[
                        { key: 'timestamp', label: 'Timestamp', icon: '📅', sortable: true },
                        { key: 'open', label: 'Open', icon: '📈', sortable: true },
                        { key: 'high', label: 'High', icon: '🔺', sortable: true },
                        { key: 'low', label: 'Low', icon: '🔻', sortable: true },
                        { key: 'close', label: 'Close', icon: '💰', sortable: true },
                        { key: 'candle_type', label: 'Candle', icon: '🕯️', sortable: false },
                        { key: 'volume', label: 'Volume', icon: '📊', sortable: true }
                      ].map(({ key, label, icon, sortable }) => (
                        <th
                          key={key}
                          onClick={sortable ? () => handleColumnSort(key) : undefined}
                          className={`px-4 py-3 text-left text-xs font-medium uppercase tracking-wider select-none transition-colors duration-200 ${
                            isDarkMode ? 'text-gray-400' : 'text-gray-500'
                          } ${sortable 
                            ? isDarkMode 
                              ? 'cursor-pointer hover:bg-gray-600' 
                              : 'cursor-pointer hover:bg-gray-100'
                            : 'cursor-default'}`}
                        >
                          <div className="flex items-center">
                            <span className="mr-1">{icon}</span>
                            {label}
                            {sortable && sortColumn === key && (
                              <span className="ml-1 text-blue-600">
                                {sortOrder === 'asc' ? '↑' : '↓'}
                              </span>
                            )}
                            {sortable && <span className="ml-1 text-gray-400">⇅</span>}
                          </div>
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody className={`divide-y transition-colors duration-200 ${
                    isDarkMode ? 'bg-gray-800 divide-gray-700' : 'bg-white divide-gray-200'
                  }`}>
                    {getFilteredData().map((row, index) => {
                      const rowNumber = ((currentPage - 1) * rowLimit) + index + 1;
                      return (
                        <tr 
                          key={index} 
                          className={`transition-colors duration-200 cursor-pointer ${
                            isCandleSelected(row, index) 
                              ? isDarkMode 
                                ? 'bg-blue-900/30 border-l-4 border-blue-400' 
                                : 'bg-blue-50 border-l-4 border-blue-500'
                              : isDarkMode 
                                ? 'hover:bg-gray-700' 
                                : 'hover:bg-gray-50'
                          }`}
                          onClick={() => handleRowSelect(index, row)}
                        >
                                                      <td className="px-4 py-3 whitespace-nowrap" onClick={(e) => e.stopPropagation()}>
                            <input
                              type="checkbox"
                              checked={isCandleSelected(row, index)}
                              onChange={() => handleRowSelect(index, row)}
                              className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                            />
                          </td>
                          <td className={`px-4 py-3 whitespace-nowrap text-sm font-mono ${
                            isDarkMode ? 'text-gray-400' : 'text-gray-500'
                          }`}>
                            {rowNumber}
                          </td>
                          <td className={`px-4 py-3 whitespace-nowrap text-sm font-mono ${
                            isDarkMode ? 'text-gray-300' : 'text-gray-900'
                          }`}>
                            {formatTimestamp(row.timestamp)}
                          </td>
                          <td className={`px-4 py-3 whitespace-nowrap text-sm font-mono ${
                            isDarkMode ? 'text-gray-300' : 'text-gray-900'
                          }`}>
                            {formatPrice(row.open)}
                          </td>
                          <td className={`px-4 py-3 whitespace-nowrap text-sm font-mono font-semibold ${
                            isDarkMode ? 'text-gray-300' : 'text-gray-900'
                          }`}>
                            {formatPrice(row.high)}
                          </td>
                          <td className={`px-4 py-3 whitespace-nowrap text-sm font-mono font-semibold ${
                            isDarkMode ? 'text-gray-300' : 'text-gray-900'
                          }`}>
                            {formatPrice(row.low)}
                          </td>
                          <td className={`px-4 py-3 whitespace-nowrap text-sm font-mono font-bold ${
                            isDarkMode ? 'text-gray-200' : 'text-gray-900'
                          }`}>
                            {formatPrice(row.close)}
                          </td>
                          <td className="px-4 py-3 whitespace-nowrap text-sm text-center">
                            {(() => {
                              const isGreen = row.close > row.open;
                              const isEqual = row.close === row.open;
                              if (isEqual) {
                                return (
                                  <div className="flex items-center justify-center">
                                    <span className="text-gray-500 text-lg">➖</span>
                                    <span className="ml-1 text-xs text-gray-500 font-medium">DOJI</span>
        </div>
      );
    }
                              return (
                                <div className="flex items-center justify-center">
                                  <span className={`text-lg ${isGreen ? 'text-green-600' : 'text-red-600'}`}>
                                    {isGreen ? '🟢' : '🔴'}
                                  </span>
                                  <span className={`ml-1 text-xs font-medium ${isGreen ? 'text-green-700' : 'text-red-700'}`}>
                                    {isGreen ? 'G' : 'R'}
                                  </span>
                                </div>
                              );
                            })()}
                          </td>
                          <td className={`px-4 py-3 whitespace-nowrap text-sm font-mono ${
                            isDarkMode ? 'text-gray-400' : 'text-gray-500'
                          }`}>
                            {row.volume ? row.volume.toLocaleString() : 'N/A'}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </>
          ) : (
            <div className={`p-8 text-center ${
              isDarkMode ? 'text-gray-400' : 'text-gray-500'
            }`}>
              <div className="text-4xl mb-4">📊</div>
              <h3 className={`text-lg font-medium mb-2 ${
                isDarkMode ? 'text-gray-300' : 'text-gray-900'
              }`}>No Data Available</h3>
              <p>No trading data available for {selectedSymbol.toUpperCase()}_{selectedTimeframe}</p>
              <button
                onClick={handleRefresh}
                className={`mt-4 px-4 py-2 rounded-md transition-colors duration-200 ${
                  isDarkMode 
                    ? 'bg-blue-600 text-white hover:bg-blue-700' 
                    : 'bg-blue-500 text-white hover:bg-blue-600'
                }`}
              >
                🔄 Try Refresh
              </button>
            </div>
          )}
        </div>

        {/* Tables Overview */}
        {tables.length > 0 && (
          <div className={`mt-8 rounded-lg shadow-md overflow-hidden transition-colors duration-200 ${
            isDarkMode ? 'bg-gray-800' : 'bg-white'
          }`}>
            <div className={`px-6 py-4 border-b transition-colors duration-200 ${
              isDarkMode ? 'border-gray-700' : 'border-gray-200'
            }`}>
              <div className="flex items-center">
              <h3 className={`text-lg font-semibold ${
                isDarkMode ? 'text-white' : 'text-gray-900'
              }`}>Available Tables</h3>
                <InfoTooltip id="available-tables" content={
                  <div>
                    <p className="font-semibold mb-2">📊 Database Tables</p>
                    <p className="mb-2">Quick overview of all trading data tables in your database:</p>
                    <ul className="list-disc list-inside space-y-1 text-xs">
                      <li><strong>Table Name:</strong> Format is symbol_timeframe (e.g., es_1d)</li>
                      <li><strong>Records:</strong> Total number of data points stored</li>
                      <li><strong>Latest Data:</strong> Most recent timestamp in the table</li>
                      <li><strong>Load Data:</strong> One-click loading of table data</li>
                    </ul>
                    <p className="mt-2 text-xs">Use this to quickly switch between different symbols and timeframes.</p>
                  </div>
                } isDarkMode={isDarkMode} />
              </div>
              <p className={`text-sm mt-1 ${
                isDarkMode ? 'text-gray-400' : 'text-gray-600'
              }`}>Click any table to quickly load its data • Total: {tables.length} tables</p>
            </div>
            <div className="overflow-x-auto">
              <table className="min-w-full divide-y divide-gray-200">
                <thead className={`transition-colors duration-200 ${
                  isDarkMode ? 'bg-gray-700' : 'bg-gray-50'
                }`}>
                  <tr>
                    <th className={`px-6 py-3 text-left text-xs font-medium uppercase tracking-wider ${
                      isDarkMode ? 'text-gray-400' : 'text-gray-500'
                    }`}>
                      Table Name
                    </th>
                    <th className={`px-6 py-3 text-left text-xs font-medium uppercase tracking-wider ${
                      isDarkMode ? 'text-gray-400' : 'text-gray-500'
                    }`}>
                      Symbol
                    </th>
                    <th className={`px-6 py-3 text-left text-xs font-medium uppercase tracking-wider ${
                      isDarkMode ? 'text-gray-400' : 'text-gray-500'
                    }`}>
                      Timeframe
                    </th>
                    <th className={`px-6 py-3 text-left text-xs font-medium uppercase tracking-wider ${
                      isDarkMode ? 'text-gray-400' : 'text-gray-500'
                    }`}>
                      Records
                    </th>
                    <th className={`px-6 py-3 text-left text-xs font-medium uppercase tracking-wider ${
                      isDarkMode ? 'text-gray-400' : 'text-gray-500'
                    }`}>
                      Latest Data
                    </th>
                    <th className={`px-6 py-3 text-left text-xs font-medium uppercase tracking-wider ${
                      isDarkMode ? 'text-gray-400' : 'text-gray-500'
                    }`}>
                      Actions
                    </th>
                  </tr>
                </thead>
                <tbody className={`divide-y transition-colors duration-200 ${
                  isDarkMode ? 'bg-gray-800 divide-gray-700' : 'bg-white divide-gray-200'
                }`}>
                  {tables.map((table, index) => (
                    <tr key={index} className={`transition-colors duration-200 ${
                      isDarkMode ? 'hover:bg-gray-700' : 'hover:bg-gray-50'
                    }`}>
                      <td className={`px-6 py-4 whitespace-nowrap text-sm font-medium font-mono ${
                        isDarkMode ? 'text-gray-300' : 'text-gray-900'
                      }`}>
                        {table.table_name}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <span className={`text-sm font-medium ${getSymbolColor(table.symbol)} px-2 py-1 rounded ${
                          isDarkMode ? 'bg-gray-700' : 'bg-gray-100'
                        }`}>
                          {table.symbol.toUpperCase()}
                        </span>
                      </td>
                      <td className={`px-6 py-4 whitespace-nowrap text-sm font-medium ${
                        isDarkMode ? 'text-gray-300' : 'text-gray-900'
                      }`}>
                        {table.timeframe}
                      </td>
                      <td className={`px-6 py-4 whitespace-nowrap text-sm ${
                        isDarkMode ? 'text-gray-300' : 'text-gray-900'
                      }`}>
                        <span className={`px-2 py-1 rounded font-mono transition-colors duration-200 ${
                          isDarkMode 
                            ? 'bg-blue-800 text-blue-300' 
                            : 'bg-blue-100 text-blue-800'
                        }`}>
                          {table.row_count ? table.row_count.toLocaleString() : 'N/A'}
                        </span>
                      </td>
                      <td className={`px-6 py-4 whitespace-nowrap text-sm font-mono ${
                        isDarkMode ? 'text-gray-400' : 'text-gray-500'
                      }`}>
                        {table.latest_timestamp ? formatTimestamp(table.latest_timestamp) : 'N/A'}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm">
                        <button
                          onClick={() => quickSelectTable(table)}
                          className={`inline-flex items-center px-3 py-1 rounded-md font-medium transition-colors duration-200 ${
                            isDarkMode 
                              ? 'bg-blue-800 text-blue-300 hover:bg-blue-700' 
                              : 'bg-blue-100 text-blue-700 hover:bg-blue-200'
                          }`}
                        >
                          📊 Load Data
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
          </>
        )}
      </div>
    </div>
  );
};

export default TradingDashboard; 