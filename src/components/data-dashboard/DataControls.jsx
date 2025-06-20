import PropTypes from 'prop-types';
import { SYMBOLS, TIMEFRAMES, ROW_LIMITS, SORT_ORDERS } from '../../utils/constants';
import InfoTooltip from '../shared/InfoTooltip';

/**
 * DataControls - Controls for data selection and filtering
 */
const DataControls = ({ 
  selectedSymbol,
  setSelectedSymbol,
  selectedTimeframe,
  setSelectedTimeframe,
  rowLimit,
  setRowLimit,
  sortOrder,
  setSortOrder,
  showColumnFilters,
  setShowColumnFilters,
  showDebugInfo,
  setShowDebugInfo,
  onRefresh,
  isDarkMode
}) => {
  const sortOrderTooltipContent = (
    <div>
      <p className="font-semibold mb-2">⬇️ Sort Order Options</p>
      <ul className="list-disc list-inside space-y-1 text-xs">
        <li><strong>Descending (Newest first):</strong> Shows most recent trading data</li>
        <li><strong>Ascending (Oldest first):</strong> Shows historical data from the beginning</li>
      </ul>
      <p className="mt-2 text-xs"><strong>Note:</strong> Backend sorting quality is shown in debug info. Client-side fallback ensures proper ordering.</p>
    </div>
  );

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
            onClick={() => setShowColumnFilters(!showColumnFilters)}
            className={`px-3 py-2 text-sm rounded-md transition-colors flex items-center gap-1 ${
              isDarkMode 
                ? 'bg-gray-700 hover:bg-gray-600 text-gray-300' 
                : 'bg-gray-100 hover:bg-gray-200'
            }`}
          >
            🔍 Filters {showColumnFilters ? '▼' : '▶'}
          </button>
          <button
            onClick={onRefresh}
            className={`px-3 py-2 text-sm rounded-md transition-colors flex items-center gap-1 ${
              isDarkMode 
                ? 'bg-blue-800 hover:bg-blue-700 text-blue-300' 
                : 'bg-blue-100 hover:bg-blue-200 text-blue-700'
            }`}
          >
            🔄 Refresh
          </button>
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
            {Object.values(SYMBOLS).map(symbol => (
              <option key={symbol.value} value={symbol.value}>
                {symbol.label}
              </option>
            ))}
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
            {Object.values(TIMEFRAMES).map(timeframe => (
              <option key={timeframe.value} value={timeframe.value}>
                {timeframe.label}
              </option>
            ))}
          </select>
        </div>

        <div>
          <label className={`block text-sm font-medium mb-2 ${
            isDarkMode ? 'text-gray-300' : 'text-gray-700'
          }`}>
            Rows per page
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
            {ROW_LIMITS.map(limit => (
              <option key={limit} value={limit}>
                {limit} rows
              </option>
            ))}
          </select>
        </div>

        <div>
          <div className="flex items-center mb-2">
            <label className={`block text-sm font-medium ${
              isDarkMode ? 'text-gray-300' : 'text-gray-700'
            }`}>
              Sort Order
            </label>
            <InfoTooltip 
              id="sort-order" 
              content={sortOrderTooltipContent} 
              isDarkMode={isDarkMode} 
            />
          </div>
          <select
            value={sortOrder}
            onChange={(e) => setSortOrder(e.target.value)}
            className={`w-full rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500 transition-colors duration-200 ${
              isDarkMode 
                ? 'border-gray-600 bg-gray-700 text-white' 
                : 'border-gray-300 bg-white text-gray-900'
            }`}
          >
            <option value={SORT_ORDERS.DESC}>⬇ Descending (Newest first)</option>
            <option value={SORT_ORDERS.ASC}>⬆ Ascending (Oldest first)</option>
          </select>
        </div>
      </div>
    </div>
  );
};

DataControls.propTypes = {
  selectedSymbol: PropTypes.string.isRequired,
  setSelectedSymbol: PropTypes.func.isRequired,
  selectedTimeframe: PropTypes.string.isRequired,
  setSelectedTimeframe: PropTypes.func.isRequired,
  rowLimit: PropTypes.number.isRequired,
  setRowLimit: PropTypes.func.isRequired,
  sortOrder: PropTypes.string.isRequired,
  setSortOrder: PropTypes.func.isRequired,
  showColumnFilters: PropTypes.bool.isRequired,
  setShowColumnFilters: PropTypes.func.isRequired,
  showDebugInfo: PropTypes.bool.isRequired,
  setShowDebugInfo: PropTypes.func.isRequired,
  onRefresh: PropTypes.func.isRequired,
  isDarkMode: PropTypes.bool.isRequired
};

export default DataControls; 