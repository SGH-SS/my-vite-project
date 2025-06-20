import PropTypes from 'prop-types';
import InfoTooltip from '../shared/InfoTooltip';
import { SYMBOLS, TIMEFRAMES, ROW_LIMITS } from '../../utils/constants';

/**
 * VectorControls - Controls for vector data selection
 */
const VectorControls = ({
  selectedSymbol,
  setSelectedSymbol,
  selectedTimeframe,
  setSelectedTimeframe,
  rowLimit,
  setRowLimit,
  onRefresh,
  isDarkMode
}) => {
  const vectorLimitInfo = (
    <div>
      <p className="font-semibold mb-2">⚡ Vector Limit</p>
      <p className="mb-2">Number of trading periods to load and analyze:</p>
      <ul className="list-disc list-inside space-y-1 text-xs">
        <li><strong>25-50:</strong> Fast loading, good for quick analysis</li>
        <li><strong>100-250:</strong> Balanced performance and data coverage</li>
        <li><strong>500+:</strong> Comprehensive analysis, slower loading</li>
      </ul>
      <p className="mt-2 text-xs">More vectors provide better pattern analysis but require more processing time and memory.</p>
    </div>
  );

  const dataInfoContent = (
    <div>
      <p className="font-semibold mb-2">📊 Data Information</p>
      <p className="mb-2">Current status of loaded vector data:</p>
      <ul className="list-disc list-inside space-y-1 text-xs">
        <li><strong>Vectors Loaded:</strong> Number of trading periods currently in memory</li>
        <li><strong>Real-time:</strong> Updates when you change symbol, timeframe, or limit</li>
        <li><strong>Performance:</strong> More vectors = more analysis depth</li>
      </ul>
      <p className="mt-2 text-xs">This shows actual data availability for the selected symbol and timeframe.</p>
    </div>
  );

  return (
    <div className={`rounded-lg shadow-md p-6 mb-8 transition-colors duration-200 ${
      isDarkMode ? 'bg-gray-800' : 'bg-white'
    }`}>
      <div className="flex items-center justify-between mb-4">
        <h3 className={`text-lg font-semibold ${
          isDarkMode ? 'text-white' : 'text-gray-900'
        }`}>Vector Data Selection & Controls</h3>
        <div className="flex gap-2">
          <button
            onClick={onRefresh}
            className={`px-3 py-2 text-sm rounded-md transition-colors flex items-center gap-1 ${
              isDarkMode 
                ? 'bg-blue-800 hover:bg-blue-700 text-blue-300' 
                : 'bg-blue-100 hover:bg-blue-200 text-blue-700'
            }`}
          >
            🔄 Refresh Vectors
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
            {Object.entries(SYMBOLS).map(([key, symbol]) => (
              <option key={key} value={symbol.value}>
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
            {Object.entries(TIMEFRAMES).map(([key, timeframe]) => (
              <option key={key} value={timeframe.value}>
                {timeframe.label}
              </option>
            ))}
          </select>
        </div>

        <div>
          <div className="flex items-center mb-2">
            <label className={`block text-sm font-medium ${
              isDarkMode ? 'text-gray-300' : 'text-gray-700'
            }`}>
              Vector Limit
            </label>
            <InfoTooltip 
              id="vector-limit" 
              content={vectorLimitInfo} 
              isDarkMode={isDarkMode} 
            />
          </div>
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
                {limit} vectors
              </option>
            ))}
          </select>
        </div>

        <div>
          <div className="flex items-center mb-2">
            <label className={`block text-sm font-medium ${
              isDarkMode ? 'text-gray-300' : 'text-gray-700'
            }`}>
              Data Info
            </label>
            <InfoTooltip 
              id="data-info" 
              content={dataInfoContent} 
              isDarkMode={isDarkMode} 
            />
          </div>
          <div className={`px-3 py-2 rounded-md text-sm font-mono ${
            isDarkMode ? 'bg-gray-700 text-gray-300' : 'bg-gray-100 text-gray-700'
          }`}>
            Ready for analysis
          </div>
        </div>
      </div>
    </div>
  );
};

VectorControls.propTypes = {
  selectedSymbol: PropTypes.string.isRequired,
  setSelectedSymbol: PropTypes.func.isRequired,
  selectedTimeframe: PropTypes.string.isRequired,
  setSelectedTimeframe: PropTypes.func.isRequired,
  rowLimit: PropTypes.number.isRequired,
  setRowLimit: PropTypes.func.isRequired,
  onRefresh: PropTypes.func.isRequired,
  isDarkMode: PropTypes.bool.isRequired
};

export default VectorControls; 