import React from 'react';
import { useTrading } from '../../context/TradingContext';

const InfoTooltip = ({ id, content, isDarkMode, asSpan = false }) => {
  const [activeTooltip, setActiveTooltip] = React.useState(null);
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
          <div className={`absolute top-full left-1/2 transform -translate-x-1/2 w-0 h-0 border-l-[8px] border-r-[8px] border-t-[8px] border-l-transparent border-r-transparent ${
            isDarkMode ? 'border-t-gray-800' : 'border-t-white'
          }`}></div>
        </div>
      )}
    </div>
  );
};

const SelectedCandlesPanel = ({ isDarkMode, canSelectCandles = false }) => {
  const { selectedCandles, clearSelectedCandles, removeSelectedCandle } = useTrading();

  const formatPrice = (price) => {
    return price ? price.toFixed(4) : 'N/A';
  };

  const formatTimestamp = (timestamp) => {
    return new Date(timestamp).toLocaleString();
  };

  const getCandleType = (candle) => {
    if (candle.close > candle.open) return { type: 'BULL', color: 'text-green-500', emoji: '🟢' };
    if (candle.close < candle.open) return { type: 'BEAR', color: 'text-red-500', emoji: '🔴' };
    return { type: 'DOJI', color: 'text-gray-500', emoji: '➖' };
  };

  const handleRemoveCandle = (candleId) => {
    removeSelectedCandle(candleId);
  };

  return (
    <div className={`rounded-lg shadow-md p-6 mb-8 transition-colors duration-200 ${
      isDarkMode ? 'bg-gray-800' : 'bg-white'
    }`}>
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center">
          <h3 className={`text-lg font-semibold ${
            isDarkMode ? 'text-white' : 'text-gray-900'
          }`}>
            Selected Candles
            {selectedCandles.length > 0 && (
              <span className={`ml-2 px-2 py-1 text-sm rounded-full ${
                isDarkMode ? 'bg-blue-900/30 text-blue-300' : 'bg-blue-100 text-blue-700'
              }`}>
                {selectedCandles.length}
              </span>
            )}
          </h3>
          <InfoTooltip id="selected-candles" content={
            <div>
              <p className="font-semibold mb-2">🕯️ Selected Candles</p>
              <p className="mb-2">Candles selected for analysis across all dashboards:</p>
              <ul className="list-disc list-inside space-y-1 text-xs">
                <li><strong>Synchronized:</strong> Selection is shared across Data, Vector, Chart, and LLM dashboards</li>
                <li><strong>Analysis Ready:</strong> Selected candles can be used for pattern matching and AI analysis</li>
                <li><strong>Persistent:</strong> Selections are saved and restored across browser sessions</li>
                <li><strong>Interactive:</strong> Click on candles in compatible dashboards to select/deselect</li>
              </ul>
              <p className="mt-2 text-xs">Selected candles maintain their symbol, timeframe, and complete OHLCV data for analysis.</p>
            </div>
          } isDarkMode={isDarkMode} />
        </div>
        
        <div className="flex gap-2">
          {selectedCandles.length > 0 && (
            <button
              onClick={clearSelectedCandles}
              className={`px-3 py-2 text-sm rounded-md transition-colors ${
                isDarkMode 
                  ? 'bg-red-800 hover:bg-red-700 text-red-300' 
                  : 'bg-red-100 hover:bg-red-200 text-red-700'
              }`}
            >
              🗑️ Clear All
            </button>
          )}
        </div>
      </div>

      {selectedCandles.length === 0 ? (
        <div className={`text-center py-8 ${
          isDarkMode ? 'text-gray-400' : 'text-gray-500'
        }`}>
          <div className="text-4xl mb-4">🕯️</div>
          <h4 className={`text-lg font-medium mb-2 ${
            isDarkMode ? 'text-gray-300' : 'text-gray-700'
          }`}>No Candles Selected</h4>
          <p className="text-sm">
            {canSelectCandles 
              ? 'Click on candles in the table or chart to select them for analysis'
              : 'Navigate to the Data, Vector, or Chart dashboard to select candles for analysis'
            }
          </p>
        </div>
      ) : (
        <div className="space-y-4">
          {/* Summary Stats */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <div className={`p-3 rounded-lg border transition-colors duration-200 ${
              isDarkMode ? 'bg-gray-700 border-gray-600' : 'bg-gray-50 border-gray-200'
            }`}>
              <div className={`text-sm ${isDarkMode ? 'text-gray-400' : 'text-gray-600'}`}>Total Selected</div>
              <div className="text-xl font-bold text-blue-500">{selectedCandles.length}</div>
            </div>
            
            <div className={`p-3 rounded-lg border transition-colors duration-200 ${
              isDarkMode ? 'bg-gray-700 border-gray-600' : 'bg-gray-50 border-gray-200'
            }`}>
              <div className={`text-sm ${isDarkMode ? 'text-gray-400' : 'text-gray-600'}`}>Symbols</div>
              <div className="text-xl font-bold text-green-500">
                {new Set(selectedCandles.map(c => c.symbol)).size}
              </div>
            </div>
            
            <div className={`p-3 rounded-lg border transition-colors duration-200 ${
              isDarkMode ? 'bg-gray-700 border-gray-600' : 'bg-gray-50 border-gray-200'
            }`}>
              <div className={`text-sm ${isDarkMode ? 'text-gray-400' : 'text-gray-600'}`}>Timeframes</div>
              <div className="text-xl font-bold text-purple-500">
                {new Set(selectedCandles.map(c => c.timeframe)).size}
              </div>
            </div>
            
            <div className={`p-3 rounded-lg border transition-colors duration-200 ${
              isDarkMode ? 'bg-gray-700 border-gray-600' : 'bg-gray-50 border-gray-200'
            }`}>
              <div className={`text-sm ${isDarkMode ? 'text-gray-400' : 'text-gray-600'}`}>Bullish</div>
              <div className="text-xl font-bold text-green-500">
                {selectedCandles.filter(c => c.close > c.open).length}
              </div>
            </div>
          </div>

          {/* Selected Candles List */}
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className={`transition-colors duration-200 ${
                isDarkMode ? 'bg-gray-700' : 'bg-gray-50'
              }`}>
                <tr>
                  <th className={`px-4 py-3 text-left text-xs font-medium uppercase tracking-wider ${
                    isDarkMode ? 'text-gray-400' : 'text-gray-500'
                  }`}>Symbol</th>
                  <th className={`px-4 py-3 text-left text-xs font-medium uppercase tracking-wider ${
                    isDarkMode ? 'text-gray-400' : 'text-gray-500'
                  }`}>Timeframe</th>
                  <th className={`px-4 py-3 text-left text-xs font-medium uppercase tracking-wider ${
                    isDarkMode ? 'text-gray-400' : 'text-gray-500'
                  }`}>Timestamp</th>
                  <th className={`px-4 py-3 text-left text-xs font-medium uppercase tracking-wider ${
                    isDarkMode ? 'text-gray-400' : 'text-gray-500'
                  }`}>OHLC</th>
                  <th className={`px-4 py-3 text-left text-xs font-medium uppercase tracking-wider ${
                    isDarkMode ? 'text-gray-400' : 'text-gray-500'
                  }`}>Type</th>
                  <th className={`px-4 py-3 text-left text-xs font-medium uppercase tracking-wider ${
                    isDarkMode ? 'text-gray-400' : 'text-gray-500'
                  }`}>Actions</th>
                </tr>
              </thead>
              <tbody className={`divide-y transition-colors duration-200 ${
                isDarkMode ? 'bg-gray-800 divide-gray-700' : 'bg-white divide-gray-200'
              }`}>
                {selectedCandles.map((candle, index) => {
                  const candleType = getCandleType(candle);
                  return (
                    <tr key={candle.id} className={`transition-colors duration-200 ${
                      isDarkMode ? 'hover:bg-gray-700' : 'hover:bg-gray-50'
                    }`}>
                      <td className={`px-4 py-3 whitespace-nowrap text-sm font-medium ${
                        isDarkMode ? 'text-white' : 'text-gray-900'
                      }`}>
                        <span className={`px-2 py-1 rounded ${
                          isDarkMode ? 'bg-blue-900/30 text-blue-300' : 'bg-blue-100 text-blue-700'
                        }`}>
                          {candle.symbol.toUpperCase()}
                        </span>
                      </td>
                      <td className={`px-4 py-3 whitespace-nowrap text-sm ${
                        isDarkMode ? 'text-gray-300' : 'text-gray-700'
                      }`}>
                        {candle.timeframe}
                      </td>
                      <td className={`px-4 py-3 whitespace-nowrap text-sm font-mono ${
                        isDarkMode ? 'text-gray-300' : 'text-gray-700'
                      }`}>
                        {formatTimestamp(candle.timestamp)}
                      </td>
                      <td className={`px-4 py-3 whitespace-nowrap text-sm font-mono ${
                        isDarkMode ? 'text-gray-300' : 'text-gray-700'
                      }`}>
                        O: {formatPrice(candle.open)} | H: {formatPrice(candle.high)} | 
                        L: {formatPrice(candle.low)} | C: {formatPrice(candle.close)}
                      </td>
                      <td className={`px-4 py-3 whitespace-nowrap text-sm ${candleType.color}`}>
                        <div className="flex items-center">
                          <span className="mr-1">{candleType.emoji}</span>
                          <span className="font-medium">{candleType.type}</span>
                        </div>
                      </td>
                      <td className="px-4 py-3 whitespace-nowrap text-sm">
                        <button
                          onClick={() => handleRemoveCandle(candle.id)}
                          className={`text-red-500 hover:text-red-700 transition-colors duration-200`}
                          title="Remove this candle from selection"
                        >
                          🗑️
                        </button>
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

export default SelectedCandlesPanel; 