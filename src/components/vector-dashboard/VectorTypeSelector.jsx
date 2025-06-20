import PropTypes from 'prop-types';
import InfoTooltip from '../shared/InfoTooltip';
import { VECTOR_TYPES } from '../../utils/constants';

/**
 * VectorTypeSelector - Component for selecting vector types
 */
const VectorTypeSelector = ({ 
  selectedVectorType, 
  setSelectedVectorType, 
  availableVectors, 
  missingVectors,
  vectorData,
  isDarkMode 
}) => {
  const vectorTypesInfo = (
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
  );

  const vectorStatusInfo = (
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
  );

  const withoutVolumeInfo = (
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
  );

  const withVolumeInfo = (
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
  );

  const getVectorTypeInfo = (type) => {
    const baseInfo = (
      <div>
        <p className="font-semibold mb-2">🎯 {type.name}</p>
        <p className="mb-2">{type.description}</p>
        <ul className="list-disc list-inside space-y-1 text-xs">
          {type.key.includes('raw') && !type.hasVolume && (
            <>
              <li><strong>Raw Values:</strong> Actual price numbers from market data</li>
              <li><strong>Best for:</strong> Price analysis, absolute value comparisons</li>
              <li><strong>Dimensions:</strong> 4 (Open, High, Low, Close)</li>
            </>
          )}
          {type.key.includes('raw') && type.hasVolume && (
            <>
              <li><strong>Raw Values:</strong> Actual price + volume numbers from market data</li>
              <li><strong>Best for:</strong> Volume analysis, institutional flow detection</li>
              <li><strong>Dimensions:</strong> 5 (Open, High, Low, Close, Volume)</li>
            </>
          )}
          {type.key.includes('norm') && !type.hasVolume && (
            <>
              <li><strong>Normalized:</strong> Z-score standardized values (mean=0, std=1)</li>
              <li><strong>Best for:</strong> Pattern recognition across different price levels</li>
              <li><strong>Dimensions:</strong> 4 (Open, High, Low, Close)</li>
            </>
          )}
          {type.key.includes('norm') && type.hasVolume && (
            <>
              <li><strong>Normalized:</strong> Z-score standardized OHLCV values</li>
              <li><strong>Best for:</strong> Volume-weighted pattern recognition</li>
              <li><strong>Dimensions:</strong> 5 (Open, High, Low, Close, Volume)</li>
            </>
          )}
          {type.key.includes('BERT') && !type.hasVolume && (
            <>
              <li><strong>BERT Embeddings:</strong> AI semantic representation of price patterns</li>
              <li><strong>Best for:</strong> Advanced pattern matching and similarity analysis</li>
              <li><strong>Dimensions:</strong> 384 (semantic features)</li>
            </>
          )}
          {type.key.includes('BERT') && type.hasVolume && (
            <>
              <li><strong>BERT Embeddings:</strong> AI semantic representation including volume</li>
              <li><strong>Best for:</strong> Advanced pattern matching with volume context</li>
              <li><strong>Dimensions:</strong> 384 (semantic features)</li>
            </>
          )}
        </ul>
        <p className="mt-2 text-xs">
          {type.hasVolume 
            ? "Volume data adds depth for institutional analysis and market strength detection."
            : "Click to select this vector type for analysis."}
        </p>
      </div>
    );
    return baseInfo;
  };

  return (
    <div className={`p-6 rounded-lg border transition-colors duration-200 ${
      isDarkMode 
        ? 'bg-gray-800 border-gray-700 text-white' 
        : 'bg-white border-gray-200'
    }`}>
      <div className="flex items-center mb-4">
        <h3 className={`text-lg font-semibold ${
          isDarkMode ? 'text-white' : 'text-gray-900'
        }`}>Vector Type Selection</h3>
        <InfoTooltip id="vector-types" content={vectorTypesInfo} isDarkMode={isDarkMode} />
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
            <InfoTooltip id="vector-status" content={vectorStatusInfo} isDarkMode={isDarkMode} />
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
              <InfoTooltip id="without-volume" content={withoutVolumeInfo} isDarkMode={isDarkMode} />
            </div>
            <p className={`text-xs mt-1 ${
              isDarkMode ? 'text-gray-400' : 'text-gray-600'
            }`}>OHLC (Open, High, Low, Close) only</p>
          </div>
          <div className="space-y-3">
            {VECTOR_TYPES.filter(type => !type.hasVolume).map(type => {
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
                        <InfoTooltip 
                          id={`vector-type-${type.key}`} 
                          content={getVectorTypeInfo(type)} 
                          isDarkMode={isDarkMode} 
                          asSpan={true} 
                        />
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
              <InfoTooltip id="with-volume" content={withVolumeInfo} isDarkMode={isDarkMode} />
            </div>
            <p className={`text-xs mt-1 ${
              isDarkMode ? 'text-gray-400' : 'text-gray-600'
            }`}>OHLCV (Open, High, Low, Close, Volume)</p>
          </div>
          <div className="space-y-3">
            {VECTOR_TYPES.filter(type => type.hasVolume).map(type => {
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
                        <InfoTooltip 
                          id={`vector-type-volume-${type.key}`} 
                          content={getVectorTypeInfo(type)} 
                          isDarkMode={isDarkMode} 
                          asSpan={true} 
                        />
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
  );
};

VectorTypeSelector.propTypes = {
  selectedVectorType: PropTypes.string.isRequired,
  setSelectedVectorType: PropTypes.func.isRequired,
  availableVectors: PropTypes.arrayOf(PropTypes.shape({
    key: PropTypes.string,
    name: PropTypes.string,
    color: PropTypes.string,
    description: PropTypes.string,
    hasVolume: PropTypes.bool
  })).isRequired,
  missingVectors: PropTypes.arrayOf(PropTypes.shape({
    key: PropTypes.string,
    name: PropTypes.string,
    color: PropTypes.string,
    description: PropTypes.string,
    hasVolume: PropTypes.bool
  })).isRequired,
  vectorData: PropTypes.shape({
    data: PropTypes.array
  }),
  isDarkMode: PropTypes.bool.isRequired
};

export default VectorTypeSelector; 