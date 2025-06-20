import PropTypes from 'prop-types';
import InfoTooltip from '../shared/InfoTooltip';
import { getColorFromValue } from '../../utils/vectorCalculations';
import { HEATMAP_INFO } from '../../utils/tooltipContent.jsx';

/**
 * VectorHeatmap - Displays vector data as a color-coded heatmap
 */
const VectorHeatmap = ({ vectorData, selectedVectorType, isDarkMode }) => {
  if (!vectorData?.data) return null;

  try {
    const vectors = vectorData.data
      .map(row => row[selectedVectorType])
      .filter(v => v && Array.isArray(v))
      .slice(0, 20); // Show first 20 for visibility

    if (vectors.length === 0) {
      return (
        <div className={`${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>
          No vector data available
        </div>
      );
    }

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
    
    const colorScaleInfo = (
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
    );

    const rowsInfo = (
      <div>
        <p className="font-semibold mb-2">📋 Rows</p>
        <p className="mb-2">Each row represents one trading period (candle):</p>
        <ul className="list-disc list-inside space-y-1 text-xs">
          <li><strong>Row #1:</strong> Most recent period (if sorted newest first)</li>
          <li><strong>Order:</strong> Depends on your sort setting in Data Dashboard</li>
          <li><strong>Data:</strong> Each row contains OHLC(V) data for that time period</li>
          <li><strong>Hover:</strong> Click any cell to see exact timestamp and values</li>
        </ul>
      </div>
    );

    const dimensionsInfo = (
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
    );
    
    return (
      <div className="space-y-2">
        {/* Color scale legend with info tooltip */}
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-4 text-xs">
            <div className="flex items-center">
              <span className={isDarkMode ? 'text-gray-300' : 'text-gray-600'}>Color Scale:</span>
              <InfoTooltip id="color-scale" content={colorScaleInfo} isDarkMode={isDarkMode} />
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
          <InfoTooltip id="heatmap-info" content={HEATMAP_INFO.content} isDarkMode={isDarkMode} />
        </div>
        
        {/* Dimension headers */}
        <div className="flex items-center space-x-1 text-xs">
          <div className="flex items-center">
            <div className={`w-12 ${isDarkMode ? 'text-gray-400' : 'text-gray-600'}`}>Row</div>
            <InfoTooltip id="rows-explanation" content={rowsInfo} isDarkMode={isDarkMode} />
          </div>
          <div className="flex items-center space-x-0.5">
            <div className="flex space-x-0.5">
              {Array.from({ length: maxDimensions }, (_, i) => (
                <div key={i} className={`w-8 text-center ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>
                  D{i}
                </div>
              ))}
            </div>
            <InfoTooltip id="dimensions" content={dimensionsInfo} isDarkMode={isDarkMode} />
          </div>
        </div>
        
        {/* Vector heatmap */}
        <div className="space-y-1">
          {vectors.map((vector, rowIndex) => (
            <div key={rowIndex} className="flex items-center space-x-1">
              <div className={`w-12 text-xs font-mono ${isDarkMode ? 'text-gray-400' : 'text-gray-600'}`}>
                #{rowIndex + 1}
              </div>
              <div className="flex space-x-0.5">
                {vector.slice(0, maxDimensions).map((value, dimIndex) => {
                  const color = getColorFromValue(value, globalMin, globalMax);
                  return (
                    <div
                      key={dimIndex}
                      className="w-8 h-8 rounded-sm border border-gray-300 cursor-pointer hover:scale-110 transition-transform"
                      style={{ backgroundColor: color }}
                      title={`Row ${rowIndex + 1}, Dim ${dimIndex}: ${value.toFixed(4)}\nRange: ${globalMin.toFixed(2)} to ${globalMax.toFixed(2)}`}
                    />
                  );
                })}
              </div>
            </div>
          ))}
        </div>
        
        {vectors.length === 20 && vectorData.data.length > 20 && (
          <div className={`text-xs ${isDarkMode ? 'text-gray-400' : 'text-gray-500'} text-center mt-2`}>
            Showing first 20 of {vectorData.data.length} vectors
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

VectorHeatmap.propTypes = {
  vectorData: PropTypes.shape({
    data: PropTypes.arrayOf(PropTypes.object)
  }),
  selectedVectorType: PropTypes.string.isRequired,
  isDarkMode: PropTypes.bool.isRequired
};

export default VectorHeatmap; 