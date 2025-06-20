import PropTypes from 'prop-types';
import InfoTooltip from '../shared/InfoTooltip';
import { getVectorStats } from '../../utils/vectorCalculations';
import { VECTOR_STATS_INFO } from '../../utils/tooltipContent.jsx';

/**
 * VectorStats - Displays statistics about the loaded vector data
 */
const VectorStats = ({ vectorData, selectedVectorType, isDarkMode }) => {
  if (!vectorData?.data) return null;

  try {
    const vectors = vectorData.data
      .map(row => row[selectedVectorType])
      .filter(v => v);
    
    if (vectors.length === 0) {
      return (
        <div className={`${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>
          No vector data available for {selectedVectorType}
        </div>
      );
    }

    const stats = getVectorStats(vectors);
    if (!stats) {
      return (
        <div className={`${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>
          No vector statistics available
        </div>
      );
    }

    const statCards = [
      {
        label: 'Vector Count',
        value: stats.count,
        color: 'text-blue-500',
        tooltipId: 'vector-count',
        tooltipContent: VECTOR_STATS_INFO.count
      },
      {
        label: 'Dimensions',
        value: stats.dimensions,
        color: 'text-green-500',
        tooltipId: 'vector-dimensions',
        tooltipContent: VECTOR_STATS_INFO.dimensions
      },
      {
        label: 'Range',
        value: `${stats.min.toFixed(3)} to ${stats.max.toFixed(3)}`,
        color: 'text-purple-500',
        tooltipId: 'vector-range',
        tooltipContent: VECTOR_STATS_INFO.range,
        small: true
      },
      {
        label: 'Std Dev',
        value: stats.std.toFixed(4),
        color: 'text-orange-500',
        tooltipId: 'vector-std',
        tooltipContent: VECTOR_STATS_INFO.std
      }
    ];

    return (
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
          <strong>What this shows:</strong> Statistical analysis of {vectorData.data.length} trading vectors of type "{selectedVectorType}". 
          {selectedVectorType.includes('raw') && ' Raw vectors contain direct OHLC price values.'}
          {selectedVectorType.includes('norm') && ' Normalized vectors use z-score scaling for pattern matching across different price levels.'}
          {selectedVectorType.includes('BERT') && ' BERT vectors are semantic embeddings (384 dimensions) created by converting price data to natural language.'}
        </div>
        
        {/* Stats Grid */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {statCards.map((stat) => (
            <div 
              key={stat.tooltipId}
              className={`p-4 rounded-lg border transition-colors duration-200 ${
                isDarkMode ? 'bg-gray-700 border-gray-600' : 'bg-white border-gray-200'
              }`}
            >
              <div className={`flex items-center text-xs mb-1 ${
                isDarkMode ? 'text-gray-400' : 'text-gray-500'
              }`}>
                {stat.label}
                <InfoTooltip 
                  id={stat.tooltipId} 
                  content={stat.tooltipContent} 
                  isDarkMode={isDarkMode} 
                />
              </div>
              <div className={`${stat.small ? 'text-sm' : 'text-2xl'} font-bold ${stat.color}`}>
                {stat.value}
              </div>
            </div>
          ))}
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

VectorStats.propTypes = {
  vectorData: PropTypes.shape({
    data: PropTypes.arrayOf(PropTypes.object)
  }),
  selectedVectorType: PropTypes.string.isRequired,
  isDarkMode: PropTypes.bool.isRequired
};

export default VectorStats; 