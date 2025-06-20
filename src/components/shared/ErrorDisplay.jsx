import PropTypes from 'prop-types';

/**
 * ErrorDisplay - A reusable error display component
 * @param {boolean} isDarkMode - Current theme state
 * @param {string} error - Error message to display
 * @param {Function} onRetry - Optional retry function
 */
const ErrorDisplay = ({ isDarkMode, error, onRetry }) => {
  return (
    <div className={`rounded-lg p-4 border transition-colors duration-200 ${
      isDarkMode 
        ? 'bg-red-900/20 border-red-800' 
        : 'bg-red-50 border-red-200'
    }`}>
      <div className="flex items-center">
        <div className="text-red-600 mr-2">⚠️</div>
        <div className="text-red-500 text-sm font-medium">Error</div>
      </div>
      <div className="text-red-500 text-sm mt-1">{error}</div>
      {onRetry && (
        <button 
          onClick={onRetry}
          className="mt-2 px-3 py-1 bg-red-600 text-white rounded text-sm hover:bg-red-700 transition-colors duration-200"
        >
          Retry
        </button>
      )}
    </div>
  );
};

ErrorDisplay.propTypes = {
  isDarkMode: PropTypes.bool.isRequired,
  error: PropTypes.string.isRequired,
  onRetry: PropTypes.func
};

export default ErrorDisplay; 