import PropTypes from 'prop-types';

/**
 * LoadingSpinner - A reusable loading spinner component
 * @param {boolean} isDarkMode - Current theme state
 * @param {string} message - Optional loading message
 */
const LoadingSpinner = ({ isDarkMode, message = 'Loading...' }) => {
  return (
    <div className="flex items-center justify-center h-64">
      <div className={`animate-spin rounded-full h-12 w-12 border-b-2 ${
        isDarkMode ? 'border-blue-400' : 'border-blue-600'
      }`}></div>
      <span className={`ml-3 ${
        isDarkMode ? 'text-gray-300' : 'text-gray-600'
      }`}>{message}</span>
    </div>
  );
};

LoadingSpinner.propTypes = {
  isDarkMode: PropTypes.bool.isRequired,
  message: PropTypes.string
};

export default LoadingSpinner; 