import PropTypes from 'prop-types';

/**
 * ThemeToggle - A button to toggle between light and dark themes
 * @param {boolean} isDarkMode - Current theme state
 * @param {Function} toggleTheme - Function to toggle the theme
 */
const ThemeToggle = ({ isDarkMode, toggleTheme }) => {
  return (
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
  );
};

ThemeToggle.propTypes = {
  isDarkMode: PropTypes.bool.isRequired,
  toggleTheme: PropTypes.func.isRequired
};

export default ThemeToggle; 