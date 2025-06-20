import PropTypes from 'prop-types';
import InfoTooltip from '../shared/InfoTooltip';
import { SEARCH_FUNCTIONALITY_INFO } from '../../utils/tooltipContent.jsx';

/**
 * AdvancedFilters - Search and filter controls for the data table
 */
const AdvancedFilters = ({
  searchTerm,
  setSearchTerm,
  selectedSearchColumn,
  setSelectedSearchColumn,
  sortColumn,
  setSortColumn,
  sortOrder,
  setSortOrder,
  currentPage,
  setCurrentPage,
  isDarkMode
}) => {
  const handleResetFilters = () => {
    setSearchTerm('');
    setSelectedSearchColumn('all');
    setSortColumn('timestamp');
    setSortOrder('desc');
    setCurrentPage(1);
  };

  return (
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
            <InfoTooltip 
              id="search-functionality" 
              content={SEARCH_FUNCTIONALITY_INFO.content} 
              isDarkMode={isDarkMode} 
            />
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
            onClick={handleResetFilters}
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
  );
};

AdvancedFilters.propTypes = {
  searchTerm: PropTypes.string.isRequired,
  setSearchTerm: PropTypes.func.isRequired,
  selectedSearchColumn: PropTypes.string.isRequired,
  setSelectedSearchColumn: PropTypes.func.isRequired,
  sortColumn: PropTypes.string.isRequired,
  setSortColumn: PropTypes.func.isRequired,
  sortOrder: PropTypes.string.isRequired,
  setSortOrder: PropTypes.func.isRequired,
  currentPage: PropTypes.number.isRequired,
  setCurrentPage: PropTypes.func.isRequired,
  isDarkMode: PropTypes.bool.isRequired
};

export default AdvancedFilters; 