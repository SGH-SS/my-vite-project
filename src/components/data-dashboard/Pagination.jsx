import PropTypes from 'prop-types';

/**
 * Pagination - Reusable pagination component for table navigation
 */
const Pagination = ({
  currentPage,
  totalPages,
  totalRecords,
  rowLimit,
  onPageChange,
  isDarkMode
}) => {
  const startRecord = ((currentPage - 1) * rowLimit) + 1;
  const endRecord = Math.min(currentPage * rowLimit, totalRecords);

  return (
    <div className={`px-6 py-4 border-b flex items-center justify-between transition-colors duration-200 ${
      isDarkMode ? 'bg-gray-800 border-gray-700' : 'bg-white border-gray-200'
    }`}>
      <div className={`flex items-center text-sm ${
        isDarkMode ? 'text-gray-300' : 'text-gray-700'
      }`}>
        <span>
          Showing {startRecord} to {endRecord} of {totalRecords?.toLocaleString()} results
        </span>
      </div>
      <div className="flex items-center space-x-2">
        <button
          onClick={() => {
            console.log(`🔸 First button clicked - setting page to 1 (was ${currentPage}), totalPages: ${totalPages}`);
            onPageChange(1);
          }}
          disabled={currentPage === 1}
          className={`px-3 py-2 text-sm border rounded-md disabled:opacity-50 disabled:cursor-not-allowed transition-colors duration-200 ${
            isDarkMode 
              ? 'border-gray-600 text-gray-300 hover:bg-gray-700' 
              : 'border-gray-300 text-gray-700 hover:bg-gray-50'
          }`}
        >
          ⏮ First
        </button>
        <button
          onClick={() => {
            const newPage = Math.max(1, currentPage - 1);
            console.log(`🔸 Previous button clicked - setting page to ${newPage} (was ${currentPage}), totalPages: ${totalPages}`);
            onPageChange(newPage);
          }}
          disabled={currentPage === 1}
          className={`px-3 py-2 text-sm border rounded-md disabled:opacity-50 disabled:cursor-not-allowed transition-colors duration-200 ${
            isDarkMode 
              ? 'border-gray-600 text-gray-300 hover:bg-gray-700' 
              : 'border-gray-300 text-gray-700 hover:bg-gray-50'
          }`}
        >
          ⬅ Previous
        </button>
        <span className={`px-3 py-2 text-sm rounded-md transition-colors duration-200 ${
          isDarkMode ? 'bg-gray-700 text-gray-300' : 'bg-gray-100 text-gray-700'
        }`}>
          Page {currentPage} of {totalPages} ({totalRecords?.toLocaleString()} total records)
        </span>
        <button
          onClick={() => {
            const newPage = Math.min(totalPages, currentPage + 1);
            console.log(`🔸 Next button clicked - setting page to ${newPage} (was ${currentPage}), totalPages: ${totalPages}`);
            onPageChange(newPage);
          }}
          disabled={currentPage >= totalPages}
          className={`px-3 py-2 text-sm border rounded-md disabled:opacity-50 disabled:cursor-not-allowed transition-colors duration-200 ${
            isDarkMode 
              ? 'border-gray-600 text-gray-300 hover:bg-gray-700' 
              : 'border-gray-300 text-gray-700 hover:bg-gray-50'
          }`}
        >
          Next ➡
        </button>
        <button
          onClick={() => {
            console.log(`🔸 Last button clicked - setting page to ${totalPages} (was ${currentPage}), totalPages: ${totalPages}`);
            onPageChange(totalPages);
          }}
          disabled={currentPage >= totalPages}
          className={`px-3 py-2 text-sm border rounded-md disabled:opacity-50 disabled:cursor-not-allowed transition-colors duration-200 ${
            isDarkMode 
              ? 'border-gray-600 text-gray-300 hover:bg-gray-700' 
              : 'border-gray-300 text-gray-700 hover:bg-gray-50'
          }`}
        >
          Last ⏭
        </button>
      </div>
    </div>
  );
};

Pagination.propTypes = {
  currentPage: PropTypes.number.isRequired,
  totalPages: PropTypes.number.isRequired,
  totalRecords: PropTypes.number.isRequired,
  rowLimit: PropTypes.number.isRequired,
  onPageChange: PropTypes.func.isRequired,
  isDarkMode: PropTypes.bool.isRequired
};

export default Pagination; 