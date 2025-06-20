import PropTypes from 'prop-types';

/**
 * DebugPanel - Displays debug information about data fetching
 */
const DebugPanel = ({ lastFetchInfo, isDarkMode }) => {
  if (!lastFetchInfo) return null;

  return (
    <div className={`mt-4 pt-4 border-t rounded-md p-4 transition-colors duration-200 ${
      isDarkMode 
        ? 'border-gray-600 bg-yellow-900/20' 
        : 'border-gray-200 bg-yellow-50'
    }`}>
      <h4 className={`text-md font-medium mb-3 ${
        isDarkMode ? 'text-yellow-300' : 'text-yellow-800'
      }`}>🐛 Debug Information</h4>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
        <div>
          <p><strong>Last Fetch:</strong> {lastFetchInfo.timestamp}</p>
          <p><strong>Requested Data:</strong> 
            <span className={lastFetchInfo.requestedData === 'OLDEST' ? 'text-green-600' : 'text-blue-600'}>
              {lastFetchInfo.requestedData}
            </span> {lastFetchInfo.rowLimit} records
          </p>
          <p><strong>Page:</strong> {lastFetchInfo.currentPage} (offset: {lastFetchInfo.offset})</p>
          <p><strong>Sort:</strong> {lastFetchInfo.sortOrder} by {lastFetchInfo.sortColumn}</p>
          <p><strong>Data Quality:</strong> 
            <span className={lastFetchInfo.dataQuality?.includes('GOOD') ? 'text-green-600' : 'text-red-600'}>
              {lastFetchInfo.dataQuality}
            </span>
          </p>
        </div>
        <div>
          <p><strong>Backend Sorting:</strong> 
            <span className={lastFetchInfo.backendHandledSorting ? 'text-green-600' : 'text-red-600'}>
              {lastFetchInfo.backendHandledSorting ? '✅ Working' : '❌ Not Working'}
            </span>
          </p>
          <p><strong>Total Records:</strong> {lastFetchInfo.totalRecords?.toLocaleString() || 'Unknown'}</p>
          {lastFetchInfo.actualDataRange && (
            <>
              <p><strong>Data Span:</strong> {lastFetchInfo.actualDataRange.span}</p>
              <p><strong>First Record:</strong> {new Date(lastFetchInfo.actualDataRange.first).toLocaleString()}</p>
              <p><strong>Last Record:</strong> {new Date(lastFetchInfo.actualDataRange.last).toLocaleString()}</p>
            </>
          )}
        </div>
      </div>
      <div className="mt-3 p-2 bg-yellow-100 rounded text-xs">
        <strong>💡 Tip:</strong> If "Data Quality" shows "POOR" for ascending order, your backend may only be returning recent data. 
        For true oldest records, you may need to modify your backend API to support historical data access.
      </div>
      <div className="mt-2 text-xs text-gray-600 break-all">
        <strong>API URL:</strong> {lastFetchInfo.apiUrl}
      </div>
    </div>
  );
};

DebugPanel.propTypes = {
  lastFetchInfo: PropTypes.shape({
    timestamp: PropTypes.string,
    requestedData: PropTypes.string,
    rowLimit: PropTypes.number,
    currentPage: PropTypes.number,
    offset: PropTypes.number,
    sortOrder: PropTypes.string,
    sortColumn: PropTypes.string,
    dataQuality: PropTypes.string,
    backendHandledSorting: PropTypes.bool,
    totalRecords: PropTypes.number,
    apiUrl: PropTypes.string,
    actualDataRange: PropTypes.shape({
      span: PropTypes.string,
      first: PropTypes.string,
      last: PropTypes.string
    })
  }),
  isDarkMode: PropTypes.bool.isRequired
};

export default DebugPanel; 