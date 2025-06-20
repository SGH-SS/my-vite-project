import PropTypes from 'prop-types';
import InfoTooltip from '../shared/InfoTooltip';
import LoadingSpinner from '../shared/LoadingSpinner';
import Pagination from './Pagination';
import { formatPrice, formatTimestamp, getCandleType } from '../../utils/formatters';
import { DATA_ORDER_INFO } from '../../utils/tooltipContent.jsx';

/**
 * DataTable - Main trading data table with sorting, selection, and pagination
 */
const DataTable = ({
  tradingData,
  loading,
  selectedSymbol,
  selectedTimeframe,
  sortOrder,
  sortColumn,
  onColumnSort,
  selectedRows,
  onRowSelect,
  onSelectAll,
  searchTerm,
  filteredData,
  currentPage,
  rowLimit,
  totalPages,
  onPageChange,
  onExportCSV,
  onBulkDelete,
  onRefresh,
  lastFetchInfo,
  isDarkMode
}) => {
  const tableColumns = [
    { key: 'timestamp', label: 'Timestamp', icon: '📅', sortable: true },
    { key: 'open', label: 'Open', icon: '📈', sortable: true },
    { key: 'high', label: 'High', icon: '🔺', sortable: true },
    { key: 'low', label: 'Low', icon: '🔻', sortable: true },
    { key: 'close', label: 'Close', icon: '💰', sortable: true },
    { key: 'candle_type', label: 'Candle', icon: '🕯️', sortable: false },
    { key: 'volume', label: 'Volume', icon: '📊', sortable: true }
  ];

  const renderTableHeader = () => (
    <div className={`px-6 py-4 border-b flex items-center justify-between transition-colors duration-200 ${
      isDarkMode ? 'border-gray-700' : 'border-gray-200'
    }`}>
      <div>
        <h3 className={`text-lg font-semibold ${
          isDarkMode ? 'text-white' : 'text-gray-900'
        }`}>
          {selectedSymbol.toUpperCase()} - {selectedTimeframe} Data
          {tradingData && (
            <span className={`ml-2 text-sm ${
              isDarkMode ? 'text-gray-400' : 'text-gray-500'
            }`}>
              ({(() => {
                if (!searchTerm) return tradingData.count || tradingData.data?.length || 0;
                const filteredCount = filteredData.length;
                return `${filteredCount} filtered / ${tradingData.count || tradingData.data?.length || 0} total`;
              })()} records)
            </span>
          )}
          <br />
          <div className="flex items-center">
            <span className={`text-sm font-medium ${sortOrder === 'asc' ? 'text-green-600' : 'text-blue-600'}`}>
              {sortOrder === 'asc' ? '🟢 Showing OLDEST data first' : '🔵 Showing NEWEST data first'} 
              {lastFetchInfo && (
                <span className={`ml-2 ${
                  isDarkMode ? 'text-gray-400' : 'text-gray-500'
                }`}>
                  (Backend sorting: {lastFetchInfo.backendHandledSorting ? '✅' : '❌ using client-side fallback'})
                </span>
              )}
            </span>
            <InfoTooltip 
              id="data-order" 
              content={DATA_ORDER_INFO.content} 
              isDarkMode={isDarkMode} 
            />
          </div>
        </h3>
        {selectedRows.size > 0 && (
          <p className="text-sm text-blue-600 mt-1">
            {selectedRows.size} row(s) selected
          </p>
        )}
      </div>
      <div className="flex gap-2">
        <button
          onClick={onExportCSV}
          disabled={!tradingData?.data?.length}
          className={`px-3 py-2 text-sm rounded-md transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-1 ${
            isDarkMode 
              ? 'bg-green-800 hover:bg-green-700 text-green-300' 
              : 'bg-green-100 hover:bg-green-200 text-green-700'
          }`}
        >
          📊 Export CSV
        </button>
        {selectedRows.size > 0 && (
          <>
            <button
              onClick={onBulkDelete}
              className={`px-3 py-2 text-sm rounded-md transition-colors flex items-center gap-1 ${
                isDarkMode 
                  ? 'bg-red-800 hover:bg-red-700 text-red-300' 
                  : 'bg-red-100 hover:bg-red-200 text-red-700'
              }`}
            >
              🗑️ Delete Selected
            </button>
            <button
              onClick={() => onRowSelect('clear')}
              className={`px-3 py-2 text-sm rounded-md transition-colors ${
                isDarkMode 
                  ? 'bg-gray-700 hover:bg-gray-600 text-gray-300' 
                  : 'bg-gray-100 hover:bg-gray-200 text-gray-700'
              }`}
            >
              ✖️ Clear Selection
            </button>
          </>
        )}
      </div>
    </div>
  );

  const renderTableContent = () => {
    if (loading) {
      return <LoadingSpinner isDarkMode={isDarkMode} message="Loading trading data..." />;
    }

    if (!tradingData || !tradingData.data || tradingData.data.length === 0) {
      return (
        <div className={`p-8 text-center ${
          isDarkMode ? 'text-gray-400' : 'text-gray-500'
        }`}>
          <div className="text-4xl mb-4">📊</div>
          <h3 className={`text-lg font-medium mb-2 ${
            isDarkMode ? 'text-gray-300' : 'text-gray-900'
          }`}>No Data Available</h3>
          <p>No trading data available for {selectedSymbol.toUpperCase()}_{selectedTimeframe}</p>
          <button
            onClick={onRefresh}
            className={`mt-4 px-4 py-2 rounded-md transition-colors duration-200 ${
              isDarkMode 
                ? 'bg-blue-600 text-white hover:bg-blue-700' 
                : 'bg-blue-500 text-white hover:bg-blue-600'
            }`}
          >
            🔄 Try Refresh
          </button>
        </div>
      );
    }

    return (
      <>
        {/* Enhanced Pagination - TOP */}
        {totalPages > 1 && (
          <Pagination
            currentPage={currentPage}
            totalPages={totalPages}
            totalRecords={tradingData.total_count || tradingData.count || tradingData.data.length}
            rowLimit={rowLimit}
            onPageChange={onPageChange}
            isDarkMode={isDarkMode}
          />
        )}

        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200">
            <thead className={`transition-colors duration-200 ${
              isDarkMode ? 'bg-gray-700' : 'bg-gray-50'
            }`}>
              <tr>
                <th className="px-4 py-3 text-left">
                  <input
                    type="checkbox"
                    checked={selectedRows.size > 0 && selectedRows.size === filteredData.length}
                    onChange={onSelectAll}
                    className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                  />
                </th>
                <th className={`px-4 py-3 text-left text-xs font-medium uppercase tracking-wider ${
                  isDarkMode ? 'text-gray-400' : 'text-gray-500'
                }`}>
                  #
                </th>
                {tableColumns.map(({ key, label, icon, sortable }) => (
                  <th
                    key={key}
                    onClick={sortable ? () => onColumnSort(key) : undefined}
                    className={`px-4 py-3 text-left text-xs font-medium uppercase tracking-wider select-none transition-colors duration-200 ${
                      isDarkMode ? 'text-gray-400' : 'text-gray-500'
                    } ${sortable 
                      ? isDarkMode 
                        ? 'cursor-pointer hover:bg-gray-600' 
                        : 'cursor-pointer hover:bg-gray-100'
                      : 'cursor-default'}`}
                  >
                    <div className="flex items-center">
                      <span className="mr-1">{icon}</span>
                      {label}
                      {sortable && sortColumn === key && (
                        <span className="ml-1 text-blue-600">
                          {sortOrder === 'asc' ? '↑' : '↓'}
                        </span>
                      )}
                      {sortable && <span className="ml-1 text-gray-400">⇅</span>}
                    </div>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody className={`divide-y transition-colors duration-200 ${
              isDarkMode ? 'bg-gray-800 divide-gray-700' : 'bg-white divide-gray-200'
            }`}>
              {filteredData.map((row, index) => {
                const rowNumber = ((currentPage - 1) * rowLimit) + index + 1;
                const candleType = getCandleType(row.open, row.close);
                
                return (
                  <tr 
                    key={index} 
                    className={`transition-colors duration-200 ${
                      selectedRows.has(index) 
                        ? isDarkMode 
                          ? 'bg-blue-900/30 border-l-4 border-blue-400' 
                          : 'bg-blue-50 border-l-4 border-blue-500'
                        : isDarkMode 
                          ? 'hover:bg-gray-700' 
                          : 'hover:bg-gray-50'
                    }`}
                  >
                    <td className="px-4 py-3 whitespace-nowrap">
                      <input
                        type="checkbox"
                        checked={selectedRows.has(index)}
                        onChange={() => onRowSelect(index)}
                        className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                      />
                    </td>
                    <td className={`px-4 py-3 whitespace-nowrap text-sm font-mono ${
                      isDarkMode ? 'text-gray-400' : 'text-gray-500'
                    }`}>
                      {rowNumber}
                    </td>
                    <td className={`px-4 py-3 whitespace-nowrap text-sm font-mono ${
                      isDarkMode ? 'text-gray-300' : 'text-gray-900'
                    }`}>
                      {formatTimestamp(row.timestamp)}
                    </td>
                    <td className={`px-4 py-3 whitespace-nowrap text-sm font-mono ${
                      isDarkMode ? 'text-gray-300' : 'text-gray-900'
                    }`}>
                      {formatPrice(row.open)}
                    </td>
                    <td className={`px-4 py-3 whitespace-nowrap text-sm font-mono font-semibold ${
                      isDarkMode ? 'text-gray-300' : 'text-gray-900'
                    }`}>
                      {formatPrice(row.high)}
                    </td>
                    <td className={`px-4 py-3 whitespace-nowrap text-sm font-mono font-semibold ${
                      isDarkMode ? 'text-gray-300' : 'text-gray-900'
                    }`}>
                      {formatPrice(row.low)}
                    </td>
                    <td className={`px-4 py-3 whitespace-nowrap text-sm font-mono font-bold ${
                      isDarkMode ? 'text-gray-200' : 'text-gray-900'
                    }`}>
                      {formatPrice(row.close)}
                    </td>
                    <td className="px-4 py-3 whitespace-nowrap text-sm text-center">
                      <div className="flex items-center justify-center">
                        <span className={`text-lg ${candleType.color}`}>
                          {candleType.icon}
                        </span>
                        <span className={`ml-1 text-xs font-medium ${
                          candleType.type === 'DOJI' ? 'text-gray-500' : candleType.color
                        }`}>
                          {candleType.type === 'BULL' ? 'G' : candleType.type === 'BEAR' ? 'R' : candleType.type}
                        </span>
                      </div>
                    </td>
                    <td className={`px-4 py-3 whitespace-nowrap text-sm font-mono ${
                      isDarkMode ? 'text-gray-400' : 'text-gray-500'
                    }`}>
                      {row.volume ? row.volume.toLocaleString() : 'N/A'}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </>
    );
  };

  return (
    <div className={`rounded-lg shadow-md overflow-hidden transition-colors duration-200 ${
      isDarkMode ? 'bg-gray-800' : 'bg-white'
    }`}>
      {renderTableHeader()}
      {renderTableContent()}
    </div>
  );
};

DataTable.propTypes = {
  tradingData: PropTypes.shape({
    data: PropTypes.array,
    count: PropTypes.number,
    total_count: PropTypes.number
  }),
  loading: PropTypes.bool.isRequired,
  selectedSymbol: PropTypes.string.isRequired,
  selectedTimeframe: PropTypes.string.isRequired,
  sortOrder: PropTypes.string.isRequired,
  sortColumn: PropTypes.string.isRequired,
  onColumnSort: PropTypes.func.isRequired,
  selectedRows: PropTypes.instanceOf(Set).isRequired,
  onRowSelect: PropTypes.func.isRequired,
  onSelectAll: PropTypes.func.isRequired,
  searchTerm: PropTypes.string,
  filteredData: PropTypes.array.isRequired,
  currentPage: PropTypes.number.isRequired,
  rowLimit: PropTypes.number.isRequired,
  totalPages: PropTypes.number.isRequired,
  onPageChange: PropTypes.func.isRequired,
  onExportCSV: PropTypes.func.isRequired,
  onBulkDelete: PropTypes.func.isRequired,
  onRefresh: PropTypes.func.isRequired,
  lastFetchInfo: PropTypes.object,
  isDarkMode: PropTypes.bool.isRequired
};

export default DataTable;
