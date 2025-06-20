import PropTypes from 'prop-types';
import InfoTooltip from '../shared/InfoTooltip';
import { formatTimestamp, getSymbolColor } from '../../utils/formatters';

/**
 * TablesList - Displays a list of available database tables
 */
const TablesList = ({ tables, onTableSelect, isDarkMode }) => {
  if (!tables || tables.length === 0) return null;

  const availableTablesInfo = {
    content: (
      <div>
        <p className="font-semibold mb-2">📊 Database Tables</p>
        <p className="mb-2">Quick overview of all trading data tables in your database:</p>
        <ul className="list-disc list-inside space-y-1 text-xs">
          <li><strong>Table Name:</strong> Format is symbol_timeframe (e.g., es_1d)</li>
          <li><strong>Records:</strong> Total number of data points stored</li>
          <li><strong>Latest Data:</strong> Most recent timestamp in the table</li>
          <li><strong>Load Data:</strong> One-click loading of table data</li>
        </ul>
        <p className="mt-2 text-xs">Use this to quickly switch between different symbols and timeframes.</p>
      </div>
    )
  };

  return (
    <div className={`mt-8 rounded-lg shadow-md overflow-hidden transition-colors duration-200 ${
      isDarkMode ? 'bg-gray-800' : 'bg-white'
    }`}>
      <div className={`px-6 py-4 border-b transition-colors duration-200 ${
        isDarkMode ? 'border-gray-700' : 'border-gray-200'
      }`}>
        <div className="flex items-center">
          <h3 className={`text-lg font-semibold ${
            isDarkMode ? 'text-white' : 'text-gray-900'
          }`}>Available Tables</h3>
          <InfoTooltip 
            id="available-tables" 
            content={availableTablesInfo.content} 
            isDarkMode={isDarkMode} 
          />
        </div>
        <p className={`text-sm mt-1 ${
          isDarkMode ? 'text-gray-400' : 'text-gray-600'
        }`}>Click any table to quickly load its data • Total: {tables.length} tables</p>
      </div>
      <div className="overflow-x-auto">
        <table className="min-w-full divide-y divide-gray-200">
          <thead className={`transition-colors duration-200 ${
            isDarkMode ? 'bg-gray-700' : 'bg-gray-50'
          }`}>
            <tr>
              <th className={`px-6 py-3 text-left text-xs font-medium uppercase tracking-wider ${
                isDarkMode ? 'text-gray-400' : 'text-gray-500'
              }`}>
                Table Name
              </th>
              <th className={`px-6 py-3 text-left text-xs font-medium uppercase tracking-wider ${
                isDarkMode ? 'text-gray-400' : 'text-gray-500'
              }`}>
                Symbol
              </th>
              <th className={`px-6 py-3 text-left text-xs font-medium uppercase tracking-wider ${
                isDarkMode ? 'text-gray-400' : 'text-gray-500'
              }`}>
                Timeframe
              </th>
              <th className={`px-6 py-3 text-left text-xs font-medium uppercase tracking-wider ${
                isDarkMode ? 'text-gray-400' : 'text-gray-500'
              }`}>
                Records
              </th>
              <th className={`px-6 py-3 text-left text-xs font-medium uppercase tracking-wider ${
                isDarkMode ? 'text-gray-400' : 'text-gray-500'
              }`}>
                Latest Data
              </th>
              <th className={`px-6 py-3 text-left text-xs font-medium uppercase tracking-wider ${
                isDarkMode ? 'text-gray-400' : 'text-gray-500'
              }`}>
                Actions
              </th>
            </tr>
          </thead>
          <tbody className={`divide-y transition-colors duration-200 ${
            isDarkMode ? 'bg-gray-800 divide-gray-700' : 'bg-white divide-gray-200'
          }`}>
            {tables.map((table, index) => (
              <tr key={index} className={`transition-colors duration-200 ${
                isDarkMode ? 'hover:bg-gray-700' : 'hover:bg-gray-50'
              }`}>
                <td className={`px-6 py-4 whitespace-nowrap text-sm font-medium font-mono ${
                  isDarkMode ? 'text-gray-300' : 'text-gray-900'
                }`}>
                  {table.table_name}
                </td>
                <td className="px-6 py-4 whitespace-nowrap">
                  <span className={`text-sm font-medium ${getSymbolColor(table.symbol)} px-2 py-1 rounded ${
                    isDarkMode ? 'bg-gray-700' : 'bg-gray-100'
                  }`}>
                    {table.symbol.toUpperCase()}
                  </span>
                </td>
                <td className={`px-6 py-4 whitespace-nowrap text-sm font-medium ${
                  isDarkMode ? 'text-gray-300' : 'text-gray-900'
                }`}>
                  {table.timeframe}
                </td>
                <td className={`px-6 py-4 whitespace-nowrap text-sm ${
                  isDarkMode ? 'text-gray-300' : 'text-gray-900'
                }`}>
                  <span className={`px-2 py-1 rounded font-mono transition-colors duration-200 ${
                    isDarkMode 
                      ? 'bg-blue-800 text-blue-300' 
                      : 'bg-blue-100 text-blue-800'
                  }`}>
                    {table.row_count ? table.row_count.toLocaleString() : 'N/A'}
                  </span>
                </td>
                <td className={`px-6 py-4 whitespace-nowrap text-sm font-mono ${
                  isDarkMode ? 'text-gray-400' : 'text-gray-500'
                }`}>
                  {table.latest_timestamp ? formatTimestamp(table.latest_timestamp) : 'N/A'}
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-sm">
                  <button
                    onClick={() => onTableSelect(table)}
                    className={`inline-flex items-center px-3 py-1 rounded-md font-medium transition-colors duration-200 ${
                      isDarkMode 
                        ? 'bg-blue-800 text-blue-300 hover:bg-blue-700' 
                        : 'bg-blue-100 text-blue-700 hover:bg-blue-200'
                    }`}
                  >
                    📊 Load Data
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};

TablesList.propTypes = {
  tables: PropTypes.arrayOf(PropTypes.shape({
    table_name: PropTypes.string,
    symbol: PropTypes.string,
    timeframe: PropTypes.string,
    row_count: PropTypes.number,
    latest_timestamp: PropTypes.string
  })).isRequired,
  onTableSelect: PropTypes.func.isRequired,
  isDarkMode: PropTypes.bool.isRequired
};

export default TablesList; 