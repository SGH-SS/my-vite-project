import PropTypes from 'prop-types';

/**
 * DataStats - Displays database statistics cards
 * @param {Object} stats - Database statistics
 * @param {boolean} isDarkMode - Current theme
 */
const DataStats = ({ stats, isDarkMode }) => {
  if (!stats) return null;

  const statCards = [
    {
      icon: '📊',
      title: 'Total Tables',
      value: stats.total_tables,
      color: 'text-blue-500'
    },
    {
      icon: '💾',
      title: 'Total Records',
      value: stats.total_rows.toLocaleString(),
      color: 'text-green-500'
    },
    {
      icon: '🔄',
      title: 'Status',
      value: 'Connected',
      color: 'text-green-500'
    }
  ];

  return (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
      {statCards.map((stat, index) => (
        <div 
          key={index}
          className={`rounded-lg shadow-md p-6 transition-colors duration-200 ${
            isDarkMode ? 'bg-gray-800' : 'bg-white'
          }`}
        >
          <div className="flex items-center">
            <div className="text-3xl mr-4">{stat.icon}</div>
            <div>
              <h3 className={`text-lg font-semibold ${
                isDarkMode ? 'text-white' : 'text-gray-900'
              }`}>
                {stat.title}
              </h3>
              <p className={`text-2xl font-bold ${stat.color}`}>
                {stat.value}
              </p>
            </div>
          </div>
        </div>
      ))}
    </div>
  );
};

DataStats.propTypes = {
  stats: PropTypes.shape({
    total_tables: PropTypes.number,
    total_rows: PropTypes.number
  }),
  isDarkMode: PropTypes.bool.isRequired
};

export default DataStats; 