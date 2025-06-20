import { useState, useEffect } from 'react';
import PropTypes from 'prop-types';
import { useTrading } from '../../context/TradingContext';
import { useTradingData } from '../../hooks/useTradingData';
import { API_BASE_URL, DEFAULTS } from '../../utils/constants';
import { cleanSearchTerm, formatPrice, formatTimestamp } from '../../utils/formatters';

// Import Data Dashboard Components
import DataStats from './DataStats';
import DataControls from './DataControls';
import AdvancedFilters from './AdvancedFilters';
import DebugPanel from './DebugPanel';
import DataTable from './DataTable';
import TablesList from './TablesList';
import ErrorDisplay from '../shared/ErrorDisplay';

/**
 * DataDashboard - Main container for the data dashboard
 */
const DataDashboard = ({ isDarkMode }) => {
  // Global state from context
  const { 
    selectedSymbol, 
    setSelectedSymbol, 
    selectedTimeframe, 
    setSelectedTimeframe,
    rowLimit,
    setRowLimit
  } = useTrading();

  // Local state
  const [stats, setStats] = useState(null);
  const [tables, setTables] = useState([]);
  const [sortOrder, setSortOrder] = useState(DEFAULTS.SORT_ORDER);
  const [sortColumn, setSortColumn] = useState(DEFAULTS.SORT_COLUMN);
  const [currentPage, setCurrentPage] = useState(1);
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedSearchColumn, setSelectedSearchColumn] = useState('all');
  const [selectedRows, setSelectedRows] = useState(new Set());
  const [showColumnFilters, setShowColumnFilters] = useState(false);
  const [showDebugInfo, setShowDebugInfo] = useState(false);
  const [error, setError] = useState(null);

  // Custom hook for trading data
  const { 
    tradingData, 
    loading, 
    error: dataError, 
    lastFetchInfo, 
    refetch 
  } = useTradingData(
    selectedSymbol, 
    selectedTimeframe, 
    rowLimit, 
    sortOrder, 
    sortColumn, 
    currentPage
  );

  useEffect(() => {
    fetchStats();
    fetchTables();
  }, []);

  // Set error from data fetching
  useEffect(() => {
    if (dataError) {
      setError(dataError);
    }
  }, [dataError]);

  const fetchStats = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/stats`);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const text = await response.text();
      if (!text) {
        throw new Error('Empty response from server');
      }
      const data = JSON.parse(text);
      setStats(data);
    } catch (err) {
      console.error('Error fetching stats:', err);
      setError('Failed to fetch database stats');
    }
  };

  const fetchTables = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/tables`);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const text = await response.text();
      if (!text) {
        throw new Error('Empty response from server');
      }
      const data = JSON.parse(text);
      setTables(data);
    } catch (err) {
      console.error('Error fetching tables:', err);
      setError('Failed to fetch tables');
    }
  };

  const getFilteredData = () => {
    if (!tradingData?.data) return [];
    if (!searchTerm) return tradingData.data;
    
    const cleanTerm = cleanSearchTerm(searchTerm);
    const searchLower = cleanTerm.toLowerCase();
    
    return tradingData.data.filter(row => {
      if (selectedSearchColumn === 'all') {
        return (
          row.timestamp?.toString().toLowerCase().includes(searchLower) ||
          row.open?.toString().includes(cleanTerm) ||
          row.high?.toString().includes(cleanTerm) ||
          row.low?.toString().includes(cleanTerm) ||
          row.close?.toString().includes(cleanTerm) ||
          row.volume?.toString().includes(cleanTerm) ||
          formatTimestamp(row.timestamp).toLowerCase().includes(searchLower) ||
          formatPrice(row.open).includes(cleanTerm) ||
          formatPrice(row.high).includes(cleanTerm) ||
          formatPrice(row.low).includes(cleanTerm) ||
          formatPrice(row.close).includes(cleanTerm)
        );
      }
      
      switch (selectedSearchColumn) {
        case 'timestamp':
          return (
            row.timestamp?.toString().toLowerCase().includes(searchLower) ||
            formatTimestamp(row.timestamp).toLowerCase().includes(searchLower)
          );
        case 'open':
          return (
            row.open?.toString().includes(cleanTerm) ||
            formatPrice(row.open).includes(cleanTerm)
          );
        case 'high':
          return (
            row.high?.toString().includes(cleanTerm) ||
            formatPrice(row.high).includes(cleanTerm)
          );
        case 'low':
          return (
            row.low?.toString().includes(cleanTerm) ||
            formatPrice(row.low).includes(cleanTerm)
          );
        case 'close':
          return (
            row.close?.toString().includes(cleanTerm) ||
            formatPrice(row.close).includes(cleanTerm)
          );
        case 'volume':
          return row.volume?.toString().includes(cleanTerm);
        default:
          return true;
      }
    });
  };

  const handleColumnSort = (column) => {
    console.log(`🔄 Column sort clicked: ${column}, current sort: ${sortColumn} ${sortOrder}`);
    
    if (sortColumn === column) {
      setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc');
    } else {
      setSortColumn(column);
      setSortOrder('desc');
    }
    setCurrentPage(1);
  };

  const handleRowSelect = (index) => {
    if (index === 'clear') {
      setSelectedRows(new Set());
      return;
    }
    
    const newSelected = new Set(selectedRows);
    if (newSelected.has(index)) {
      newSelected.delete(index);
    } else {
      newSelected.add(index);
    }
    setSelectedRows(newSelected);
  };

  const handleSelectAll = () => {
    const filteredData = getFilteredData();
    if (selectedRows.size === filteredData.length && filteredData.length > 0) {
      setSelectedRows(new Set());
    } else {
      setSelectedRows(new Set(Array.from({ length: filteredData.length }, (_, i) => i)));
    }
  };

  const exportToCSV = () => {
    if (!tradingData?.data) return;
    
    const headers = ['Row', 'Timestamp', 'Open', 'High', 'Low', 'Close', 'Candle Type', 'Volume'];
    const csvContent = [
      headers.join(','),
      ...tradingData.data.map((row, index) => {
        const isGreen = row.close > row.open;
        const isEqual = row.close === row.open;
        const candleType = isEqual ? 'DOJI' : (isGreen ? 'BULL' : 'BEAR');
        
        return [
          ((currentPage - 1) * rowLimit) + index + 1,
          row.timestamp,
          row.open,
          row.high,
          row.low,
          row.close,
          candleType,
          row.volume
        ].join(',');
      })
    ].join('\n');

    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${selectedSymbol}_${selectedTimeframe}_data.csv`;
    a.click();
    window.URL.revokeObjectURL(url);
  };

  const quickSelectTable = (table) => {
    const [symbol, timeframe] = table.table_name.split('_');
    setSelectedSymbol(symbol);
    setSelectedTimeframe(timeframe);
  };

  const handleBulkDelete = () => {
    if (selectedRows.size === 0) return;
    
    if (confirm(`Are you sure you want to delete ${selectedRows.size} selected rows?`)) {
      alert('Bulk delete functionality would be implemented here');
      setSelectedRows(new Set());
    }
  };

  const handleRefresh = () => {
    setSelectedRows(new Set());
    refetch();
  };

  const filteredData = getFilteredData();
  const totalPages = Math.max(1, Math.ceil((tradingData?.total_count || tradingData?.count || tradingData?.data?.length || 0) / rowLimit));

  return (
    <>
      {/* Error Display */}
      {error && (
        <div className="mb-6">
          <ErrorDisplay 
            isDarkMode={isDarkMode} 
            error={error} 
            onRetry={() => {
              setError(null);
              refetch();
            }}
          />
        </div>
      )}

      {/* Database Stats */}
      <DataStats stats={stats} isDarkMode={isDarkMode} />

      {/* Enhanced Controls */}
      <DataControls
        selectedSymbol={selectedSymbol}
        setSelectedSymbol={setSelectedSymbol}
        selectedTimeframe={selectedTimeframe}
        setSelectedTimeframe={setSelectedTimeframe}
        rowLimit={rowLimit}
        setRowLimit={setRowLimit}
        sortOrder={sortOrder}
        setSortOrder={setSortOrder}
        showColumnFilters={showColumnFilters}
        setShowColumnFilters={setShowColumnFilters}
        showDebugInfo={showDebugInfo}
        setShowDebugInfo={setShowDebugInfo}
        onRefresh={handleRefresh}
        isDarkMode={isDarkMode}
      />

      {/* Advanced Filters */}
      {showColumnFilters && (
        <div className={`rounded-lg shadow-md p-6 mb-8 transition-colors duration-200 ${
          isDarkMode ? 'bg-gray-800' : 'bg-white'
        }`}>
          <AdvancedFilters
            searchTerm={searchTerm}
            setSearchTerm={setSearchTerm}
            selectedSearchColumn={selectedSearchColumn}
            setSelectedSearchColumn={setSelectedSearchColumn}
            sortColumn={sortColumn}
            setSortColumn={setSortColumn}
            sortOrder={sortOrder}
            setSortOrder={setSortOrder}
            currentPage={currentPage}
            setCurrentPage={setCurrentPage}
            isDarkMode={isDarkMode}
          />
        </div>
      )}

      {/* Debug Information */}
      {showDebugInfo && (
        <div className={`rounded-lg shadow-md p-6 mb-8 transition-colors duration-200 ${
          isDarkMode ? 'bg-gray-800' : 'bg-white'
        }`}>
          <DebugPanel lastFetchInfo={lastFetchInfo} isDarkMode={isDarkMode} />
        </div>
      )}

      {/* Trading Data Table */}
      <DataTable
        tradingData={tradingData}
        loading={loading}
        selectedSymbol={selectedSymbol}
        selectedTimeframe={selectedTimeframe}
        sortOrder={sortOrder}
        sortColumn={sortColumn}
        onColumnSort={handleColumnSort}
        selectedRows={selectedRows}
        onRowSelect={handleRowSelect}
        onSelectAll={handleSelectAll}
        searchTerm={searchTerm}
        filteredData={filteredData}
        currentPage={currentPage}
        rowLimit={rowLimit}
        totalPages={totalPages}
        onPageChange={setCurrentPage}
        onExportCSV={exportToCSV}
        onBulkDelete={handleBulkDelete}
        onRefresh={handleRefresh}
        lastFetchInfo={lastFetchInfo}
        isDarkMode={isDarkMode}
      />

      {/* Tables Overview */}
      <TablesList 
        tables={tables} 
        onTableSelect={quickSelectTable} 
        isDarkMode={isDarkMode} 
      />
    </>
  );
};

DataDashboard.propTypes = {
  isDarkMode: PropTypes.bool.isRequired
};

export default DataDashboard; 