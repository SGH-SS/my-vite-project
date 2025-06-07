import { useState, useEffect } from 'react';

const TradingDashboard = () => {
  const [stats, setStats] = useState(null);
  const [tables, setTables] = useState([]);
  const [selectedSymbol, setSelectedSymbol] = useState('es');
  const [selectedTimeframe, setSelectedTimeframe] = useState('1d');
  const [tradingData, setTradingData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  
  // New state for enhanced functionality
  const [rowLimit, setRowLimit] = useState(50);
  const [sortOrder, setSortOrder] = useState('desc');
  const [sortColumn, setSortColumn] = useState('timestamp');
  const [currentPage, setCurrentPage] = useState(1);
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedRows, setSelectedRows] = useState(new Set());
  const [showColumnFilters, setShowColumnFilters] = useState(false);
  const [showDebugInfo, setShowDebugInfo] = useState(false);
  const [lastFetchInfo, setLastFetchInfo] = useState(null);

  const API_BASE_URL = 'http://localhost:8000/api/trading';

  useEffect(() => {
    fetchStats();
    fetchTables();
  }, []);

  useEffect(() => {
    if (selectedSymbol && selectedTimeframe) {
      fetchTradingData();
    }
  }, [selectedSymbol, selectedTimeframe, rowLimit, sortOrder, sortColumn, currentPage]);

  useEffect(() => {
    if (selectedSymbol && selectedTimeframe) {
      setCurrentPage(1); // Reset to first page when selection changes
    }
  }, [selectedSymbol, selectedTimeframe, rowLimit, sortOrder, sortColumn]);

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

  const fetchTradingData = async () => {
    setLoading(true);
    console.log(`🚀 fetchTradingData called - Page: ${currentPage}, Symbol: ${selectedSymbol}, Timeframe: ${selectedTimeframe}, RowLimit: ${rowLimit}, SortOrder: ${sortOrder}`);
    try {
      // Strategy: For oldest data, we need to get the very first records from the database
      // For newest data, we get the most recent records
      
      let url = `${API_BASE_URL}/data/${selectedSymbol}/${selectedTimeframe}?limit=${rowLimit}`;
      let offset = 0;
      
      if (sortOrder === 'asc') {
        // For ASCENDING (oldest first): Get from the very beginning of the dataset
        offset = (currentPage - 1) * rowLimit;
        // Try multiple strategies to get oldest data
        url += `&offset=${offset}&order=asc&sort_by=timestamp`;
        
        // Alternative: Try to explicitly request oldest data
        const oldestUrl = `${API_BASE_URL}/data/${selectedSymbol}/${selectedTimeframe}?limit=${rowLimit}&offset=${offset}&order=asc&sort_by=timestamp&from_start=true`;
        
        console.log(`🟢 ASCENDING: Requesting OLDEST ${rowLimit} records starting from record ${offset + 1}`);
        console.log('Trying oldest-first URL:', oldestUrl);
        
        // Try the explicit oldest-first URL first
        try {
          const response = await fetch(oldestUrl);
          if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
          }
          const text = await response.text();
          if (!text) {
            throw new Error('Empty response from server');
          }
          var data = JSON.parse(text);
          var fetchUrl = oldestUrl;
        } catch (e) {
          // Fallback to standard URL
          const response = await fetch(url);
          if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
          }
          const text = await response.text();
          if (!text) {
            throw new Error('Empty response from server');
          }
          var data = JSON.parse(text);
          var fetchUrl = url;
        }
        
      } else {
        // For DESCENDING (newest first): Get the most recent data
        offset = (currentPage - 1) * rowLimit;
        url += `&offset=${offset}&order=desc&sort_by=timestamp`;
        
        console.log(`🔵 DESCENDING: Requesting NEWEST ${rowLimit} records starting from record ${offset + 1}`);
        console.log('API URL:', url);
        
        const response = await fetch(url);
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        const text = await response.text();
        if (!text) {
          throw new Error('Empty response from server');
        }
        var data = JSON.parse(text);
        var fetchUrl = url;
      }
      
      // If we're on ascending and still getting recent data, try a different approach
      if (sortOrder === 'asc' && data.data && data.data.length > 0) {
        const firstTimestamp = new Date(data.data[0].timestamp);
        const currentYear = new Date().getFullYear();
        const dataYear = firstTimestamp.getFullYear();
        
        // If the data is from recent times (same year), try to get older data
        if (dataYear >= currentYear - 1) {
          console.log('⚠️ Still getting recent data for ascending, trying alternative approach...');
          
          // Try to get data with a much larger offset to reach older records
          const largeOffset = Math.max(10000, (currentPage - 1) * rowLimit);
          const alternativeUrl = `${API_BASE_URL}/data/${selectedSymbol}/${selectedTimeframe}?limit=${rowLimit}&offset=${largeOffset}&order=asc&sort_by=timestamp`;
          
          try {
            console.log('Trying large offset approach:', alternativeUrl);
            const altResponse = await fetch(alternativeUrl);
            if (!altResponse.ok) {
              throw new Error(`HTTP error! status: ${altResponse.status}`);
            }
            const altText = await altResponse.text();
            if (!altText) {
              throw new Error('Empty response from server');
            }
            const altData = JSON.parse(altText);
            
            if (altData.data && altData.data.length > 0) {
              const altFirstTimestamp = new Date(altData.data[0].timestamp);
              // If we got older data, use it
              if (altFirstTimestamp < firstTimestamp) {
                console.log('✅ Found older data with large offset approach');
                data = altData;
                fetchUrl = alternativeUrl;
              }
            }
          } catch (e) {
            console.log('Large offset approach failed, using original data');
          }
        }
      }
      
      // Estimate total count
      if (!data.total_count) {
        // Try to get a better estimate of total records
        try {
          const countResponse = await fetch(`${API_BASE_URL}/data/${selectedSymbol}/${selectedTimeframe}?limit=1&count_only=true`);
          if (countResponse.ok) {
            const countText = await countResponse.text();
            if (countText) {
              const countData = JSON.parse(countText);
              data.total_count = countData.total_count || countData.count || data.count || 50000; // Better fallback
            }
          }
        } catch (e) {
          console.warn('Failed to get count estimate:', e);
          data.total_count = data.count || 50000; // Fallback estimate
        }
      }
      
      // Check if backend properly handled our sort order
      let backendHandledSorting = false;
      
      if (data.data && Array.isArray(data.data) && data.data.length > 1) {
        const firstTimestamp = new Date(data.data[0].timestamp);
        const lastTimestamp = new Date(data.data[data.data.length - 1].timestamp);
        
        if (sortOrder === 'asc') {
          // For ascending, first should be older than last
          backendHandledSorting = firstTimestamp <= lastTimestamp;
        } else {
          // For descending, first should be newer than last
          backendHandledSorting = firstTimestamp >= lastTimestamp;
        }
      }

      console.log(`Backend handled sorting properly: ${backendHandledSorting}`);

      if (data.data && Array.isArray(data.data)) {
        let sortedData = [...data.data];
        
        // Always apply client-side sorting to ensure proper ordering
        console.log(`🔀 Applying client-side sorting by ${sortColumn} (${sortOrder})...`);
        console.log(`📊 Sample data before sort:`, sortedData.slice(0, 3).map(row => ({ [sortColumn]: row[sortColumn] })));
        
        sortedData.sort((a, b) => {
          let aVal = a[sortColumn];
          let bVal = b[sortColumn];
          
          // Handle different data types
          if (sortColumn === 'timestamp') {
            aVal = new Date(aVal);
            bVal = new Date(bVal);
          } else if (typeof aVal === 'string') {
            aVal = aVal.toLowerCase();
            bVal = bVal.toLowerCase();
          } else if (typeof aVal === 'number') {
            // Numbers - handle null/undefined
            aVal = aVal || 0;
            bVal = bVal || 0;
          }
          
          if (aVal < bVal) return sortOrder === 'asc' ? -1 : 1;
          if (aVal > bVal) return sortOrder === 'asc' ? 1 : -1;
          return 0;
        });
        
        data.data = sortedData;
        console.log(`✅ Sorting complete. Sample data after sort:`, sortedData.slice(0, 3).map(row => ({ [sortColumn]: row[sortColumn] })));
        
        // Enhanced debug information
        setLastFetchInfo({
          timestamp: new Date().toLocaleTimeString(),
          sortOrder: sortOrder,
          requestedData: sortOrder === 'asc' ? 'OLDEST' : 'NEWEST',
          rowLimit: rowLimit,
          currentPage: currentPage,
          offset: offset,
          apiUrl: fetchUrl,
          backendHandledSorting: backendHandledSorting,
          actualDataRange: data.data.length > 0 ? {
            first: data.data[0].timestamp,
            last: data.data[data.data.length - 1].timestamp,
            firstYear: new Date(data.data[0].timestamp).getFullYear(),
            lastYear: new Date(data.data[data.data.length - 1].timestamp).getFullYear(),
            span: `${new Date(data.data[0].timestamp).toLocaleDateString()} to ${new Date(data.data[data.data.length - 1].timestamp).toLocaleDateString()}`
          } : null,
          totalRecords: data.total_count || data.count,
          dataQuality: (() => {
            if (sortOrder === 'desc') return 'GOOD (Recent data)';
            
            // For ascending, check if we're actually getting properly sorted data
            if (data.data.length > 1) {
              const firstTime = new Date(data.data[0].timestamp);
              const lastTime = new Date(data.data[data.data.length - 1].timestamp);
              
              // If properly sorted ascending and we're on page 1, this should be good
              if (firstTime <= lastTime && currentPage === 1) {
                return 'GOOD (Historical data - oldest first)';
              } else if (firstTime <= lastTime) {
                return 'GOOD (Historical data - properly sorted)';
              } else {
                return 'POOR (Sorting may be incorrect)';
              }
            }
            
            // Single record or no data
            return currentPage === 1 ? 'GOOD (Historical data)' : 'OK (Paginated data)';
          })()
        });
      }
      
      console.log(`📊 Setting trading data - Total Count: ${data.total_count}, Data Length: ${data.data?.length}, Current Page: ${currentPage}`);
      setTradingData(data);
      setError(null);
    } catch (err) {
      setError(`Failed to fetch trading data for ${selectedSymbol}_${selectedTimeframe}: ${err.message}`);
      console.error('Fetch error:', err);
    } finally {
      setLoading(false);
    }
  };

  const formatPrice = (price) => {
    return price ? price.toFixed(4) : 'N/A';
  };

  const formatTimestamp = (timestamp) => {
    return new Date(timestamp).toLocaleString();
  };

  const getSymbolColor = (symbol) => {
    switch (symbol) {
      case 'es': return 'text-blue-600';
      case 'eurusd': return 'text-green-600';
      case 'spy': return 'text-purple-600';
      default: return 'text-gray-600';
    }
  };

  const handleColumnSort = (column) => {
    console.log(`🔄 Column sort clicked: ${column}, current sort: ${sortColumn} ${sortOrder}`);
    
    if (sortColumn === column) {
      // Toggle sort order if clicking the same column
      setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc');
    } else {
      // Set new column and default to descending
      setSortColumn(column);
      setSortOrder('desc');
    }
    setCurrentPage(1);
  };

  const handleRowSelect = (index) => {
    const newSelected = new Set(selectedRows);
    if (newSelected.has(index)) {
      newSelected.delete(index);
    } else {
      newSelected.add(index);
    }
    setSelectedRows(newSelected);
  };

  const handleSelectAll = () => {
    if (selectedRows.size === tradingData?.data.length) {
      setSelectedRows(new Set());
    } else {
      const allRows = new Set(tradingData?.data.map((_, index) => index) || []);
      setSelectedRows(allRows);
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
      // TODO: Implement bulk delete API call
      alert('Bulk delete functionality would be implemented here');
      setSelectedRows(new Set());
    }
  };

  const handleRefresh = () => {
    setSelectedRows(new Set());
    fetchTradingData();
  };

  const totalPages = Math.max(1, Math.ceil((tradingData?.total_count || tradingData?.count || tradingData?.data?.length || 0) / rowLimit));
  
  // Debug pagination calculations
  if (tradingData) {
    console.log(`📄 Pagination Debug - Total Count: ${tradingData.total_count}, Count: ${tradingData.count}, Data Length: ${tradingData.data?.length}, Row Limit: ${rowLimit}, Total Pages: ${totalPages}, Current Page: ${currentPage}`);
  }

  return (
    <div className="min-h-screen bg-gray-50 p-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-4xl font-bold text-gray-900 mb-2">
            Trading Data Dashboard
          </h1>
          <p className="text-gray-600">
            Real-time access to your PostgreSQL trading database • pgAdmin but better
          </p>
        </div>

        {/* Error Display */}
        {error && (
          <div className="mb-6 bg-red-50 border border-red-200 rounded-lg p-4">
            <div className="flex items-center">
              <div className="text-red-600 mr-2">⚠️</div>
              <div className="text-red-800">{error}</div>
            </div>
          </div>
        )}

        {/* Database Stats */}
        {stats && (
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
            <div className="bg-white rounded-lg shadow-md p-6">
              <div className="flex items-center">
                <div className="text-3xl mr-4">📊</div>
                <div>
                  <h3 className="text-lg font-semibold text-gray-900">Total Tables</h3>
                  <p className="text-2xl font-bold text-blue-600">{stats.total_tables}</p>
                </div>
              </div>
            </div>
            
            <div className="bg-white rounded-lg shadow-md p-6">
              <div className="flex items-center">
                <div className="text-3xl mr-4">💾</div>
                <div>
                  <h3 className="text-lg font-semibold text-gray-900">Total Records</h3>
                  <p className="text-2xl font-bold text-green-600">
                    {stats.total_rows.toLocaleString()}
                  </p>
                </div>
              </div>
            </div>
            
            <div className="bg-white rounded-lg shadow-md p-6">
              <div className="flex items-center">
                <div className="text-3xl mr-4">🔄</div>
                <div>
                  <h3 className="text-lg font-semibold text-gray-900">Status</h3>
                  <p className="text-2xl font-bold text-green-600">Connected</p>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Enhanced Controls */}
        <div className="bg-white rounded-lg shadow-md p-6 mb-8">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold text-gray-900">Data Selection & Controls</h3>
            <div className="flex gap-2">
              <button
                onClick={() => setShowColumnFilters(!showColumnFilters)}
                className="px-3 py-2 text-sm bg-gray-100 hover:bg-gray-200 rounded-md transition-colors flex items-center gap-1"
              >
                🔍 Filters {showColumnFilters ? '▼' : '▶'}
              </button>
              <button
                onClick={handleRefresh}
                className="px-3 py-2 text-sm bg-blue-100 hover:bg-blue-200 text-blue-700 rounded-md transition-colors flex items-center gap-1"
              >
                🔄 Refresh
              </button>
              <button
                onClick={() => setShowDebugInfo(!showDebugInfo)}
                className="px-3 py-2 text-sm bg-yellow-100 hover:bg-yellow-200 text-yellow-700 rounded-md transition-colors flex items-center gap-1"
              >
                🐛 Debug {showDebugInfo ? '▼' : '▶'}
              </button>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Symbol
              </label>
              <select
                value={selectedSymbol}
                onChange={(e) => setSelectedSymbol(e.target.value)}
                className="w-full border border-gray-300 rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <option value="es">ES (E-mini S&P 500)</option>
                <option value="eurusd">EURUSD (Euro/US Dollar)</option>
                <option value="spy">SPY (SPDR S&P 500 ETF)</option>
              </select>
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Timeframe
              </label>
              <select
                value={selectedTimeframe}
                onChange={(e) => setSelectedTimeframe(e.target.value)}
                className="w-full border border-gray-300 rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <option value="1m">1 Minute</option>
                <option value="5m">5 Minutes</option>
                <option value="15m">15 Minutes</option>
                <option value="30m">30 Minutes</option>
                <option value="1h">1 Hour</option>
                <option value="4h">4 Hours</option>
                <option value="1d">1 Day</option>
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Rows per page
              </label>
              <select
                value={rowLimit}
                onChange={(e) => setRowLimit(Number(e.target.value))}
                className="w-full border border-gray-300 rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <option value={25}>25 rows</option>
                <option value={50}>50 rows</option>
                <option value={100}>100 rows</option>
                <option value={250}>250 rows</option>
                <option value={500}>500 rows</option>
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Sort Order
              </label>
              <select
                value={sortOrder}
                onChange={(e) => setSortOrder(e.target.value)}
                className="w-full border border-gray-300 rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <option value="desc">⬇ Descending (Newest first)</option>
                <option value="asc">⬆ Ascending (Oldest first)</option>
              </select>
            </div>
          </div>

          {/* Search and Advanced Controls */}
          {showColumnFilters && (
            <div className="mt-4 pt-4 border-t border-gray-200 bg-gray-50 rounded-md p-4">
              <h4 className="text-md font-medium text-gray-800 mb-3">Advanced Filters & Controls</h4>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Search Data
                  </label>
                  <input
                    type="text"
                    value={searchTerm}
                    onChange={(e) => setSearchTerm(e.target.value)}
                    placeholder="Search in all columns..."
                    className="w-full border border-gray-300 rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Sort Column
                  </label>
                  <select
                    value={sortColumn}
                    onChange={(e) => setSortColumn(e.target.value)}
                    className="w-full border border-gray-300 rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
                  >
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
                    onClick={() => {
                      setSearchTerm('');
                      setSortColumn('timestamp');
                      setSortOrder('desc');
                      setCurrentPage(1);
                    }}
                    className="w-full px-3 py-2 text-sm bg-gray-200 hover:bg-gray-300 rounded-md transition-colors"
                  >
                    🔄 Reset Filters
                  </button>
                </div>
              </div>
            </div>
          )}

          {/* Debug Information Panel */}
          {showDebugInfo && lastFetchInfo && (
            <div className="mt-4 pt-4 border-t border-gray-200 bg-yellow-50 rounded-md p-4">
              <h4 className="text-md font-medium text-yellow-800 mb-3">🐛 Debug Information</h4>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
                <div>
                  <p><strong>Last Fetch:</strong> {lastFetchInfo.timestamp}</p>
                  <p><strong>Requested Data:</strong> <span className={lastFetchInfo.requestedData === 'OLDEST' ? 'text-green-600' : 'text-blue-600'}>{lastFetchInfo.requestedData}</span> {lastFetchInfo.rowLimit} records</p>
                  <p><strong>Page:</strong> {lastFetchInfo.currentPage} (offset: {lastFetchInfo.offset})</p>
                  <p><strong>Sort:</strong> {lastFetchInfo.sortOrder} by {lastFetchInfo.sortColumn}</p>
                  <p><strong>Data Quality:</strong> <span className={lastFetchInfo.dataQuality?.includes('GOOD') ? 'text-green-600' : 'text-red-600'}>{lastFetchInfo.dataQuality}</span></p>
                </div>
                <div>
                  <p><strong>Backend Sorting:</strong> <span className={lastFetchInfo.backendHandledSorting ? 'text-green-600' : 'text-red-600'}>{lastFetchInfo.backendHandledSorting ? '✅ Working' : '❌ Not Working'}</span></p>
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
          )}
        </div>

        {/* Trading Data Table */}
        <div className="bg-white rounded-lg shadow-md overflow-hidden">
          <div className="px-6 py-4 border-b border-gray-200 flex items-center justify-between">
            <div>
              <h3 className="text-lg font-semibold text-gray-900">
                {selectedSymbol.toUpperCase()} - {selectedTimeframe} Data
                {tradingData && (
                  <span className="ml-2 text-sm text-gray-500">
                    ({tradingData.count || tradingData.data?.length || 0} records)
                  </span>
                )}
                <br />
                <span className={`text-sm font-medium ${sortOrder === 'asc' ? 'text-green-600' : 'text-blue-600'}`}>
                  {sortOrder === 'asc' ? '🟢 Showing OLDEST data first' : '🔵 Showing NEWEST data first'} 
                  {lastFetchInfo && (
                    <span className="text-gray-500 ml-2">
                      (Backend sorting: {lastFetchInfo.backendHandledSorting ? '✅' : '❌ using client-side fallback'})
                    </span>
                  )}
                </span>
              </h3>
              {selectedRows.size > 0 && (
                <p className="text-sm text-blue-600 mt-1">
                  {selectedRows.size} row(s) selected
                </p>
              )}
            </div>
            <div className="flex gap-2">
              <button
                onClick={exportToCSV}
                disabled={!tradingData?.data?.length}
                className="px-3 py-2 text-sm bg-green-100 hover:bg-green-200 text-green-700 rounded-md transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-1"
              >
                📊 Export CSV
              </button>
              {selectedRows.size > 0 && (
                <>
                  <button
                    onClick={handleBulkDelete}
                    className="px-3 py-2 text-sm bg-red-100 hover:bg-red-200 text-red-700 rounded-md transition-colors flex items-center gap-1"
                  >
                    🗑️ Delete Selected
                  </button>
                  <button
                    onClick={() => setSelectedRows(new Set())}
                    className="px-3 py-2 text-sm bg-gray-100 hover:bg-gray-200 rounded-md transition-colors"
                  >
                    ✖️ Clear Selection
                  </button>
                </>
              )}
            </div>
          </div>
          
          {loading ? (
            <div className="p-8 text-center">
              <div className="inline-block animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
              <p className="mt-2 text-gray-600">Loading trading data...</p>
            </div>
          ) : tradingData && tradingData.data && tradingData.data.length > 0 ? (
            <>
              {/* Enhanced Pagination - TOP */}
              {totalPages > 1 && (
                <div className="bg-white px-6 py-4 border-b border-gray-200 flex items-center justify-between">
                  <div className="flex items-center text-sm text-gray-700">
                    <span>
                      Showing {((currentPage - 1) * rowLimit) + 1} to {Math.min(currentPage * rowLimit, tradingData.total_count || tradingData.count || tradingData.data.length)} 
                      {' '}of {(tradingData.total_count || tradingData.count || tradingData.data.length)?.toLocaleString()} results
                    </span>
                  </div>
                  <div className="flex items-center space-x-2">
                    <button
                      onClick={() => {
                        console.log(`🔸 First button clicked - setting page to 1 (was ${currentPage}), totalPages: ${totalPages}`);
                        setCurrentPage(1);
                      }}
                      disabled={currentPage === 1}
                      className="px-3 py-2 text-sm border border-gray-300 rounded-md disabled:opacity-50 disabled:cursor-not-allowed hover:bg-gray-50 transition-colors"
                    >
                      ⏮ First
                    </button>
                    <button
                      onClick={() => {
                        const newPage = Math.max(1, currentPage - 1);
                        console.log(`🔸 Previous button clicked - setting page to ${newPage} (was ${currentPage}), totalPages: ${totalPages}`);
                        setCurrentPage(newPage);
                      }}
                      disabled={currentPage === 1}
                      className="px-3 py-2 text-sm border border-gray-300 rounded-md disabled:opacity-50 disabled:cursor-not-allowed hover:bg-gray-50 transition-colors"
                    >
                      ⬅ Previous
                    </button>
                    <span className="px-3 py-2 text-sm bg-gray-100 rounded-md">
                      Page {currentPage} of {totalPages} ({(tradingData.total_count || tradingData.count || 0).toLocaleString()} total records)
                    </span>
                    <button
                      onClick={() => {
                        const newPage = Math.min(totalPages, currentPage + 1);
                        console.log(`🔸 Next button clicked - setting page to ${newPage} (was ${currentPage}), totalPages: ${totalPages}`);
                        setCurrentPage(newPage);
                      }}
                      disabled={currentPage >= totalPages}
                      className="px-3 py-2 text-sm border border-gray-300 rounded-md disabled:opacity-50 disabled:cursor-not-allowed hover:bg-gray-50 transition-colors"
                    >
                      Next ➡
                    </button>
                    <button
                      onClick={() => {
                        console.log(`🔸 Last button clicked - setting page to ${totalPages} (was ${currentPage}), totalPages: ${totalPages}`);
                        setCurrentPage(totalPages);
                      }}
                      disabled={currentPage >= totalPages}
                      className="px-3 py-2 text-sm border border-gray-300 rounded-md disabled:opacity-50 disabled:cursor-not-allowed hover:bg-gray-50 transition-colors"
                    >
                      Last ⏭
                    </button>
                  </div>
                </div>
              )}

              <div className="overflow-x-auto">
                <table className="min-w-full divide-y divide-gray-200">
                  <thead className="bg-gray-50">
                    <tr>
                      <th className="px-4 py-3 text-left">
                        <input
                          type="checkbox"
                          checked={selectedRows.size === tradingData.data.length && tradingData.data.length > 0}
                          onChange={handleSelectAll}
                          className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                        />
                      </th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        #
                      </th>
                      {[
                        { key: 'timestamp', label: 'Timestamp', icon: '📅', sortable: true },
                        { key: 'open', label: 'Open', icon: '📈', sortable: true },
                        { key: 'high', label: 'High', icon: '🔺', sortable: true },
                        { key: 'low', label: 'Low', icon: '🔻', sortable: true },
                        { key: 'close', label: 'Close', icon: '💰', sortable: true },
                        { key: 'candle_type', label: 'Candle', icon: '🕯️', sortable: false },
                        { key: 'volume', label: 'Volume', icon: '📊', sortable: true }
                      ].map(({ key, label, icon, sortable }) => (
                        <th
                          key={key}
                          onClick={sortable ? () => handleColumnSort(key) : undefined}
                          className={`px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider select-none ${
                            sortable ? 'cursor-pointer hover:bg-gray-100 transition-colors' : 'cursor-default'
                          }`}
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
                  <tbody className="bg-white divide-y divide-gray-200">
                    {tradingData.data.map((row, index) => {
                      const rowNumber = ((currentPage - 1) * rowLimit) + index + 1;
                      return (
                        <tr 
                          key={index} 
                          className={`hover:bg-gray-50 transition-colors ${
                            selectedRows.has(index) ? 'bg-blue-50 border-l-4 border-blue-500' : ''
                          }`}
                        >
                          <td className="px-4 py-3 whitespace-nowrap">
                            <input
                              type="checkbox"
                              checked={selectedRows.has(index)}
                              onChange={() => handleRowSelect(index)}
                              className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                            />
                          </td>
                          <td className="px-4 py-3 whitespace-nowrap text-sm text-gray-500 font-mono">
                            {rowNumber}
                          </td>
                          <td className="px-4 py-3 whitespace-nowrap text-sm text-gray-900 font-mono">
                            {formatTimestamp(row.timestamp)}
                          </td>
                          <td className="px-4 py-3 whitespace-nowrap text-sm text-gray-900 font-mono">
                            {formatPrice(row.open)}
                          </td>
                          <td className="px-4 py-3 whitespace-nowrap text-sm text-gray-900 font-mono font-semibold">
                            {formatPrice(row.high)}
                          </td>
                          <td className="px-4 py-3 whitespace-nowrap text-sm text-gray-900 font-mono font-semibold">
                            {formatPrice(row.low)}
                          </td>
                          <td className="px-4 py-3 whitespace-nowrap text-sm text-gray-900 font-mono font-bold">
                            {formatPrice(row.close)}
                          </td>
                          <td className="px-4 py-3 whitespace-nowrap text-sm text-center">
                            {(() => {
                              const isGreen = row.close > row.open;
                              const isEqual = row.close === row.open;
                              if (isEqual) {
                                return (
                                  <div className="flex items-center justify-center">
                                    <span className="text-gray-500 text-lg">➖</span>
                                    <span className="ml-1 text-xs text-gray-500 font-medium">DOJI</span>
                                  </div>
                                );
                              }
                              return (
                                <div className="flex items-center justify-center">
                                  <span className={`text-lg ${isGreen ? 'text-green-600' : 'text-red-600'}`}>
                                    {isGreen ? '🟢' : '🔴'}
                                  </span>
                                  <span className={`ml-1 text-xs font-medium ${isGreen ? 'text-green-700' : 'text-red-700'}`}>
                                    {isGreen ? 'G' : 'R'}
                                  </span>
                                </div>
                              );
                            })()}
                          </td>
                          <td className="px-4 py-3 whitespace-nowrap text-sm text-gray-500 font-mono">
                            {row.volume ? row.volume.toLocaleString() : 'N/A'}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </>
          ) : (
            <div className="p-8 text-center text-gray-500">
              <div className="text-4xl mb-4">📊</div>
              <h3 className="text-lg font-medium mb-2">No Data Available</h3>
              <p>No trading data available for {selectedSymbol.toUpperCase()}_{selectedTimeframe}</p>
              <button
                onClick={handleRefresh}
                className="mt-4 px-4 py-2 bg-blue-500 text-white rounded-md hover:bg-blue-600 transition-colors"
              >
                🔄 Try Refresh
              </button>
            </div>
          )}
        </div>

        {/* Tables Overview */}
        {tables.length > 0 && (
          <div className="mt-8 bg-white rounded-lg shadow-md overflow-hidden">
            <div className="px-6 py-4 border-b border-gray-200">
              <h3 className="text-lg font-semibold text-gray-900">Available Tables</h3>
              <p className="text-sm text-gray-600 mt-1">Click any table to quickly load its data • Total: {tables.length} tables</p>
            </div>
            <div className="overflow-x-auto">
              <table className="min-w-full divide-y divide-gray-200">
                <thead className="bg-gray-50">
                  <tr>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Table Name
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Symbol
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Timeframe
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Records
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Latest Data
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Actions
                    </th>
                  </tr>
                </thead>
                <tbody className="bg-white divide-y divide-gray-200">
                  {tables.map((table, index) => (
                    <tr key={index} className="hover:bg-gray-50 transition-colors">
                      <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900 font-mono">
                        {table.table_name}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <span className={`text-sm font-medium ${getSymbolColor(table.symbol)} bg-gray-100 px-2 py-1 rounded`}>
                          {table.symbol.toUpperCase()}
                        </span>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 font-medium">
                        {table.timeframe}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                        <span className="bg-blue-100 text-blue-800 px-2 py-1 rounded font-mono">
                          {table.row_count ? table.row_count.toLocaleString() : 'N/A'}
                        </span>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 font-mono">
                        {table.latest_timestamp ? formatTimestamp(table.latest_timestamp) : 'N/A'}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm">
                        <button
                          onClick={() => quickSelectTable(table)}
                          className="inline-flex items-center px-3 py-1 bg-blue-100 text-blue-700 rounded-md hover:bg-blue-200 transition-colors font-medium"
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
        )}
      </div>
    </div>
  );
};

export default TradingDashboard; 