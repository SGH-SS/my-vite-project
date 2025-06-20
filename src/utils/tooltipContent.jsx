/**
 * Centralized InfoTooltip content definitions
 * All tooltip content is maintained here for consistency and easy updates
 */

export const VECTOR_STATS_INFO = {
  count: {
    title: "Vector Count",
    content: (
      <div>
        <p className="font-semibold mb-2">📊 Vector Count</p>
        <p className="mb-2">This shows the total number of trading vectors (time periods) loaded from your database.</p>
        <p className="mb-2"><strong>What it means:</strong></p>
        <ul className="list-disc list-inside space-y-1 text-xs">
          <li>Each vector represents one candle/time period (e.g., 1 minute, 1 hour)</li>
          <li>More vectors = more historical data for pattern analysis</li>
          <li>Typical ranges: 25-500 vectors depending on your settings</li>
          <li>Higher counts improve ML model training but increase processing time</li>
        </ul>
      </div>
    )
  },
  dimensions: {
    title: "Dimensions",
    content: (
      <div>
        <p className="font-semibold mb-2">🔢 Vector Dimensions</p>
        <p className="mb-2">The number of features/values in each trading vector.</p>
        <p className="mb-2"><strong>Common dimension sizes:</strong></p>
        <ul className="list-disc list-inside space-y-1 text-xs">
          <li><strong>Raw OHLC:</strong> 4 dimensions (Open, High, Low, Close)</li>
          <li><strong>Raw OHLCV:</strong> 5 dimensions (+ Volume)</li>
          <li><strong>Normalized:</strong> Same as raw but z-score scaled</li>
          <li><strong>BERT vectors:</strong> 384 dimensions (semantic embeddings)</li>
        </ul>
        <p className="mt-2 text-xs">Higher dimensions capture more complexity but require more computational resources.</p>
      </div>
    )
  },
  range: {
    title: "Value Range",
    content: (
      <div>
        <p className="font-semibold mb-2">📈 Value Range (Min to Max)</p>
        <p className="mb-2">The lowest and highest values across all vector dimensions.</p>
        <p className="mb-2"><strong>Range interpretation:</strong></p>
        <ul className="list-disc list-inside space-y-1 text-xs">
          <li><strong>Raw data:</strong> Actual price values (e.g., $4,200 - $4,800)</li>
          <li><strong>Normalized data:</strong> Usually -3 to +3 (z-scores)</li>
          <li><strong>BERT embeddings:</strong> Typically -1 to +1 range</li>
        </ul>
        <p className="mt-2 text-xs">Large ranges in raw data suggest high price volatility. Normalized data should have consistent ranges for better ML performance.</p>
      </div>
    )
  },
  std: {
    title: "Standard Deviation",
    content: (
      <div>
        <p className="font-semibold mb-2">📊 Standard Deviation</p>
        <p className="mb-2">Measures how spread out the vector values are from their average.</p>
        <p className="mb-2"><strong>What different values mean:</strong></p>
        <ul className="list-disc list-inside space-y-1 text-xs">
          <li><strong>Low Std Dev (&lt; 1.0):</strong> Values are clustered, low volatility</li>
          <li><strong>Medium Std Dev (1.0-5.0):</strong> Moderate spread, normal market</li>
          <li><strong>High Std Dev (&gt; 5.0):</strong> High volatility, trending market</li>
        </ul>
        <p className="mt-2 text-xs">For normalized data, std dev close to 1.0 indicates proper scaling. For raw price data, higher values suggest more volatile trading periods.</p>
      </div>
    )
  }
};

// Vector Comparison Information
export const VECTOR_COMPARISON_INFO = {
  content: (
    <div>
      <p className="font-semibold mb-2">⚖️ Vector Comparison</p>
      <p className="mb-2">Compare two trading periods side-by-side:</p>
      <ul className="list-disc list-inside space-y-1 text-xs">
        <li><strong>Select Candles:</strong> Choose any two time periods to compare</li>
        <li><strong>OHLC Data:</strong> See actual price values for each period</li>
        <li><strong>Vector Values:</strong> Compare mathematical representations</li>
        <li><strong>Differences:</strong> Red highlighting shows significant variations</li>
      </ul>
      <p className="mt-2 text-xs">Useful for finding similar market conditions and understanding pattern variations.</p>
    </div>
  )
};

export const SIMILARITY_SCORE_INFO = {
  content: (
    <div>
      <p className="font-semibold mb-2">📈 Similarity Score</p>
      <p className="mb-2">Cosine similarity between the two selected vectors:</p>
      <ul className="list-disc list-inside space-y-1 text-xs">
        <li><strong>80-100%:</strong> <span className="text-green-600">Very Similar</span> - Strong pattern match</li>
        <li><strong>60-79%:</strong> <span className="text-yellow-600">Moderately Similar</span> - Some correlation</li>
        <li><strong>0-59%:</strong> <span className="text-red-600">Different</span> - Weak or no correlation</li>
      </ul>
      <p className="mt-2 text-xs">Higher scores suggest similar market conditions that might lead to similar future behavior.</p>
    </div>
  )
};

export const SIMILARITY_ANALYSIS_INFO = {
  content: (
    <div>
      <p className="font-semibold mb-2">🔍 Vector Similarity</p>
      <p className="mb-2">Cosine similarity measures how similar two trading periods are:</p>
      <ul className="list-disc list-inside space-y-1 text-xs">
        <li><strong>90-100%:</strong> Very Similar - Almost identical patterns</li>
        <li><strong>70-89%:</strong> Similar - Strong pattern match</li>
        <li><strong>50-69%:</strong> Somewhat Similar - Weak correlation</li>
        <li><strong>0-49%:</strong> Different - No significant pattern match</li>
      </ul>
      <p className="mt-2 text-xs">High similarity suggests similar market conditions and potential future behavior.</p>
    </div>
  )
};

// Heatmap Information
export const HEATMAP_INFO = {
  content: (
    <div>
      <p className="font-semibold mb-2">🔥 Vector Heatmap Visualization</p>
      <p className="mb-3">Interactive color-coded representation of your trading vector data.</p>
      
      <p className="font-medium mb-2">📖 How to read the heatmap:</p>
      <ul className="list-disc list-inside space-y-1 text-xs mb-3">
        <li><strong>Rows:</strong> Individual candles/time periods (newest to oldest)</li>
        <li><strong>Columns:</strong> Vector dimensions (D0, D1, D2, etc.)</li>
        <li><strong>Colors:</strong> Value intensity from blue (low) to green (mid) to red (high)</li>
        <li><strong>Hover:</strong> See exact values and ranges for each cell</li>
      </ul>

      <p className="font-medium mb-2">🎨 Color coding:</p>
      <ul className="list-disc list-inside space-y-1 text-xs mb-3">
        <li><span className="inline-block w-3 h-3 bg-blue-500 rounded mr-1"></span><strong>Blue:</strong> Values in lower range (bearish/low activity)</li>
        <li><span className="inline-block w-3 h-3 bg-green-500 rounded mr-1"></span><strong>Green:</strong> Values in middle range (neutral/balanced)</li>
        <li><span className="inline-block w-3 h-3 bg-red-500 rounded mr-1"></span><strong>Red:</strong> Values in upper range (bullish/high activity)</li>
      </ul>

      <p className="font-medium mb-2">🔍 What to look for:</p>
      <ul className="list-disc list-inside space-y-1 text-xs">
        <li><strong>Patterns:</strong> Vertical bands indicate correlated dimensions</li>
        <li><strong>Clusters:</strong> Similar colors in rows show market patterns</li>
        <li><strong>Anomalies:</strong> Isolated extreme colors may indicate significant events</li>
        <li><strong>Trends:</strong> Color shifts across rows show market direction changes</li>
      </ul>
    </div>
  )
};

// Data Order Information
export const DATA_ORDER_INFO = {
  content: (
    <div>
      <p className="font-semibold mb-2">📊 Data Order & Quality</p>
      <p className="mb-2">Understanding your data display:</p>
      <ul className="list-disc list-inside space-y-1 text-xs">
        <li><strong>🟢 OLDEST first:</strong> Historical analysis, pattern backtesting</li>
        <li><strong>🔵 NEWEST first:</strong> Current market monitoring, recent activity</li>
        <li><strong>✅ Backend sorting:</strong> Server handles ordering efficiently</li>
        <li><strong>❌ Client fallback:</strong> Browser sorts data (may be slower)</li>
      </ul>
      <p className="mt-2 text-xs">Data quality indicators help you understand if you're getting the expected time range.</p>
    </div>
  )
};

// Search Functionality Information
export const SEARCH_FUNCTIONALITY_INFO = {
  content: (
    <div>
      <p className="font-semibold mb-2">🔍 Search Features</p>
      <p className="mb-2">Powerful search across your trading data:</p>
      <ul className="list-disc list-inside space-y-1 text-xs">
        <li><strong>All Columns:</strong> Searches timestamps, prices, and volume</li>
        <li><strong>Specific Column:</strong> Target your search to one data type</li>
        <li><strong>Smart Formatting:</strong> Automatically handles price formatting</li>
        <li><strong>Real-time:</strong> Results update as you type</li>
      </ul>
      <p className="mt-2 text-xs"><strong>Tip:</strong> Remove commas from numbers for better search results.</p>
    </div>
  )
};

export const VECTOR_TYPES_INFO = {
  content: (
    <div>
      <p className="font-semibold mb-2">🧠 Vector Types Explained</p>
      <p className="mb-2">Different ways to represent trading data as mathematical vectors:</p>
      <ul className="list-disc list-inside space-y-1 text-xs">
        <li><strong>Raw OHLC:</strong> Direct price values (4 dimensions)</li>
        <li><strong>Raw OHLCV:</strong> Price + volume (5 dimensions)</li>
        <li><strong>Normalized:</strong> Z-score scaled for pattern matching</li>
        <li><strong>BERT:</strong> AI semantic embeddings (384 dimensions)</li>
      </ul>
      <p className="mt-2 text-xs">Choose based on your analysis needs - raw for price analysis, normalized for pattern matching, BERT for semantic similarity.</p>
    </div>
  )
}; 