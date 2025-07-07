# Trading Dashboard - Modular Architecture

## Overview

This Trading Dashboard has been refactored from a monolithic 2,845-line component into a modular, maintainable architecture using React best practices.

## Architecture Structure

```
src/
├── App.jsx                           # Main app entry point
├── components/
│   ├── shared/                       # Reusable UI components
│   │   ├── InfoTooltip.jsx          # Tooltip component with helpful info
│   │   ├── ThemeToggle.jsx          # Dark/light mode toggle
│   │   ├── LoadingSpinner.jsx       # Loading state indicator
│   │   └── ErrorDisplay.jsx         # Error message display
│   ├── data-dashboard/               # Data dashboard components
│   │   ├── DataDashboard.jsx        # Main container
│   │   ├── DataStats.jsx            # Database statistics cards
│   │   ├── DataControls.jsx         # Symbol/timeframe controls
│   │   ├── DataTable.jsx            # Trading data table
│   │   ├── AdvancedFilters.jsx      # Search and filter controls
│   │   ├── Pagination.jsx           # Table pagination
│   │   ├── TablesList.jsx           # Available tables overview
│   │   └── DebugPanel.jsx           # Debug information panel
│   └── vector-dashboard/             # Vector dashboard components
│       ├── VectorDashboard.jsx       # Main container
│       ├── VectorControls.jsx        # Vector data controls
│       ├── VectorStats.jsx           # Vector statistics display
│       ├── VectorTypeSelector.jsx    # Vector type selection grid
│       ├── VectorVisualization.jsx   # Visualization container
│       ├── VectorHeatmap.jsx         # Heatmap visualization
│       └── VectorComparison.jsx      # Side-by-side comparison
├── context/
│   └── TradingContext.jsx            # Global state management
├── hooks/
│   ├── useTheme.js                   # Theme management hook
│   ├── useTradingData.js             # Data fetching for trading data
│   └── useVectorData.js              # Data fetching for vectors
└── utils/
    ├── constants.js                  # App-wide constants
    ├── formatters.js                 # Data formatting functions
    ├── vectorCalculations.js         # Vector math utilities
    └── tooltipContent.js             # Centralized tooltip content

```

## Key Features Preserved

✅ **All Functionality Maintained**
- Data Dashboard: Table viewing, sorting, searching, pagination
- Vector Dashboard: Vector type selection, heatmap, comparison
- Theme switching (dark/light mode)
- All InfoTooltip explanations intact
- Database statistics and table overview
- Advanced filtering and debug info

✅ **Improved Architecture**
- Components under 200 lines each
- Proper separation of concerns
- Reusable components with PropTypes
- Context API for shared state
- Custom hooks for data management
- Centralized constants and utilities

## Component Guidelines

### Shared Components
All shared components accept `isDarkMode` prop and are fully reusable:
```jsx
<InfoTooltip id="unique-id" content={tooltipContent} isDarkMode={isDarkMode} />
<LoadingSpinner isDarkMode={isDarkMode} message="Loading..." />
<ErrorDisplay isDarkMode={isDarkMode} error={errorMessage} onRetry={handleRetry} />
<ThemeToggle isDarkMode={isDarkMode} toggleTheme={toggleTheme} />
```

### Data Flow
1. **Global State**: Managed by `TradingContext` (symbol, timeframe, row limit)
2. **Local State**: Component-specific state (search terms, filters, etc.)
3. **Data Fetching**: Custom hooks (`useTradingData`, `useVectorData`)
4. **Styling**: All components respect `isDarkMode` prop

### Adding New Features
1. Create component in appropriate folder
2. Add PropTypes validation
3. Use existing utilities and constants
4. Maintain dark mode support
5. Add to this README

## Running the Application

```bash
npm install
npm run dev
```

## API Requirements
The app expects a backend API at `http://localhost:8000/api/trading` with endpoints:
- `/data/{symbol}/{timeframe}` - Trading data
- `/stats` - Database statistics
- `/tables` - Available tables list

## Future Enhancements
- TypeScript migration for better type safety
- Unit tests for all components
- Storybook for component documentation
- Performance optimizations with React.memo
- Additional vector visualization modes 

## Labeled Dataset & Chart Label Visualization

This project now includes a workflow for creating and visualizing labeled trading data directly on the chart:

- **Labeled Dataset Creation:**
  - The backend and preprocessing pipeline generate a labeled dataset, identifying significant trading points (e.g., TJR Highs and Lows) for each symbol and timeframe.
  - For each label, a time range (pointer) is provided, and the actual high/low candle within that range is determined for precise marker placement.
  - **Current status:** The SPY 1H dataset is fully labeled for both `tjr_high` and `tjr_low` events.

- **Chart Visualization:**
  - When viewing a chart (e.g., SPY 1H), the frontend fetches these labels and overlays them as markers:
    - Green "T" markers above candles for TJR Highs
    - Red "⊥" markers below candles for TJR Lows
  - Marker toggles allow users to show/hide each label type interactively.
  - The marker system is robust: toggling always displays only the correct markers, with no ghosting or layering, thanks to a full series/marker rebuild on each update.

- **How it works in code:**
  - The chart component receives the labeled data, groups it by event, and finds the actual high/low within each labeled region.
  - Markers are built fresh on every toggle or data change, ensuring the chart always reflects the current labeled dataset and user preferences. 