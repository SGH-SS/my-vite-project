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