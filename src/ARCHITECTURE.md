# Trading Dashboard - Modular Architecture

## Overview
This document outlines the refactored modular architecture of the Trading Dashboard application, transforming a 2,845-line monolithic component into a maintainable, scalable structure.

## Directory Structure

```
src/
├── components/
│   ├── shared/
│   │   ├── InfoTooltip.jsx          # Reusable tooltip component
│   │   ├── ThemeToggle.jsx          # Theme switching button
│   │   ├── LoadingSpinner.jsx       # Loading state component
│   │   └── ErrorDisplay.jsx         # Error display component
│   ├── data-dashboard/
│   │   ├── DataDashboard.jsx        # Main data dashboard container
│   │   ├── DataStats.jsx            # Database statistics cards
│   │   ├── DataControls.jsx         # Symbol/timeframe/sorting controls
│   │   ├── DataTable.jsx            # Trading data table
│   │   ├── TablesList.jsx           # Available tables overview
│   │   ├── Pagination.jsx           # Reusable pagination
│   │   └── AdvancedFilters.jsx     # Search and filter controls
│   └── vector-dashboard/
│       ├── VectorDashboard.jsx      # Main vector dashboard container
│       ├── VectorControls.jsx       # Vector selection controls
│       ├── VectorStats.jsx          # Vector statistics display
│       ├── VectorTypeSelector.jsx   # Vector type grid selector
│       ├── VectorVisualization.jsx  # Visualization mode container
│       ├── VectorHeatmap.jsx        # Heatmap visualization
│       └── VectorComparison.jsx     # Side-by-side comparison
├── context/
│   └── TradingContext.jsx           # Shared state context
├── hooks/
│   ├── useTheme.js                  # Theme management hook
│   ├── useTradingData.js            # Data fetching hook
│   ├── useVectorData.js             # Vector data hook
│   └── usePagination.js             # Pagination logic hook
├── utils/
│   ├── constants.js                 # Application constants
│   ├── formatters.js                # Price/timestamp formatting
│   ├── vectorCalculations.js        # Vector math utilities
│   └── tooltipContent.js            # InfoTooltip content definitions
└── App.jsx                          # Main application component

```

## Component Hierarchy

```
App.jsx
├── TradingProvider (Context)
├── ThemeToggle
├── DashboardModeToggle
├── DataDashboard (when mode='data')
│   ├── DataStats
│   ├── DataControls
│   │   └── InfoTooltip
│   ├── AdvancedFilters
│   ├── DataTable
│   │   ├── TableHeader
│   │   ├── TableBody
│   │   └── Pagination
│   └── TablesList
└── VectorDashboard (when mode='vector')
    ├── VectorControls
    │   └── InfoTooltip
    ├── VectorTypeSelector
    │   └── InfoTooltip
    ├── VectorStats
    │   └── InfoTooltip
    └── VectorVisualization
        ├── VectorHeatmap
        │   └── InfoTooltip
        └── VectorComparison
            └── InfoTooltip
```

## Key Design Decisions

### 1. Context API for Shared State
- `TradingContext` manages global state (symbol, timeframe, row limit, dashboard mode)
- Reduces prop drilling across component tree
- Enables clean component interfaces

### 2. Custom Hooks
- `useTheme`: Manages dark/light mode with localStorage persistence
- `useTradingData`: Encapsulates data fetching logic and state
- `useVectorData`: Handles vector-specific data operations
- `usePagination`: Reusable pagination logic

### 3. Component Size Guidelines
- Each component is under 200 lines
- Single responsibility principle
- Clear prop interfaces with PropTypes

### 4. Utility Functions
- Centralized formatting functions
- Shared calculation logic
- Consistent constants across the app

## Data Flow

1. **Global State**: Managed by `TradingContext`
   - Selected symbol, timeframe, row limit
   - Dashboard mode (data/vector)

2. **Local State**: Component-specific state
   - Table sorting, pagination
   - Filter values, search terms
   - UI toggles (expanded sections, etc.)

3. **Data Fetching**: Custom hooks
   - Centralized API calls
   - Error handling
   - Loading states

## Component Guidelines

### Shared Components
- Must be completely reusable
- Accept all configuration via props
- No business logic, only presentation

### Feature Components
- Can use context and hooks
- Handle their own local state
- Compose shared components

### Container Components
- Orchestrate child components
- Handle data fetching
- Manage complex state logic

## Migration Path

1. **Phase 1**: Extract shared components ✅
   - InfoTooltip, ThemeToggle, LoadingSpinner, ErrorDisplay

2. **Phase 2**: Create utility functions ✅
   - Constants, formatters, calculations

3. **Phase 3**: Build custom hooks ✅
   - Theme, data fetching, pagination

4. **Phase 4**: Create feature components
   - Data dashboard components
   - Vector dashboard components

5. **Phase 5**: Wire everything together
   - Update imports
   - Connect context
   - Test functionality

## Benefits of This Architecture

1. **Maintainability**: Small, focused components are easier to understand and modify
2. **Reusability**: Shared components and hooks can be used across the app
3. **Testability**: Isolated components are easier to unit test
4. **Performance**: Smaller components can be optimized individually
5. **Scalability**: New features can be added without affecting existing code
6. **Developer Experience**: Clear structure makes onboarding easier

## Next Steps

1. Complete extraction of all components
2. Add comprehensive PropTypes/TypeScript
3. Implement unit tests for each component
4. Add Storybook for component documentation
5. Consider performance optimizations (React.memo, useMemo, etc.) 