/**
 * Feature Dashboard - Databento Feature Catalog & Explorer
 * 
 * Comprehensive dashboard for exploring, understanding, and cataloging
 * all ML features from the databento_es_features schema.
 * 
 * Features:
 * - Full feature catalog with descriptions, types, and model use tags
 * - Summary statistics (min/max/mean/std/median) per feature
 * - Null percentage tracking for data quality
 * - Group-by model purpose (regime vs entry)
 * - Expandable detail panels with sample data
 * - Date range and row count metadata
 */

import { useState, useEffect, useCallback, useMemo } from 'react';

// =============================================================================
// CONSTANTS
// =============================================================================

const API_BASE = 'http://localhost:8000/api/databento';

const MODEL_USE_COLORS = {
  regime: { bg: 'bg-purple-100', text: 'text-purple-800', darkBg: 'bg-purple-900/40', darkText: 'text-purple-300', border: 'border-purple-500' },
  entry: { bg: 'bg-amber-100', text: 'text-amber-800', darkBg: 'bg-amber-900/40', darkText: 'text-amber-300', border: 'border-amber-500' },
};

const GROUP_ICONS = {
  volatility: '🌊',
  momentum: '🚀',
  structure: '🏗️',
  toxicity: '☠️',
  book_shape: '📖',
  quote_dynamics: '⚡',
  trade_clustering: '🎯',
  cross_contract: '🔗',
  events: '💥',
};

const TYPE_COLORS = {
  continuous: 'text-blue-500',
  ratio: 'text-green-500',
  zscore: 'text-indigo-500',
  percentile: 'text-teal-500',
  bounded: 'text-orange-500',
  deviation: 'text-cyan-500',
  price: 'text-gray-500',
  volume: 'text-pink-500',
  count: 'text-yellow-600',
  time: 'text-violet-500',
  categorical: 'text-rose-500',
  composite: 'text-red-500',
  divergence: 'text-emerald-500',
};

const VIEW_MODES = {
  CATALOG: 'catalog',
  BY_MODEL: 'by_model',
  STATS: 'stats',
  DATA_QUALITY: 'data_quality',
};

// =============================================================================
// UTILITY FUNCTIONS
// =============================================================================

const formatNumber = (val, decimals = 4) => {
  if (val === null || val === undefined) return 'N/A';
  if (Math.abs(val) > 1000000) return `${(val / 1000000).toFixed(2)}M`;
  if (Math.abs(val) > 1000) return `${(val / 1000).toFixed(1)}K`;
  return val.toFixed(decimals);
};

const formatRows = (n) => {
  if (!n) return '0';
  return n.toLocaleString();
};

const getNullColor = (pct, isDark) => {
  if (pct === 0) return isDark ? 'text-green-400' : 'text-green-600';
  if (pct < 10) return isDark ? 'text-yellow-400' : 'text-yellow-600';
  if (pct < 50) return isDark ? 'text-orange-400' : 'text-orange-600';
  return isDark ? 'text-red-400' : 'text-red-600';
};

// =============================================================================
// INFO TOOLTIP
// =============================================================================

const InfoTooltip = ({ content, isDarkMode }) => {
  const [isActive, setIsActive] = useState(false);
  return (
    <div className="relative inline-flex">
      <button
        onMouseEnter={() => setIsActive(true)}
        onMouseLeave={() => setIsActive(false)}
        className={`ml-1.5 w-4 h-4 rounded-full border text-xs font-bold cursor-pointer flex items-center justify-center ${
          isDarkMode
            ? 'border-gray-500 text-gray-400 hover:border-blue-400 hover:text-blue-400'
            : 'border-gray-400 text-gray-500 hover:border-blue-500 hover:text-blue-600'
        }`}
      >i</button>
      {isActive && (
        <div className={`absolute z-50 bottom-full left-1/2 -translate-x-1/2 mb-2 w-72 p-3 rounded-lg shadow-lg border text-sm ${
          isDarkMode ? 'bg-gray-800 border-gray-600 text-gray-200' : 'bg-white border-gray-200 text-gray-800'
        }`}>
          {content}
          <div className={`absolute top-full left-1/2 -translate-x-1/2 w-0 h-0 border-l-8 border-r-8 border-t-8 border-l-transparent border-r-transparent ${
            isDarkMode ? 'border-t-gray-800' : 'border-t-white'
          }`} />
        </div>
      )}
    </div>
  );
};

// =============================================================================
// MODEL USE TAG
// =============================================================================

const ModelUseTag = ({ use, isDarkMode }) => {
  const c = MODEL_USE_COLORS[use] || MODEL_USE_COLORS.regime;
  return (
    <span className={`inline-flex items-center px-2 py-0.5 rounded-full text-xs font-semibold ${
      isDarkMode ? `${c.darkBg} ${c.darkText}` : `${c.bg} ${c.text}`
    }`}>
      {use === 'regime' ? '🔮 Regime' : '🎯 Entry'}
    </span>
  );
};

// =============================================================================
// FEATURE TYPE BADGE
// =============================================================================

const TypeBadge = ({ type, isDarkMode }) => (
  <span className={`text-xs font-mono px-1.5 py-0.5 rounded ${
    isDarkMode ? 'bg-gray-700' : 'bg-gray-100'
  } ${TYPE_COLORS[type] || 'text-gray-500'}`}>
    {type}
  </span>
);

// =============================================================================
// STAT BAR (mini visualization)
// =============================================================================

const StatBar = ({ value, min, max, isDarkMode }) => {
  if (value === null || min === null || max === null || min === max) return null;
  const pct = Math.max(0, Math.min(100, ((value - min) / (max - min)) * 100));
  return (
    <div className={`w-full h-1.5 rounded-full ${isDarkMode ? 'bg-gray-700' : 'bg-gray-200'}`}>
      <div
        className="h-full rounded-full bg-blue-500 transition-all"
        style={{ width: `${pct}%` }}
      />
    </div>
  );
};

// =============================================================================
// NULL PERCENTAGE BAR
// =============================================================================

const NullBar = ({ pct, isDarkMode }) => {
  const color = pct === 0 ? 'bg-green-500' : pct < 10 ? 'bg-yellow-500' : pct < 50 ? 'bg-orange-500' : 'bg-red-500';
  return (
    <div className="flex items-center gap-2 w-full">
      <div className={`flex-1 h-2 rounded-full ${isDarkMode ? 'bg-gray-700' : 'bg-gray-200'}`}>
        <div className={`h-full rounded-full ${color} transition-all`} style={{ width: `${Math.min(100, pct)}%` }} />
      </div>
      <span className={`text-xs font-mono w-12 text-right ${getNullColor(pct, isDarkMode)}`}>{pct.toFixed(1)}%</span>
    </div>
  );
};

// =============================================================================
// TABLE HEADER CARD
// =============================================================================

const TableHeaderCard = ({ table, isExpanded, onToggle, isDarkMode }) => {
  const icon = GROUP_ICONS[table.group] || '📊';
  const dateMin = table.date_range?.min ? new Date(table.date_range.min).toLocaleDateString() : 'N/A';
  const dateMax = table.date_range?.max ? new Date(table.date_range.max).toLocaleDateString() : 'N/A';
  const featureCount = Object.keys(table.columns_meta).length;

  return (
    <button
      onClick={onToggle}
      className={`w-full text-left rounded-lg border transition-all ${
        isExpanded
          ? isDarkMode ? 'bg-gray-750 border-blue-500 shadow-lg shadow-blue-900/20' : 'bg-blue-50 border-blue-300 shadow-lg shadow-blue-100'
          : isDarkMode ? 'bg-gray-800 border-gray-700 hover:border-gray-600' : 'bg-white border-gray-200 hover:border-gray-300'
      }`}
    >
      <div className="p-4">
        <div className="flex items-start justify-between">
          <div className="flex items-start gap-3">
            <span className="text-2xl mt-0.5">{icon}</span>
            <div>
              <h3 className={`text-lg font-bold ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>
                {table.group_label}
              </h3>
              <p className={`text-sm mt-0.5 ${isDarkMode ? 'text-gray-400' : 'text-gray-600'}`}>
                {table.description}
              </p>
              <div className="flex items-center gap-2 mt-2 flex-wrap">
                {table.model_use.map(u => <ModelUseTag key={u} use={u} isDarkMode={isDarkMode} />)}
                <span className={`text-xs ${isDarkMode ? 'text-gray-500' : 'text-gray-400'}`}>
                  {table.table}
                </span>
              </div>
            </div>
          </div>
          <div className="flex flex-col items-end gap-1 shrink-0 ml-4">
            <span className={`text-sm font-bold ${isDarkMode ? 'text-blue-400' : 'text-blue-600'}`}>
              {featureCount} features
            </span>
            <span className={`text-xs ${isDarkMode ? 'text-gray-500' : 'text-gray-400'}`}>
              {formatRows(table.row_count)} rows
            </span>
            <span className={`text-xs ${isDarkMode ? 'text-gray-500' : 'text-gray-400'}`}>
              {dateMin} - {dateMax}
            </span>
            <svg className={`w-5 h-5 mt-1 transition-transform ${isExpanded ? 'rotate-180' : ''} ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}
              fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
            </svg>
          </div>
        </div>
      </div>
    </button>
  );
};

// =============================================================================
// FEATURE DETAIL ROW
// =============================================================================

const FeatureRow = ({ name, meta, stats, isDarkMode }) => {
  const s = stats?.[name];
  return (
    <div className={`grid grid-cols-12 gap-3 items-center py-2.5 px-4 border-b last:border-b-0 ${
      isDarkMode ? 'border-gray-700/50' : 'border-gray-100'
    }`}>
      {/* Name + Description */}
      <div className="col-span-4">
        <div className="flex items-center gap-1.5">
          <span className={`font-semibold text-sm ${isDarkMode ? 'text-gray-200' : 'text-gray-800'}`}>
            {meta.label}
          </span>
          <TypeBadge type={meta.type} isDarkMode={isDarkMode} />
        </div>
        <p className={`text-xs mt-0.5 leading-relaxed ${isDarkMode ? 'text-gray-500' : 'text-gray-400'}`}>
          {meta.description}
        </p>
      </div>

      {/* Null % */}
      <div className="col-span-2">
        <NullBar pct={meta.null_pct ?? 0} isDarkMode={isDarkMode} />
      </div>

      {/* Stats */}
      {s ? (
        <>
          <div className="col-span-1 text-right">
            <span className={`text-xs font-mono ${isDarkMode ? 'text-gray-400' : 'text-gray-600'}`}>
              {formatNumber(s.min, 3)}
            </span>
          </div>
          <div className="col-span-1 text-right">
            <span className={`text-xs font-mono font-bold ${isDarkMode ? 'text-gray-200' : 'text-gray-800'}`}>
              {formatNumber(s.mean, 3)}
            </span>
          </div>
          <div className="col-span-1 text-right">
            <span className={`text-xs font-mono ${isDarkMode ? 'text-gray-400' : 'text-gray-600'}`}>
              {formatNumber(s.max, 3)}
            </span>
          </div>
          <div className="col-span-1 text-right">
            <span className={`text-xs font-mono ${isDarkMode ? 'text-gray-500' : 'text-gray-400'}`}>
              {formatNumber(s.std, 3)}
            </span>
          </div>
          <div className="col-span-2">
            <StatBar value={s.median} min={s.min} max={s.max} isDarkMode={isDarkMode} />
            <span className={`text-xs font-mono ${isDarkMode ? 'text-gray-500' : 'text-gray-400'}`}>
              med: {formatNumber(s.median, 3)}
            </span>
          </div>
        </>
      ) : (
        <div className="col-span-6">
          <span className={`text-xs italic ${isDarkMode ? 'text-gray-600' : 'text-gray-400'}`}>
            Loading stats...
          </span>
        </div>
      )}
    </div>
  );
};

// =============================================================================
// EXPANDED TABLE PANEL
// =============================================================================

const ExpandedPanel = ({ table, stats, statsLoading, isDarkMode }) => {
  const features = Object.entries(table.columns_meta);

  return (
    <div className={`rounded-lg border overflow-hidden ${
      isDarkMode ? 'bg-gray-800/50 border-gray-700' : 'bg-white border-gray-200'
    }`}>
      {/* Column Headers */}
      <div className={`grid grid-cols-12 gap-3 px-4 py-2 text-xs font-semibold uppercase tracking-wider border-b ${
        isDarkMode ? 'bg-gray-800 border-gray-700 text-gray-400' : 'bg-gray-50 border-gray-200 text-gray-500'
      }`}>
        <div className="col-span-4">Feature</div>
        <div className="col-span-2">Null %</div>
        <div className="col-span-1 text-right">Min</div>
        <div className="col-span-1 text-right">Mean</div>
        <div className="col-span-1 text-right">Max</div>
        <div className="col-span-1 text-right">Std</div>
        <div className="col-span-2">Median</div>
      </div>

      {/* Feature Rows */}
      {statsLoading ? (
        <div className="flex items-center justify-center py-8">
          <div className={`animate-spin rounded-full h-6 w-6 border-b-2 ${isDarkMode ? 'border-blue-400' : 'border-blue-600'}`} />
          <span className={`ml-3 text-sm ${isDarkMode ? 'text-gray-400' : 'text-gray-600'}`}>Loading statistics...</span>
        </div>
      ) : (
        features.map(([name, meta]) => (
          <FeatureRow key={name} name={name} meta={meta} stats={stats} isDarkMode={isDarkMode} />
        ))
      )}
    </div>
  );
};

// =============================================================================
// SUMMARY CARDS
// =============================================================================

const SummaryCards = ({ catalog, isDarkMode }) => {
  const totalFeatures = catalog.total_features || 0;
  const totalRows = catalog.tables?.reduce((sum, t) => sum + (t.row_count || 0), 0) || 0;
  const regimeTables = catalog.tables?.filter(t => t.model_use.includes('regime')) || [];
  const entryTables = catalog.tables?.filter(t => t.model_use.includes('entry')) || [];
  const regimeFeatures = regimeTables.reduce((sum, t) => sum + Object.keys(t.columns_meta).length, 0);
  const entryFeatures = entryTables.reduce((sum, t) => sum + Object.keys(t.columns_meta).length, 0);

  // Find date coverage
  const allDates = catalog.tables?.flatMap(t => [t.date_range?.min, t.date_range?.max]).filter(Boolean) || [];
  const minDate = allDates.length ? new Date(Math.min(...allDates.map(d => new Date(d)))).toLocaleDateString() : 'N/A';
  const maxDate = allDates.length ? new Date(Math.max(...allDates.map(d => new Date(d)))).toLocaleDateString() : 'N/A';

  // Full-history vs microstructure-only tables
  const fullHistory = catalog.tables?.filter(t => (t.row_count || 0) > 500000) || [];
  const microOnly = catalog.tables?.filter(t => (t.row_count || 0) > 0 && (t.row_count || 0) <= 500000) || [];

  const cards = [
    { label: 'Total Features', value: totalFeatures, sub: `across ${catalog.table_count || 0} tables`, icon: '📊' },
    { label: 'Total Rows', value: formatRows(totalRows), sub: `${minDate} - ${maxDate}`, icon: '🗃️' },
    { label: 'Regime Features', value: regimeFeatures, sub: `${regimeTables.length} tables (1D/1M primary)`, icon: '🔮', color: 'purple' },
    { label: 'Entry Features', value: entryFeatures, sub: `${entryTables.length} tables (micro primary)`, icon: '🎯', color: 'amber' },
    { label: 'Full History (5yr)', value: `${fullHistory.length} tables`, sub: 'Jan 2021 - Jan 2026', icon: '📅', color: 'green' },
    { label: 'Microstructure (~50d)', value: `${microOnly.length} tables`, sub: 'Dec 2025 - Jan 2026', icon: '🔬', color: 'blue' },
  ];

  return (
    <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3">
      {cards.map((card, i) => (
        <div key={i} className={`rounded-lg border p-4 ${
          isDarkMode ? 'bg-gray-800 border-gray-700' : 'bg-white border-gray-200'
        }`}>
          <div className="flex items-center gap-2">
            <span className="text-xl">{card.icon}</span>
            <span className={`text-xs font-medium uppercase tracking-wider ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>
              {card.label}
            </span>
          </div>
          <div className={`text-2xl font-black mt-2 ${
            card.color === 'purple' ? (isDarkMode ? 'text-purple-400' : 'text-purple-600') :
            card.color === 'amber' ? (isDarkMode ? 'text-amber-400' : 'text-amber-600') :
            card.color === 'green' ? (isDarkMode ? 'text-green-400' : 'text-green-600') :
            card.color === 'blue' ? (isDarkMode ? 'text-blue-400' : 'text-blue-600') :
            isDarkMode ? 'text-white' : 'text-gray-900'
          }`}>
            {card.value}
          </div>
          <div className={`text-xs mt-1 ${isDarkMode ? 'text-gray-500' : 'text-gray-400'}`}>
            {card.sub}
          </div>
        </div>
      ))}
    </div>
  );
};

// =============================================================================
// MODEL USE VIEW
// =============================================================================

const ModelUseView = ({ catalog, isDarkMode }) => {
  const regimeTables = catalog.tables?.filter(t => t.model_use.includes('regime')) || [];
  const entryTables = catalog.tables?.filter(t => t.model_use.includes('entry')) || [];

  const Section = ({ title, icon, description, tables, color }) => (
    <div className={`rounded-lg border ${isDarkMode ? 'bg-gray-800 border-gray-700' : 'bg-white border-gray-200'}`}>
      <div className={`p-5 border-b ${isDarkMode ? 'border-gray-700' : 'border-gray-200'}`}>
        <h3 className={`text-xl font-bold ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>
          {icon} {title}
        </h3>
        <p className={`text-sm mt-1 ${isDarkMode ? 'text-gray-400' : 'text-gray-600'}`}>{description}</p>
      </div>
      <div className="p-4 space-y-3">
        {tables.map(table => {
          const featureCount = Object.keys(table.columns_meta).length;
          return (
            <div key={table.table} className={`rounded-lg p-4 ${isDarkMode ? 'bg-gray-750 border border-gray-700' : 'bg-gray-50 border border-gray-100'}`}>
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <span className="text-lg">{GROUP_ICONS[table.group]}</span>
                  <span className={`font-semibold ${isDarkMode ? 'text-gray-200' : 'text-gray-800'}`}>{table.group_label}</span>
                  <span className={`text-xs ${isDarkMode ? 'text-gray-500' : 'text-gray-400'}`}>({table.table})</span>
                </div>
                <div className="flex items-center gap-3">
                  <span className={`text-sm font-bold ${isDarkMode ? 'text-blue-400' : 'text-blue-600'}`}>{featureCount} features</span>
                  <span className={`text-xs ${isDarkMode ? 'text-gray-500' : 'text-gray-400'}`}>{formatRows(table.row_count)} rows</span>
                </div>
              </div>
              <div className="mt-2 flex flex-wrap gap-1.5">
                {Object.entries(table.columns_meta).map(([name, meta]) => (
                  <span key={name} className={`inline-flex items-center text-xs px-2 py-1 rounded-md ${
                    isDarkMode ? 'bg-gray-700 text-gray-300' : 'bg-white text-gray-700 border border-gray-200'
                  }`}>
                    <span className={`w-1.5 h-1.5 rounded-full mr-1.5 ${
                      (meta.null_pct ?? 0) === 0 ? 'bg-green-500' : (meta.null_pct ?? 0) < 50 ? 'bg-yellow-500' : 'bg-red-500'
                    }`} />
                    {meta.label}
                  </span>
                ))}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
      <Section
        title="Regime Detection"
        icon="🔮"
        description="Primary: 1D/1M OHLCV. These features characterize the market state (trending/ranging, volatile/calm, liquid/thin). Use as main inputs for regime classification models."
        tables={regimeTables}
        color="purple"
      />
      <Section
        title="Entry / Execution"
        icon="🎯"
        description="Primary: Trades/TBBO/MBP-10. These features capture real-time microstructure for precise entry timing. Use regime output as a confirmation input."
        tables={entryTables}
        color="amber"
      />
    </div>
  );
};

// =============================================================================
// DATA QUALITY VIEW
// =============================================================================

const DataQualityView = ({ catalog, isDarkMode }) => {
  // Flatten all features with their null percentages
  const allFeatures = [];
  for (const table of (catalog.tables || [])) {
    for (const [name, meta] of Object.entries(table.columns_meta)) {
      allFeatures.push({
        name,
        label: meta.label,
        table: table.table,
        group: table.group_label,
        icon: GROUP_ICONS[table.group],
        null_pct: meta.null_pct ?? 0,
        type: meta.type,
        row_count: table.row_count,
      });
    }
  }

  // Sort by null percentage descending
  allFeatures.sort((a, b) => b.null_pct - a.null_pct);

  const cleanFeatures = allFeatures.filter(f => f.null_pct === 0);
  const partialFeatures = allFeatures.filter(f => f.null_pct > 0 && f.null_pct < 100);
  const emptyFeatures = allFeatures.filter(f => f.null_pct >= 100);

  return (
    <div className="space-y-6">
      {/* Summary */}
      <div className="grid grid-cols-3 gap-4">
        <div className={`rounded-lg border p-4 text-center ${isDarkMode ? 'bg-gray-800 border-gray-700' : 'bg-white border-gray-200'}`}>
          <div className={`text-3xl font-black ${isDarkMode ? 'text-green-400' : 'text-green-600'}`}>{cleanFeatures.length}</div>
          <div className={`text-sm ${isDarkMode ? 'text-gray-400' : 'text-gray-600'}`}>Fully Populated (0% null)</div>
        </div>
        <div className={`rounded-lg border p-4 text-center ${isDarkMode ? 'bg-gray-800 border-gray-700' : 'bg-white border-gray-200'}`}>
          <div className={`text-3xl font-black ${isDarkMode ? 'text-yellow-400' : 'text-yellow-600'}`}>{partialFeatures.length}</div>
          <div className={`text-sm ${isDarkMode ? 'text-gray-400' : 'text-gray-600'}`}>Partially Populated</div>
        </div>
        <div className={`rounded-lg border p-4 text-center ${isDarkMode ? 'bg-gray-800 border-gray-700' : 'bg-white border-gray-200'}`}>
          <div className={`text-3xl font-black ${isDarkMode ? 'text-red-400' : 'text-red-600'}`}>{emptyFeatures.length}</div>
          <div className={`text-sm ${isDarkMode ? 'text-gray-400' : 'text-gray-600'}`}>Fully Null (needs attention)</div>
        </div>
      </div>

      {/* Feature list sorted by null percentage */}
      <div className={`rounded-lg border overflow-hidden ${isDarkMode ? 'bg-gray-800 border-gray-700' : 'bg-white border-gray-200'}`}>
        <div className={`px-4 py-3 border-b ${isDarkMode ? 'bg-gray-800 border-gray-700' : 'bg-gray-50 border-gray-200'}`}>
          <h3 className={`font-semibold ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>
            All Features by Data Completeness
          </h3>
        </div>
        <div className="max-h-[600px] overflow-y-auto">
          {allFeatures.map((f, i) => (
            <div key={`${f.table}-${f.name}`} className={`flex items-center gap-3 px-4 py-2 border-b last:border-b-0 ${
              isDarkMode ? 'border-gray-700/50' : 'border-gray-100'
            }`}>
              <span className="text-sm w-6 text-right shrink-0">{GROUP_ICONS[catalog.tables?.find(t => t.table === f.table)?.group]}</span>
              <span className={`text-sm font-medium w-48 shrink-0 ${isDarkMode ? 'text-gray-200' : 'text-gray-800'}`}>
                {f.label}
              </span>
              <span className={`text-xs w-40 shrink-0 ${isDarkMode ? 'text-gray-500' : 'text-gray-400'}`}>
                {f.group}
              </span>
              <div className="flex-1">
                <NullBar pct={f.null_pct} isDarkMode={isDarkMode} />
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

// =============================================================================
// MAIN FEATURE DASHBOARD
// =============================================================================

const FeatureDashboard = ({ isDarkMode }) => {
  const [catalog, setCatalog] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [expandedTables, setExpandedTables] = useState(new Set());
  const [tableStats, setTableStats] = useState({});
  const [statsLoading, setStatsLoading] = useState(new Set());
  const [viewMode, setViewMode] = useState(VIEW_MODES.CATALOG);
  const [searchTerm, setSearchTerm] = useState('');

  // Fetch catalog on mount
  useEffect(() => {
    const fetchCatalog = async () => {
      setLoading(true);
      setError(null);
      try {
        const res = await fetch(`${API_BASE}/features/catalog`);
        if (!res.ok) throw new Error(`HTTP ${res.status}: ${res.statusText}`);
        const data = await res.json();
        setCatalog(data);
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };
    fetchCatalog();
  }, []);

  // Fetch stats when a table is expanded
  const fetchStats = useCallback(async (tableName) => {
    if (tableStats[tableName] || statsLoading.has(tableName)) return;
    setStatsLoading(prev => new Set(prev).add(tableName));
    try {
      const res = await fetch(`${API_BASE}/features/sample-stats/${tableName}`);
      if (res.ok) {
        const data = await res.json();
        setTableStats(prev => ({ ...prev, [tableName]: data.stats }));
      }
    } catch (err) {
      console.error(`Failed to fetch stats for ${tableName}:`, err);
    } finally {
      setStatsLoading(prev => {
        const next = new Set(prev);
        next.delete(tableName);
        return next;
      });
    }
  }, [tableStats, statsLoading]);

  const toggleTable = useCallback((tableName) => {
    setExpandedTables(prev => {
      const next = new Set(prev);
      if (next.has(tableName)) {
        next.delete(tableName);
      } else {
        next.add(tableName);
        fetchStats(tableName);
      }
      return next;
    });
  }, [fetchStats]);

  // Filter tables by search
  const filteredTables = useMemo(() => {
    if (!catalog?.tables) return [];
    if (!searchTerm.trim()) return catalog.tables;
    const term = searchTerm.toLowerCase();
    return catalog.tables.filter(t => {
      if (t.group_label.toLowerCase().includes(term)) return true;
      if (t.table.toLowerCase().includes(term)) return true;
      if (t.description.toLowerCase().includes(term)) return true;
      return Object.values(t.columns_meta).some(m =>
        m.label.toLowerCase().includes(term) || m.description.toLowerCase().includes(term)
      );
    });
  }, [catalog, searchTerm]);

  // Loading state
  if (loading) {
    return (
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="flex items-center justify-center h-64">
          <div className={`animate-spin rounded-full h-12 w-12 border-b-2 ${isDarkMode ? 'border-blue-400' : 'border-blue-600'}`} />
          <span className={`ml-3 text-lg ${isDarkMode ? 'text-gray-300' : 'text-gray-600'}`}>Loading feature catalog...</span>
        </div>
      </div>
    );
  }

  // Error state
  if (error) {
    return (
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className={`rounded-lg p-6 border ${isDarkMode ? 'bg-red-900/20 border-red-800' : 'bg-red-50 border-red-200'}`}>
          <h3 className="text-red-500 font-bold text-lg">Failed to load Feature Catalog</h3>
          <p className="text-red-400 mt-2">{error}</p>
          <p className={`text-sm mt-3 ${isDarkMode ? 'text-gray-400' : 'text-gray-600'}`}>
            Make sure the backend is running at {API_BASE} and the database is accessible.
          </p>
          <button onClick={() => window.location.reload()}
            className="mt-4 px-4 py-2 bg-red-600 text-white rounded-lg text-sm hover:bg-red-700">
            Retry
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-6">
      {/* Header */}
      <div className={`p-6 rounded-lg ${
        isDarkMode
          ? 'bg-gradient-to-r from-indigo-900 via-purple-900 to-pink-900'
          : 'bg-gradient-to-r from-indigo-600 via-purple-600 to-pink-600'
      } text-white`}>
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-3xl font-black tracking-tight" style={{ textShadow: '0 2px 4px rgba(0,0,0,0.3)' }}>
              Day<span className="text-purple-200">gent</span>{' '}
              <span className="text-xl font-semibold text-purple-100">Feature Catalog</span>
            </h2>
            <p className="text-purple-100 mt-1">
              {catalog?.total_features || 0} engineered features across {catalog?.table_count || 0} tables from Databento ES futures data
            </p>
          </div>
          <InfoTooltip isDarkMode={isDarkMode} content={
            <div>
              <p className="font-semibold mb-2">Feature Dashboard</p>
              <p>Explore all ML features extracted from your Databento data pipeline. Features are organized by model purpose (regime detection vs entry timing) and data source (OHLCV, TBBO, MBP-10, trades).</p>
            </div>
          } />
        </div>
      </div>

      {/* Summary Cards */}
      <SummaryCards catalog={catalog} isDarkMode={isDarkMode} />

      {/* View Mode Tabs + Search */}
      <div className={`flex items-center justify-between flex-wrap gap-4 rounded-lg p-3 ${
        isDarkMode ? 'bg-gray-800' : 'bg-white border border-gray-200'
      }`}>
        <div className={`flex rounded-lg p-1 ${isDarkMode ? 'bg-gray-700' : 'bg-gray-100'}`}>
          {[
            { key: VIEW_MODES.CATALOG, label: '📋 Full Catalog', },
            { key: VIEW_MODES.BY_MODEL, label: '🔮 By Model Use', },
            { key: VIEW_MODES.DATA_QUALITY, label: '🔍 Data Quality', },
          ].map(tab => (
            <button
              key={tab.key}
              onClick={() => setViewMode(tab.key)}
              className={`px-4 py-2 text-sm font-medium rounded-md transition-all ${
                viewMode === tab.key
                  ? isDarkMode ? 'bg-gray-600 text-white shadow-sm' : 'bg-white text-gray-900 shadow-sm'
                  : isDarkMode ? 'text-gray-300 hover:text-white' : 'text-gray-600 hover:text-gray-900'
              }`}
            >
              {tab.label}
            </button>
          ))}
        </div>

        {viewMode === VIEW_MODES.CATALOG && (
          <div className="relative">
            <input
              type="text"
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              placeholder="Search features..."
              className={`w-64 rounded-lg px-4 py-2 text-sm border focus:outline-none focus:ring-2 focus:ring-blue-500 ${
                isDarkMode ? 'bg-gray-700 border-gray-600 text-white placeholder-gray-400' : 'bg-white border-gray-300 text-gray-900 placeholder-gray-400'
              }`}
            />
            {searchTerm && (
              <button onClick={() => setSearchTerm('')}
                className={`absolute right-2 top-1/2 -translate-y-1/2 text-sm ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>
                ✕
              </button>
            )}
          </div>
        )}
      </div>

      {/* View Content */}
      {viewMode === VIEW_MODES.CATALOG && (
        <div className="space-y-3">
          {filteredTables.length === 0 ? (
            <div className={`text-center py-12 ${isDarkMode ? 'text-gray-500' : 'text-gray-400'}`}>
              No features match "{searchTerm}"
            </div>
          ) : (
            filteredTables.map(table => (
              <div key={table.table}>
                <TableHeaderCard
                  table={table}
                  isExpanded={expandedTables.has(table.table)}
                  onToggle={() => toggleTable(table.table)}
                  isDarkMode={isDarkMode}
                />
                {expandedTables.has(table.table) && (
                  <div className="mt-2 ml-4">
                    <ExpandedPanel
                      table={table}
                      stats={tableStats[table.table]}
                      statsLoading={statsLoading.has(table.table)}
                      isDarkMode={isDarkMode}
                    />
                  </div>
                )}
              </div>
            ))
          )}
        </div>
      )}

      {viewMode === VIEW_MODES.BY_MODEL && (
        <ModelUseView catalog={catalog} isDarkMode={isDarkMode} />
      )}

      {viewMode === VIEW_MODES.DATA_QUALITY && (
        <DataQualityView catalog={catalog} isDarkMode={isDarkMode} />
      )}

      {/* Architecture Note */}
      <div className={`rounded-lg border p-4 ${isDarkMode ? 'bg-gray-800/50 border-gray-700' : 'bg-gray-50 border-gray-200'}`}>
        <h4 className={`text-sm font-semibold mb-2 ${isDarkMode ? 'text-gray-300' : 'text-gray-700'}`}>
          Pipeline Architecture
        </h4>
        <div className={`text-xs leading-relaxed space-y-1 ${isDarkMode ? 'text-gray-500' : 'text-gray-400'}`}>
          <p><strong>Full History (5 yrs):</strong> volatility_1m, rolling_features_1m, price_levels_1m, cross_contract_1m - Derived from 1D/1M OHLCV data (Jan 2021 - Jan 2026)</p>
          <p><strong>Microstructure (~50 days):</strong> book_shape_1m, quote_dynamics_1m, trade_clustering_1m, microstructure_toxicity, sweep_iceberg_events - Derived from TBBO/MBP-10/Trades (Dec 2025 - Jan 2026)</p>
          <p><strong>Views:</strong> v_features_complete_1m joins OHLCV + microstructure + all features for a single unified query surface</p>
        </div>
      </div>
    </div>
  );
};

export default FeatureDashboard;
