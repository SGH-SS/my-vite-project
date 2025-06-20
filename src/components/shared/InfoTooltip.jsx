import { useState } from 'react';
import PropTypes from 'prop-types';

/**
 * InfoTooltip - A reusable tooltip component for displaying helpful information
 * @param {string} id - Unique identifier for the tooltip
 * @param {React.ReactNode} content - The content to display in the tooltip
 * @param {boolean} isDarkMode - Whether dark mode is enabled
 * @param {boolean} asSpan - Whether to render the trigger as a span instead of button
 */
const InfoTooltip = ({ id, content, isDarkMode, asSpan = false }) => {
  const [activeTooltip, setActiveTooltip] = useState(null);
  const isActive = activeTooltip === id;
  
  const triggerClasses = `ml-2 w-4 h-4 rounded-full border text-xs font-bold transition-all duration-200 cursor-pointer ${
    isDarkMode 
      ? 'border-gray-500 text-gray-400 hover:border-blue-400 hover:text-blue-400' 
      : 'border-gray-400 text-gray-500 hover:border-blue-500 hover:text-blue-600'
  }`;

  const triggerProps = {
    onClick: () => setActiveTooltip(isActive ? null : id),
    onMouseEnter: () => setActiveTooltip(id),
    onMouseLeave: () => setActiveTooltip(null),
    className: triggerClasses
  };
  
  return (
    <div className="relative">
      {asSpan ? (
        <span {...triggerProps}>i</span>
      ) : (
        <button {...triggerProps}>i</button>
      )}
      {isActive && (
        <div className={`absolute z-50 bottom-full left-1/2 transform -translate-x-1/2 mb-2 w-80 p-4 rounded-lg shadow-lg border transition-colors duration-200 ${
          isDarkMode 
            ? 'bg-gray-800 border-gray-600 text-gray-200' 
            : 'bg-white border-gray-200 text-gray-800'
        }`}>
          <div className="text-sm leading-relaxed">{content}</div>
          {/* Arrow pointing down */}
          <div className={`absolute top-full left-1/2 transform -translate-x-1/2 w-0 h-0 border-l-[8px] border-r-[8px] border-t-[8px] border-l-transparent border-r-transparent ${
            isDarkMode ? 'border-t-gray-800' : 'border-t-white'
          }`}></div>
        </div>
      )}
    </div>
  );
};

InfoTooltip.propTypes = {
  id: PropTypes.string.isRequired,
  content: PropTypes.node.isRequired,
  isDarkMode: PropTypes.bool.isRequired,
  asSpan: PropTypes.bool
};

export default InfoTooltip; 