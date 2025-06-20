/**
 * Vector calculation utility functions
 */

/**
 * Calculate statistics for a set of vectors
 * @param {Array<Array<number>>} vectors - Array of vectors
 * @returns {Object|null} Statistics object or null if invalid
 */
export const getVectorStats = (vectors) => {
  if (!vectors || vectors.length === 0) return null;
  
  const flatVectors = vectors.filter(v => v && Array.isArray(v)).flat();
  if (flatVectors.length === 0) return null;

  // Use iterative approach to avoid call stack overflow with large arrays
  let min = flatVectors[0];
  let max = flatVectors[0];
  let sum = 0;
  
  for (let i = 0; i < flatVectors.length; i++) {
    const val = flatVectors[i];
    if (val < min) min = val;
    if (val > max) max = val;
    sum += val;
  }
  
  const avg = sum / flatVectors.length;
  
  // Calculate standard deviation
  let sumSquaredDiffs = 0;
  for (let i = 0; i < flatVectors.length; i++) {
    sumSquaredDiffs += Math.pow(flatVectors[i] - avg, 2);
  }
  const std = Math.sqrt(sumSquaredDiffs / flatVectors.length);

  return { 
    min, 
    max, 
    avg, 
    std, 
    count: vectors.length, 
    dimensions: vectors[0]?.length || 0 
  };
};

/**
 * Calculate cosine similarity between two vectors
 * @param {Array<number>} vec1 - First vector
 * @param {Array<number>} vec2 - Second vector
 * @returns {number} Similarity score between 0 and 1
 */
export const calculateVectorSimilarity = (vec1, vec2) => {
  if (!vec1 || !vec2 || vec1.length !== vec2.length) return 0;
  
  // Calculate cosine similarity
  let dotProduct = 0;
  let norm1 = 0;
  let norm2 = 0;
  
  for (let i = 0; i < vec1.length; i++) {
    dotProduct += vec1[i] * vec2[i];
    norm1 += vec1[i] * vec1[i];
    norm2 += vec2[i] * vec2[i];
  }
  
  if (norm1 === 0 || norm2 === 0) return 0;
  return dotProduct / (Math.sqrt(norm1) * Math.sqrt(norm2));
};

/**
 * Get color from value based on min/max range
 * @param {number} value - The value to colorize
 * @param {number} min - Minimum value in range
 * @param {number} max - Maximum value in range
 * @returns {string} RGB color string
 */
export const getColorFromValue = (value, min, max) => {
  // Normalize value to 0-1 range
  const normalized = (value - min) / (max - min);
  
  // Create a color scale from blue (low) to red (high)
  if (normalized < 0.5) {
    // Blue to green
    const factor = normalized * 2;
    const r = Math.floor(0 * (1 - factor) + 0 * factor);
    const g = Math.floor(100 * (1 - factor) + 255 * factor);
    const b = Math.floor(255 * (1 - factor) + 0 * factor);
    return `rgb(${r}, ${g}, ${b})`;
  } else {
    // Green to red
    const factor = (normalized - 0.5) * 2;
    const r = Math.floor(0 * (1 - factor) + 255 * factor);
    const g = Math.floor(255 * (1 - factor) + 0 * factor);
    const b = Math.floor(0 * (1 - factor) + 0 * factor);
    return `rgb(${r}, ${g}, ${b})`;
  }
};

/**
 * Get similarity interpretation based on score
 * @param {number} similarity - Similarity score
 * @returns {Object} Interpretation object with text and color
 */
export const getSimilarityInterpretation = (similarity) => {
  if (similarity > 0.9) {
    return { text: 'Very Similar', color: 'text-green-600' };
  } else if (similarity > 0.7) {
    return { text: 'Similar', color: 'text-yellow-600' };
  } else if (similarity > 0.5) {
    return { text: 'Somewhat Similar', color: 'text-orange-600' };
  } else {
    return { text: 'Different', color: 'text-red-600' };
  }
}; 