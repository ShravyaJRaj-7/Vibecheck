/**
 * Modular NLP & Machine Learning Engine for VibeCheck Travel
 * Implements client-side TF-IDF vectorization and Cosine Similarity calculations to analyze travel preferences.
 */

// Basic list of English stopwords to clean features
const STOP_WORDS = new Set([
  "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at",
  "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "can", "could",
  "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had", "has",
  "have", "having", "he", "her", "here", "hers", "herself", "him", "himself", "his", "how", "i", "if", "in",
  "into", "is", "it", "its", "itself", "just", "me", "more", "most", "my", "myself", "no", "nor", "not", "of",
  "off", "on", "once", "only", "or", "other", "our", "ours", "ourselves", "out", "over", "own", "same", "she",
  "should", "so", "some", "such", "than", "that", "the", "their", "theirs", "them", "themselves", "then",
  "there", "these", "they", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was",
  "we", "were", "what", "when", "where", "which", "while", "who", "whom", "why", "with", "you", "your", "yours",
  "yourself", "yourselves", "want", "like", "love", "would", "prefer", "trip", "travel", "vacation"
]);

/**
 * Normalizes text and tokenizes it by stripping punctuation and removing top words.
 */
export function tokenize(text: string): string[] {
  return text
    .toLowerCase()
    .replace(/[.,\/#!$%\^&\*;:{}=\-_`~()?"']/g, " ")
    .split(/\s+/)
    .filter(word => word.length > 2 && !STOP_WORDS.has(word) && /^[a-z]+$/.test(word));
}

export interface TFIDFResult {
  vocabulary: string[];
  vectors: number[][];
  wordWeights: { word: string; weight: number }[];
}

/**
 * Custom TF-IDF (Term Frequency - Inverse Document Frequency) Vectorizer Class.
 * Quantifies freeform travel request prompts into measurable vector coordinates.
 */
export function computeTFIDF(documents: string[]): TFIDFResult {
  const tokenizedDocs = documents.map(tokenize);
  
  // 1. Build Global Unique Vocabulary
  const vocabSet = new Set<string>();
  tokenizedDocs.forEach(tokens => tokens.forEach(t => vocabSet.add(t)));
  const vocabulary = Array.from(vocabSet).sort();
  
  const N = documents.length;
  if (N === 0 || vocabulary.length === 0) {
    return { vocabulary: [], vectors: [], wordWeights: [] };
  }

  // 2. Calculate Smoothed Inverse Document Frequencies (IDF)
  const idf: Record<string, number> = {};
  vocabulary.forEach(word => {
    const docFrequency = tokenizedDocs.filter(tokens => tokens.includes(word)).length;
    // Smoothed logarithmic IDF formula
    idf[word] = Math.log(1 + N / (1 + docFrequency));
  });

  // 3. Compute TF-IDF Coordinate Matrix
  const vectors = tokenizedDocs.map(tokens => {
    const termCount: Record<string, number> = {};
    tokens.forEach(word => {
      termCount[word] = (termCount[word] || 0) + 1;
    });

    return vocabulary.map(word => {
      if (!termCount[word]) return 0;
      const tf = termCount[word] / tokens.length;
      return tf * idf[word];
    });
  });

  // 4. Summarize and Rank Keywords globally
  const globalWeights: Record<string, number> = {};
  vectors.forEach(vector => {
    vocabulary.forEach((word, index) => {
      const val = vector[index];
      if (val > 0) {
        globalWeights[word] = (globalWeights[word] || 0) + val;
      }
    });
  });

  const wordWeights = Object.entries(globalWeights)
    .map(([word, weight]) => ({ word, weight: weight / N }))
    .sort((a, b) => b.weight - a.weight);

  return { vocabulary, vectors, wordWeights };
}

/**
 * Computes the Cosine Similarity metric between two high-dimensional text vectors.
 * Returns a value [0.0 - 1.0] indicating mutual direction alignment of wishes.
 */
export function calculateCosineSimilarity(v1: number[], v2: number[]): number {
  if (v1.length !== v2.length || v1.length === 0) return 0;
  
  let dotProduct = 0;
  let magnitudeSquare1 = 0;
  let magnitudeSquare2 = 0;

  for (let i = 0; i < v1.length; i++) {
    dotProduct += v1[i] * v2[i];
    magnitudeSquare1 += v1[i] * v1[i];
    magnitudeSquare2 += v2[i] * v2[i];
  }

  const mag1 = Math.sqrt(magnitudeSquare1);
  const mag2 = Math.sqrt(magnitudeSquare2);

  if (mag1 === 0 || mag2 === 0) return 0;
  return dotProduct / (mag1 * mag2);
}

export interface CompatibilityPair {
  friend1: string;
  friend2: string;
  similarity: number;
}

/**
 * Calculates complete group synergy parameters and similarity scores.
 */
export function analyzeGroupSynergy(
  friends: { name: string }[],
  preferences: { name: string; text: string }[]
): {
  matrix: Record<string, Record<string, number>>;
  pairs: CompatibilityPair[];
  topWeights: { word: string; weight: number }[];
  averageSynergy: number;
  strongestPair?: CompatibilityPair;
} {
  const prefTexts = preferences.map(p => p.text);
  const { vocabulary, vectors, wordWeights } = computeTFIDF(prefTexts);

  // Map each person name to their respective vector index
  const personToVectorIndex: Record<string, number> = {};
  preferences.forEach((pref, index) => {
    personToVectorIndex[pref.name] = index;
  });

  const matrix: Record<string, Record<string, number>> = {};
  const pairs: CompatibilityPair[] = [];

  // Initialize empty grid structure
  friends.forEach(f => {
    matrix[f.name] = {};
    friends.forEach(f2 => {
      matrix[f.name][f2.name] = f.name === f2.name ? 1.0 : 0.0;
    });
  });

  // Populate similarity calculations where preferences exist
  for (let i = 0; i < friends.length; i++) {
    const name1 = friends[i].name;
    const idx1 = personToVectorIndex[name1];

    for (let j = 0; j < friends.length; j++) {
      const name2 = friends[j].name;
      const idx2 = personToVectorIndex[name2];

      if (idx1 !== undefined && idx2 !== undefined) {
        if (name1 === name2) {
          matrix[name1][name2] = 1.0;
        } else {
          const sim = calculateCosineSimilarity(vectors[idx1], vectors[idx2]);
          matrix[name1][name2] = sim;
          
          if (i < j) {
            pairs.push({ friend1: name1, friend2: name2, similarity: sim });
          }
        }
      }
    }
  }

  // Calculate averaged overall synergy index
  let totalSimSum = 0;
  let calculationCount = 0;
  pairs.forEach(p => {
    totalSimSum += p.similarity;
    calculationCount++;
  });

  const averageSynergy = calculationCount > 0 ? (totalSimSum / calculationCount) : 0;
  const strongestPair = pairs.length > 0
    ? [...pairs].sort((a, b) => b.similarity - a.similarity)[0]
    : undefined;

  return {
    matrix,
    pairs,
    topWeights: wordWeights.slice(0, 10),
    averageSynergy,
    strongestPair
  };
}
