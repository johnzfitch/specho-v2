/**
 * SpecHO Client - Tier 1 Feature Extraction
 * 
 * Runs entirely in browser, no dependencies except pako (gzip)
 * Computes lightweight features in <50ms
 */

const Specho = (function() {
    'use strict';
    
    // ========================================
    // TIER 1 FEATURES (instant, no GPU)
    // ========================================
    
    /**
     * Compression ratio - lower = more AI-like
     * Uses pako for gzip compression
     */
    function compressionRatio(text) {
        if (!text || !window.pako) return 0;
        
        const encoded = new TextEncoder().encode(text);
        if (encoded.length === 0) return 0;
        
        try {
            const compressed = pako.gzip(encoded, { level: 9 });
            return compressed.length / encoded.length;
        } catch (e) {
            return 0;
        }
    }
    
    /**
     * Hapax rate - proportion of words appearing exactly once
     * Higher = more human-like
     */
    function hapaxRate(text) {
        const words = (text.toLowerCase().match(/\b[a-z]+\b/g) || []);
        if (words.length === 0) return 0;
        
        const counts = {};
        for (const word of words) {
            counts[word] = (counts[word] || 0) + 1;
        }
        
        let hapax = 0;
        for (const word in counts) {
            if (counts[word] === 1) hapax++;
        }
        
        return hapax / words.length;
    }
    
    /**
     * Sentence initial entropy - variety in sentence starters
     * Higher = more human-like
     */
    function sentenceInitialEntropy(text) {
        const sentences = text.split(/[.!?]+/);
        const firstWords = [];
        
        for (const sent of sentences) {
            const words = sent.trim().split(/\s+/);
            if (words[0]) {
                firstWords.push(words[0].toLowerCase());
            }
        }
        
        if (firstWords.length < 2) return 0;
        
        // Count frequencies
        const counts = {};
        for (const word of firstWords) {
            counts[word] = (counts[word] || 0) + 1;
        }
        
        // Calculate entropy
        const total = firstWords.length;
        let entropy = 0;
        for (const word in counts) {
            const p = counts[word] / total;
            entropy -= p * Math.log2(p);
        }
        
        // Normalize by max possible entropy
        const maxEntropy = Math.log2(total);
        return maxEntropy > 0 ? entropy / maxEntropy : 0;
    }
    
    /**
     * Function word coefficient of variation
     * Lower = more AI-like
     */
    const FUNCTION_WORDS = [
        'the', 'of', 'and', 'to', 'a', 'in', 'that', 'is', 'was', 'for',
        'on', 'with', 'as', 'it', 'be', 'at', 'by', 'this', 'from', 'or'
    ];
    
    function functionWordVariance(text) {
        const words = (text.toLowerCase().match(/\b[a-z]+\b/g) || []);
        if (words.length === 0) return 0;
        
        const total = words.length;
        const freqs = FUNCTION_WORDS.map(fw => {
            let count = 0;
            for (const w of words) {
                if (w === fw) count++;
            }
            return count / total;
        });
        
        const mean = freqs.reduce((a, b) => a + b, 0) / freqs.length;
        if (mean === 0) return 0;
        
        const variance = freqs.reduce((sum, f) => sum + Math.pow(f - mean, 2), 0) / freqs.length;
        const std = Math.sqrt(variance);
        
        return std / mean; // Coefficient of variation
    }
    
    /**
     * Punctuation entropy
     * Higher = more AI-like (counterintuitively)
     */
    function punctuationEntropy(text) {
        const punct = (text.match(/[.,;:!?\-\(\)\[\]"'…—–]/g) || []);
        if (punct.length < 5) return 0;
        
        const counts = {};
        for (const p of punct) {
            counts[p] = (counts[p] || 0) + 1;
        }
        
        const total = punct.length;
        let entropy = 0;
        for (const p in counts) {
            const prob = counts[p] / total;
            entropy -= prob * Math.log2(prob);
        }
        
        return entropy;
    }
    
    /**
     * Paragraph length coefficient of variation
     * Lower = more AI-like
     */
    function paragraphLengthVariance(text) {
        let paragraphs = text.split(/\n\n+/).filter(p => p.trim());
        
        if (paragraphs.length < 2) {
            paragraphs = text.split(/\n/).filter(p => p.trim());
        }
        
        if (paragraphs.length < 2) return 0;
        
        const lengths = paragraphs.map(p => p.split(/\s+/).length);
        const mean = lengths.reduce((a, b) => a + b, 0) / lengths.length;
        
        if (mean === 0) return 0;
        
        const variance = lengths.reduce((sum, l) => sum + Math.pow(l - mean, 2), 0) / lengths.length;
        return Math.sqrt(variance) / mean;
    }
    
    /**
     * Sentence length coefficient of variation
     * Lower = more AI-like (BEST FEATURE, d=-1.47)
     */
    function sentenceLengthVariance(text) {
        const sentences = text.split(/[.!?]+/).map(s => s.trim()).filter(s => s);
        
        if (sentences.length < 2) return 0;
        
        const lengths = sentences.map(s => s.split(/\s+/).length);
        const mean = lengths.reduce((a, b) => a + b, 0) / lengths.length;
        
        if (mean === 0) return 0;
        
        const variance = lengths.reduce((sum, l) => sum + Math.pow(l - mean, 2), 0) / lengths.length;
        return Math.sqrt(variance) / mean;
    }
    
    /**
     * List marker rate - proportion of lines that are list items
     * Higher = more AI-like
     */
    function listMarkerRate(text) {
        const lines = text.split(/\n/);
        if (lines.length === 0) return 0;
        
        let listMarkers = 0;
        for (const line of lines) {
            const trimmed = line.trim();
            // Bullet points: -, *, •
            if (/^[\-\*\•]/.test(trimmed)) {
                listMarkers++;
            }
            // Numbered lists: 1. or 1)
            else if (/^\d+[\.\)]/.test(trimmed)) {
                listMarkers++;
            }
        }
        
        return listMarkers / lines.length;
    }
    
    /**
     * Type-token ratio (lexical diversity)
     * Lower = more AI-like
     */
    function typeTokenRatio(text) {
        const words = (text.toLowerCase().match(/\b[a-z]+\b/g) || []);
        if (words.length === 0) return 0;
        
        const uniqueWords = new Set(words);
        return uniqueWords.size / words.length;
    }
    
    /**
     * Em-dash rate per 100 words
     * Most AI avoids em-dashes (except GPT-4o)
     */
    function emDashRate(text) {
        const words = text.split(/\s+/).length;
        if (words === 0) return 0;
        
        const emDashes = (text.match(/[—–]|--/g) || []).length;
        return (emDashes / words) * 100;
    }
    
    // ========================================
    // PUBLIC API
    // ========================================
    
    /**
     * Extract all Tier 1 (lightweight) features
     * @param {string} text - Input text
     * @returns {Object} Feature values
     */
    function extractLightweight(text) {
        if (!text || typeof text !== 'string') {
            return {};
        }
        
        const start = performance.now();
        
        const features = {
            compression_ratio: compressionRatio(text),
            hapax_rate: hapaxRate(text),
            sentence_initial_entropy: sentenceInitialEntropy(text),
            function_word_cv: functionWordVariance(text),
            punctuation_entropy: punctuationEntropy(text),
            paragraph_length_cv: paragraphLengthVariance(text),
            sentence_length_cv: sentenceLengthVariance(text),
            list_marker_rate: listMarkerRate(text),
            type_token_ratio: typeTokenRatio(text),
            em_dash_rate: emDashRate(text),
        };
        
        features._compute_time_ms = performance.now() - start;
        
        return features;
    }
    
    /**
     * Quick heuristic score from Tier 1 features
     * @param {Object} features - Feature values from extractLightweight
     * @returns {Object} {label, confidence, human_prob, ai_prob}
     */
    function scoreHeuristic(features) {
        const referenceStats = {
            compression_ratio: { human: 0.564, ai: 0.473, dir: -1, weight: 0.188 },
            sentence_length_cv: { human: 0.822, ai: 0.422, dir: -1, weight: 0.267 },
            paragraph_length_cv: { human: 0.847, ai: 0.493, dir: -1, weight: 0.119 },
            list_marker_rate: { human: 0.010, ai: 0.107, dir: 1, weight: 0.110 },
            type_token_ratio: { human: 0.596, ai: 0.481, dir: -1, weight: 0.085 },
            punctuation_entropy: { human: 1.590, ai: 1.982, dir: 1, weight: 0.068 },
            hapax_rate: { human: 0.417, ai: 0.355, dir: -1, weight: 0.049 },
            sentence_initial_entropy: { human: 0.860, ai: 0.777, dir: -1, weight: 0.059 },
            function_word_cv: { human: 1.420, ai: 1.217, dir: -1, weight: 0.027 },
            em_dash_rate: { human: 0.283, ai: 0.091, dir: -1, weight: 0.029 },
        };
        
        let aiScore = 0;
        let totalWeight = 0;
        const featureSignals = {};
        
        for (const [name, value] of Object.entries(features)) {
            if (name.startsWith('_')) continue;
            const ref = referenceStats[name];
            if (!ref) continue;
            
            const { human, ai, dir, weight } = ref;
            let signal;
            
            if (Math.abs(ai - human) > 0.001) {
                if (dir < 0) {
                    signal = (human - value) / (human - ai);
                } else {
                    signal = (value - human) / (ai - human);
                }
                signal = Math.max(0, Math.min(1, signal));
            } else {
                signal = 0.5;
            }
            
            featureSignals[name] = signal;
            aiScore += signal * weight;
            totalWeight += weight;
        }
        
        const aiProb = totalWeight > 0 ? aiScore / totalWeight : 0.5;
        
        let label;
        if (aiProb > 0.6) {
            label = 'AI';
        } else if (aiProb < 0.4) {
            label = 'HUMAN';
        } else {
            label = 'UNCERTAIN';
        }
        
        return {
            label,
            confidence: Math.abs(aiProb - 0.5) * 2,
            human_prob: 1 - aiProb,
            ai_prob: aiProb,
            feature_signals: featureSignals,
        };
    }
    
    /**
     * Full client-side analysis (Tier 1 only)
     * @param {string} text - Input text
     * @returns {Object} Complete analysis result
     */
    function analyze(text) {
        const features = extractLightweight(text);
        const score = scoreHeuristic(features);
        
        return {
            ...score,
            features,
            tier: 1,
            compute_mode: 'client',
        };
    }
    
    // ========================================
    // WEBGPU TIER 2 (optional, loaded async)
    // ========================================
    
    let ortSession = null;
    let webgpuReady = false;
    
    /**
     * Initialize WebGPU + ONNX Runtime for Tier 2 features
     * Loads sentence-transformers model for embeddings
     */
    async function initWebGPU() {
        if (!('gpu' in navigator)) {
            console.log('WebGPU not available');
            return false;
        }
        
        try {
            // Check for ONNX Runtime Web
            if (typeof ort === 'undefined') {
                // Load ONNX Runtime dynamically
                await loadScript('https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.0/dist/ort.webgpu.min.js');
            }
            
            // Configure for WebGPU
            ort.env.wasm.numThreads = 4;
            
            // TODO: Load sentence embedding model
            // ortSession = await ort.InferenceSession.create('/static/models/all-MiniLM-L6-v2.onnx', {
            //     executionProviders: ['webgpu', 'wasm']
            // });
            
            webgpuReady = true;
            console.log('WebGPU initialized');
            return true;
        } catch (e) {
            console.error('WebGPU init failed:', e);
            return false;
        }
    }
    
    function loadScript(src) {
        return new Promise((resolve, reject) => {
            const script = document.createElement('script');
            script.src = src;
            script.onload = resolve;
            script.onerror = reject;
            document.head.appendChild(script);
        });
    }
    
    /**
     * Extract Tier 2 features using WebGPU
     * Requires initWebGPU() to be called first
     */
    async function extractTier2(text) {
        if (!webgpuReady || !ortSession) {
            throw new Error('WebGPU not initialized');
        }
        
        // TODO: Implement sentence embeddings + tortuosity
        // This is where the expensive compute happens
        
        return {
            path_tortuosity: 0,
            semantic_mean: 0,
            phonetic_mean: 0,
            // ... other Tier 2 features
        };
    }
    
    // Public API
    return {
        extractLightweight,
        scoreHeuristic,
        analyze,
        initWebGPU,
        extractTier2,
        isWebGPUReady: () => webgpuReady,
        
        // Feature reference (for UI)
        FEATURE_INFO: {
            compression_ratio: { name: 'Compression Ratio', tier: 1, aiDir: 'lower' },
            hapax_rate: { name: 'Hapax Rate', tier: 1, aiDir: 'lower' },
            sentence_initial_entropy: { name: 'Sentence Initial Entropy', tier: 1, aiDir: 'lower' },
            function_word_cv: { name: 'Function Word CV', tier: 1, aiDir: 'lower' },
            punctuation_entropy: { name: 'Punctuation Entropy', tier: 1, aiDir: 'higher' },
            paragraph_length_cv: { name: 'Paragraph Length CV', tier: 1, aiDir: 'lower' },
            sentence_length_cv: { name: 'Sentence Length CV', tier: 1, aiDir: 'lower' },
            list_marker_rate: { name: 'List Marker Rate', tier: 1, aiDir: 'higher' },
            type_token_ratio: { name: 'Type-Token Ratio', tier: 1, aiDir: 'lower' },
            em_dash_rate: { name: 'Em-Dash Rate', tier: 1, aiDir: 'lower' },
            path_tortuosity: { name: 'Semantic Tortuosity', tier: 2, aiDir: 'higher' },
            semantic_mean: { name: 'Semantic Similarity', tier: 2, aiDir: 'higher' },
            phonetic_mean: { name: 'Phonetic Echo', tier: 2, aiDir: 'higher' },
        },
    };
})();

// Export for Node.js (testing)
if (typeof module !== 'undefined' && module.exports) {
    module.exports = Specho;
}
