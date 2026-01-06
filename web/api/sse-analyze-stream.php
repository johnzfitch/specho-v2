<?php
/**
 * SSE Streaming Endpoint for Real-time Text Analysis
 * 
 * Streams feature extraction results in real-time as they're computed.
 * This provides a much better UX than waiting for all features.
 * 
 * Protocol:
 *   event: feature
 *   data: {"name": "sentence_length_cv", "value": 0.42, "signal": 0.8}
 *   
 *   event: verdict
 *   data: {"label": "AI", "confidence": 0.85, "human_prob": 0.15, "ai_prob": 0.85}
 *   
 *   event: model
 *   data: {"name": "GPT-4o", "probability": 0.45}
 *   
 *   event: complete
 *   data: {}
 */

declare(strict_types=1);

// Disable output buffering for real SSE streaming
if (ob_get_level()) ob_end_clean();

// SSE headers
header('Content-Type: text/event-stream');
header('Cache-Control: no-cache');
header('Connection: keep-alive');
header('X-Accel-Buffering: no'); // Disable nginx buffering

// Flush headers immediately
flush();

// Get input
$text = $_GET['text'] ?? '';

if (strlen($text) < 100) {
    sendEvent('error', ['message' => 'Text too short (minimum 100 characters)']);
    sendEvent('complete', []);
    exit;
}

// Limit text length
$text = substr($text, 0, 50000);

// Reference distributions for scoring
$referenceStats = [
    'compression_ratio' => ['human' => 0.564, 'ai' => 0.473, 'dir' => -1, 'weight' => 0.188],
    'sentence_length_cv' => ['human' => 0.822, 'ai' => 0.422, 'dir' => -1, 'weight' => 0.267],
    'paragraph_length_cv' => ['human' => 0.847, 'ai' => 0.493, 'dir' => -1, 'weight' => 0.119],
    'list_marker_rate' => ['human' => 0.010, 'ai' => 0.107, 'dir' => 1, 'weight' => 0.110],
    'type_token_ratio' => ['human' => 0.596, 'ai' => 0.481, 'dir' => -1, 'weight' => 0.085],
    'punctuation_entropy' => ['human' => 1.590, 'ai' => 1.982, 'dir' => 1, 'weight' => 0.068],
    'hapax_rate' => ['human' => 0.417, 'ai' => 0.355, 'dir' => -1, 'weight' => 0.049],
    'sentence_initial_entropy' => ['human' => 0.860, 'ai' => 0.777, 'dir' => -1, 'weight' => 0.059],
    'function_word_cv' => ['human' => 1.420, 'ai' => 1.217, 'dir' => -1, 'weight' => 0.027],
    'em_dash_rate' => ['human' => 0.283, 'ai' => 0.091, 'dir' => -1, 'weight' => 0.029],
];

$modelFingerprints = [
    'HUMAN' => ['sentence_length_cv' => 0.822, 'compression_ratio' => 0.564, 'list_marker_rate' => 0.010],
    'GPT-4o' => ['sentence_length_cv' => 0.507, 'compression_ratio' => 0.481, 'list_marker_rate' => 0.214],
    'Claude' => ['sentence_length_cv' => 0.580, 'compression_ratio' => 0.490, 'list_marker_rate' => 0.085],
    'Gemma-2' => ['sentence_length_cv' => 0.462, 'compression_ratio' => 0.504, 'list_marker_rate' => 0.158],
    'Mistral' => ['sentence_length_cv' => 0.408, 'compression_ratio' => 0.462, 'list_marker_rate' => 0.022],
    'Llama-3' => ['sentence_length_cv' => 0.372, 'compression_ratio' => 0.462, 'list_marker_rate' => 0.126],
];

$featureDisplayNames = [
    'compression_ratio' => 'Compression',
    'sentence_length_cv' => 'Sentence Variance',
    'paragraph_length_cv' => 'Paragraph Variance',
    'list_marker_rate' => 'List Usage',
    'type_token_ratio' => 'Vocabulary Diversity',
    'punctuation_entropy' => 'Punctuation Entropy',
    'hapax_rate' => 'Unique Words',
    'sentence_initial_entropy' => 'Sentence Starters',
    'function_word_cv' => 'Function Words',
    'em_dash_rate' => 'Em-Dash Usage',
];

// ============================================
// Feature Extraction Functions
// ============================================

function compressionRatio(string $text): float {
    $encoded = $text;
    if (strlen($encoded) === 0) return 0.0;
    $compressed = gzcompress($encoded, 9);
    return strlen($compressed) / strlen($encoded);
}

function sentenceLengthVariance(string $text): float {
    $sentences = preg_split('/[.!?]+/', $text, -1, PREG_SPLIT_NO_EMPTY);
    $sentences = array_filter(array_map('trim', $sentences));
    
    if (count($sentences) < 2) return 0.0;
    
    $lengths = array_map(fn($s) => str_word_count($s), $sentences);
    $mean = array_sum($lengths) / count($lengths);
    
    if ($mean == 0) return 0.0;
    
    $variance = array_sum(array_map(fn($l) => pow($l - $mean, 2), $lengths)) / count($lengths);
    return sqrt($variance) / $mean;
}

function paragraphLengthVariance(string $text): float {
    $paragraphs = preg_split('/\n\n+/', $text, -1, PREG_SPLIT_NO_EMPTY);
    $paragraphs = array_filter(array_map('trim', $paragraphs));
    
    if (count($paragraphs) < 2) {
        $paragraphs = preg_split('/\n/', $text, -1, PREG_SPLIT_NO_EMPTY);
        $paragraphs = array_filter(array_map('trim', $paragraphs));
    }
    
    if (count($paragraphs) < 2) return 0.0;
    
    $lengths = array_map(fn($p) => str_word_count($p), $paragraphs);
    $mean = array_sum($lengths) / count($lengths);
    
    if ($mean == 0) return 0.0;
    
    $variance = array_sum(array_map(fn($l) => pow($l - $mean, 2), $lengths)) / count($lengths);
    return sqrt($variance) / $mean;
}

function listMarkerRate(string $text): float {
    $lines = explode("\n", $text);
    if (count($lines) === 0) return 0.0;
    
    $listCount = 0;
    foreach ($lines as $line) {
        $trimmed = trim($line);
        if (preg_match('/^[\-\*\•]/', $trimmed) || preg_match('/^\d+[\.\)]/', $trimmed)) {
            $listCount++;
        }
    }
    
    return $listCount / count($lines);
}

function typeTokenRatio(string $text): float {
    preg_match_all('/\b[a-z]+\b/i', strtolower($text), $matches);
    $words = $matches[0];
    
    if (count($words) === 0) return 0.0;
    
    return count(array_unique($words)) / count($words);
}

function punctuationEntropy(string $text): float {
    preg_match_all('/[.,;:!?\-\(\)\[\]"\'…—–]/', $text, $matches);
    $punct = $matches[0];
    
    if (count($punct) < 5) return 0.0;
    
    $counts = array_count_values($punct);
    $total = count($punct);
    
    $entropy = 0;
    foreach ($counts as $count) {
        $p = $count / $total;
        $entropy -= $p * log($p, 2);
    }
    
    return $entropy;
}

function hapaxRate(string $text): float {
    preg_match_all('/\b[a-z]+\b/i', strtolower($text), $matches);
    $words = $matches[0];
    
    if (count($words) === 0) return 0.0;
    
    $counts = array_count_values($words);
    $hapax = count(array_filter($counts, fn($c) => $c === 1));
    
    return $hapax / count($words);
}

function sentenceInitialEntropy(string $text): float {
    $sentences = preg_split('/[.!?]+/', $text, -1, PREG_SPLIT_NO_EMPTY);
    $firstWords = [];
    
    foreach ($sentences as $s) {
        $words = preg_split('/\s+/', trim($s));
        if (!empty($words[0])) {
            $firstWords[] = strtolower($words[0]);
        }
    }
    
    if (count($firstWords) < 2) return 0.0;
    
    $counts = array_count_values($firstWords);
    $total = count($firstWords);
    
    $entropy = 0;
    foreach ($counts as $count) {
        $p = $count / $total;
        $entropy -= $p * log($p, 2);
    }
    
    $maxEntropy = log($total, 2);
    return $maxEntropy > 0 ? $entropy / $maxEntropy : 0;
}

function functionWordVariance(string $text): float {
    $functionWords = ['the', 'of', 'and', 'to', 'a', 'in', 'that', 'is', 'was', 'for',
                      'on', 'with', 'as', 'it', 'be', 'at', 'by', 'this', 'from', 'or'];
    
    preg_match_all('/\b[a-z]+\b/i', strtolower($text), $matches);
    $words = $matches[0];
    
    if (count($words) === 0) return 0.0;
    
    $total = count($words);
    $wordCounts = array_count_values($words);
    
    $freqs = [];
    foreach ($functionWords as $fw) {
        $freqs[] = ($wordCounts[$fw] ?? 0) / $total;
    }
    
    $mean = array_sum($freqs) / count($freqs);
    if ($mean == 0) return 0.0;
    
    $variance = array_sum(array_map(fn($f) => pow($f - $mean, 2), $freqs)) / count($freqs);
    return sqrt($variance) / $mean;
}

function emDashRate(string $text): float {
    $words = str_word_count($text);
    if ($words === 0) return 0.0;
    
    $emDashes = substr_count($text, '—') + substr_count($text, '–') + substr_count($text, '--');
    return ($emDashes / $words) * 100;
}

function calculateSignal(float $value, array $ref): float {
    $humanMean = $ref['human'];
    $aiMean = $ref['ai'];
    $dir = $ref['dir'];
    
    if (abs($aiMean - $humanMean) < 0.001) return 0.5;
    
    if ($dir < 0) {
        $signal = ($humanMean - $value) / ($humanMean - $aiMean);
    } else {
        $signal = ($value - $humanMean) / ($aiMean - $humanMean);
    }
    
    return max(0, min(1, $signal));
}

function sendEvent(string $event, array $data): void {
    echo "event: {$event}\n";
    echo "data: " . json_encode($data) . "\n\n";
    flush();
}

// ============================================
// STREAM FEATURES ONE BY ONE
// ============================================

$featureExtractors = [
    'sentence_length_cv' => 'sentenceLengthVariance',
    'compression_ratio' => 'compressionRatio',
    'paragraph_length_cv' => 'paragraphLengthVariance',
    'list_marker_rate' => 'listMarkerRate',
    'type_token_ratio' => 'typeTokenRatio',
    'punctuation_entropy' => 'punctuationEntropy',
    'hapax_rate' => 'hapaxRate',
    'sentence_initial_entropy' => 'sentenceInitialEntropy',
    'function_word_cv' => 'functionWordVariance',
    'em_dash_rate' => 'emDashRate',
];

$features = [];
$aiScore = 0;
$totalWeight = 0;

foreach ($featureExtractors as $name => $func) {
    // Small delay for visual effect (remove in production for max speed)
    usleep(50000); // 50ms
    
    $value = $func($text);
    $features[$name] = $value;
    
    $ref = $referenceStats[$name];
    $signal = calculateSignal($value, $ref);
    
    $aiScore += $signal * $ref['weight'];
    $totalWeight += $ref['weight'];
    
    sendEvent('feature', [
        'name' => $featureDisplayNames[$name] ?? $name,
        'value' => $value,
        'signal' => $signal,
    ]);
}

// Calculate final verdict
$aiProb = $totalWeight > 0 ? $aiScore / $totalWeight : 0.5;

if ($aiProb > 0.6) {
    $label = 'AI';
} elseif ($aiProb < 0.4) {
    $label = 'HUMAN';
} else {
    $label = 'UNCERTAIN';
}

$confidence = abs($aiProb - 0.5) * 2;

sendEvent('verdict', [
    'label' => $label,
    'confidence' => $confidence,
    'human_prob' => 1 - $aiProb,
    'ai_prob' => $aiProb,
]);

// Model fingerprint matching
$modelProbs = [];
foreach ($modelFingerprints as $model => $fingerprint) {
    $distance = 0;
    $count = 0;
    
    foreach ($fingerprint as $feat => $refVal) {
        if (isset($features[$feat])) {
            $distance += abs($features[$feat] - $refVal);
            $count++;
        }
    }
    
    if ($count > 0) {
        $avgDistance = $distance / $count;
        $modelProbs[$model] = max(0, 1 - $avgDistance * 2);
    }
}

// Normalize
$total = array_sum($modelProbs);
if ($total > 0) {
    foreach ($modelProbs as $model => &$prob) {
        $prob = $prob / $total;
    }
}

// Sort by probability
arsort($modelProbs);

// Stream model probabilities
foreach ($modelProbs as $model => $prob) {
    usleep(30000); // 30ms delay for visual effect
    sendEvent('model', [
        'name' => $model,
        'probability' => $prob,
    ]);
}

// Complete
sendEvent('complete', []);
