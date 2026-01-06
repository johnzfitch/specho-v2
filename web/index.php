<?php
/**
 * definitelynot.ai - AI Text Fingerprinting
 * 
 * Stack: FrankenPHP + HTMX + Vanilla JS + WebGPU (optional)
 * 
 * Run with: frankenphp run --config Caddyfile
 */

declare(strict_types=1);

// Configuration
define('PYTHON_PATH', '/usr/bin/python3');
define('SPECHO_SCRIPT', __DIR__ . '/api/analyze.py');
define('MAX_TEXT_LENGTH', 50000);
define('MIN_TEXT_LENGTH', 100);

// Simple router
$path = parse_url($_SERVER['REQUEST_URI'], PHP_URL_PATH);
$method = $_SERVER['REQUEST_METHOD'];

match(true) {
    $path === '/' => render_home(),
    $path === '/api/analyze' && $method === 'POST' => handle_analyze(),
    $path === '/api/health' => handle_health(),
    str_starts_with($path, '/static/') => serve_static($path),
    default => http_response_code(404) && exit('Not found'),
};

function render_home(): void {
    header('Content-Type: text/html; charset=utf-8');
    ?>
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>definitelynot.ai - AI Text Fingerprinting</title>
    <script src="https://unpkg.com/htmx.org@1.9.10"></script>
    <script src="https://unpkg.com/pako@2.1.0/dist/pako.min.js"></script>
    <style>
        :root {
            --bg: #0a0a0a;
            --surface: #141414;
            --surface-2: #1e1e1e;
            --border: #2a2a2a;
            --text: #e0e0e0;
            --text-dim: #808080;
            --accent: #00ff88;
            --human: #00ff88;
            --ai: #ff4444;
            --uncertain: #ffaa00;
        }
        
        * { box-sizing: border-box; margin: 0; padding: 0; }
        
        body {
            font-family: 'SF Mono', 'Monaco', 'Inconsolata', monospace;
            background: var(--bg);
            color: var(--text);
            line-height: 1.6;
            min-height: 100vh;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 2rem;
        }
        
        header {
            border-bottom: 1px solid var(--border);
            padding-bottom: 1rem;
            margin-bottom: 2rem;
        }
        
        h1 {
            font-size: 1.5rem;
            font-weight: 400;
            color: var(--accent);
        }
        
        h1 span { color: var(--text-dim); }
        
        .subtitle {
            color: var(--text-dim);
            font-size: 0.875rem;
            margin-top: 0.5rem;
        }
        
        .input-section {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 1.5rem;
            margin-bottom: 2rem;
        }
        
        textarea {
            width: 100%;
            min-height: 200px;
            background: var(--surface-2);
            border: 1px solid var(--border);
            border-radius: 4px;
            color: var(--text);
            font-family: inherit;
            font-size: 0.875rem;
            padding: 1rem;
            resize: vertical;
        }
        
        textarea:focus {
            outline: none;
            border-color: var(--accent);
        }
        
        .controls {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-top: 1rem;
        }
        
        .char-count {
            color: var(--text-dim);
            font-size: 0.75rem;
        }
        
        button {
            background: var(--accent);
            color: var(--bg);
            border: none;
            padding: 0.75rem 1.5rem;
            border-radius: 4px;
            font-family: inherit;
            font-size: 0.875rem;
            font-weight: 600;
            cursor: pointer;
            transition: opacity 0.2s;
        }
        
        button:hover { opacity: 0.9; }
        button:disabled { opacity: 0.5; cursor: not-allowed; }
        
        .compute-mode {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            font-size: 0.75rem;
            color: var(--text-dim);
        }
        
        .compute-mode .indicator {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: var(--accent);
        }
        
        .compute-mode.slow .indicator { background: var(--uncertain); }
        
        /* Results */
        #results {
            min-height: 100px;
        }
        
        .results-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1.5rem;
        }
        
        @media (max-width: 768px) {
            .results-grid { grid-template-columns: 1fr; }
        }
        
        .result-card {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 1.5rem;
        }
        
        .result-card h3 {
            font-size: 0.875rem;
            font-weight: 400;
            color: var(--text-dim);
            margin-bottom: 1rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        
        .verdict {
            font-size: 2rem;
            font-weight: 600;
            margin-bottom: 0.5rem;
        }
        
        .verdict.human { color: var(--human); }
        .verdict.ai { color: var(--ai); }
        .verdict.uncertain { color: var(--uncertain); }
        
        .confidence {
            font-size: 0.875rem;
            color: var(--text-dim);
        }
        
        .prob-bars {
            margin-top: 1rem;
        }
        
        .prob-bar {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            margin-bottom: 0.5rem;
        }
        
        .prob-bar label {
            width: 60px;
            font-size: 0.75rem;
            color: var(--text-dim);
        }
        
        .prob-bar .bar {
            flex: 1;
            height: 8px;
            background: var(--surface-2);
            border-radius: 4px;
            overflow: hidden;
        }
        
        .prob-bar .fill {
            height: 100%;
            transition: width 0.3s ease;
        }
        
        .prob-bar.human .fill { background: var(--human); }
        .prob-bar.ai .fill { background: var(--ai); }
        
        .prob-bar .value {
            width: 50px;
            text-align: right;
            font-size: 0.75rem;
        }
        
        /* Feature table */
        .feature-table {
            width: 100%;
            font-size: 0.75rem;
            border-collapse: collapse;
        }
        
        .feature-table th,
        .feature-table td {
            padding: 0.5rem;
            text-align: left;
            border-bottom: 1px solid var(--border);
        }
        
        .feature-table th {
            color: var(--text-dim);
            font-weight: 400;
        }
        
        .feature-table .signal {
            width: 24px;
            text-align: center;
        }
        
        .feature-table .value {
            font-family: inherit;
            text-align: right;
        }
        
        .feature-table .bar-cell {
            width: 100px;
        }
        
        .feature-table .mini-bar {
            height: 4px;
            background: var(--surface-2);
            border-radius: 2px;
            overflow: hidden;
        }
        
        .feature-table .mini-bar .fill {
            height: 100%;
            background: var(--accent);
        }
        
        /* Model fingerprint */
        .model-probs {
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
        }
        
        .model-prob {
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        .model-prob .name {
            width: 100px;
            font-size: 0.75rem;
            color: var(--text-dim);
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }
        
        .model-prob .bar {
            flex: 1;
            height: 16px;
            background: var(--surface-2);
            border-radius: 2px;
            overflow: hidden;
        }
        
        .model-prob .fill {
            height: 100%;
            background: linear-gradient(90deg, var(--accent), #00cc6a);
            transition: width 0.3s ease;
        }
        
        .model-prob .pct {
            width: 45px;
            text-align: right;
            font-size: 0.75rem;
        }
        
        /* Loading state */
        .htmx-request .analyze-btn {
            opacity: 0.5;
            pointer-events: none;
        }
        
        .htmx-request .analyze-btn::after {
            content: '...';
        }
        
        .loading {
            text-align: center;
            padding: 2rem;
            color: var(--text-dim);
        }
        
        .loading::after {
            content: '';
            animation: dots 1.5s infinite;
        }
        
        @keyframes dots {
            0%, 20% { content: '.'; }
            40% { content: '..'; }
            60%, 100% { content: '...'; }
        }
        
        /* Tier indicator */
        .tier-badge {
            display: inline-block;
            font-size: 0.625rem;
            padding: 0.125rem 0.375rem;
            border-radius: 2px;
            background: var(--surface-2);
            color: var(--text-dim);
            margin-left: 0.5rem;
        }
        
        .tier-badge.t1 { border-left: 2px solid var(--accent); }
        .tier-badge.t2 { border-left: 2px solid var(--uncertain); }
        
        footer {
            margin-top: 4rem;
            padding-top: 2rem;
            border-top: 1px solid var(--border);
            color: var(--text-dim);
            font-size: 0.75rem;
            text-align: center;
        }
        
        footer a {
            color: var(--accent);
            text-decoration: none;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>definitelynot<span>.ai</span></h1>
            <p class="subtitle">AI Text Fingerprinting • Model Identification • 45-Dimensional Analysis</p>
        </header>
        
        <main>
            <div class="input-section">
                <form id="analyze-form" hx-post="/api/analyze" hx-target="#results" hx-indicator=".analyze-btn">
                    <textarea 
                        name="text" 
                        id="text-input"
                        placeholder="Paste text to analyze (minimum 100 characters)..."
                        oninput="updateCharCount()"
                    ></textarea>
                    
                    <div class="controls">
                        <div>
                            <span class="char-count" id="char-count">0 characters</span>
                            <div class="compute-mode" id="compute-mode">
                                <span class="indicator"></span>
                                <span id="mode-label">Checking WebGPU...</span>
                            </div>
                        </div>
                        <button type="submit" class="analyze-btn" id="analyze-btn" disabled>
                            Analyze
                        </button>
                    </div>
                    
                    <!-- Hidden field for client-side computed features -->
                    <input type="hidden" name="client_features" id="client-features">
                    <input type="hidden" name="compute_mode" id="compute-mode-input">
                </form>
            </div>
            
            <div id="results"></div>
        </main>
        
        <footer>
            <p>
                Part of the <a href="https://aurora-protocol.org">AURORA Protocol</a> • 
                <a href="https://github.com/definitelynot-ai/specho">Source</a> • 
                Built with SpecHO v2
            </p>
        </footer>
    </div>
    
    <script src="/static/specho-client.js"></script>
    <script>
        // Initialize
        const textInput = document.getElementById('text-input');
        const charCount = document.getElementById('char-count');
        const analyzeBtn = document.getElementById('analyze-btn');
        const computeMode = document.getElementById('compute-mode');
        const modeLabel = document.getElementById('mode-label');
        const clientFeatures = document.getElementById('client-features');
        const computeModeInput = document.getElementById('compute-mode-input');
        
        let hasWebGPU = false;
        
        // Check WebGPU support
        async function checkWebGPU() {
            if ('gpu' in navigator) {
                try {
                    const adapter = await navigator.gpu.requestAdapter();
                    if (adapter) {
                        hasWebGPU = true;
                        modeLabel.textContent = 'WebGPU: Fast mode';
                        computeModeInput.value = 'webgpu';
                        return;
                    }
                } catch (e) {}
            }
            modeLabel.textContent = 'Server: Fallback mode';
            computeMode.classList.add('slow');
            computeModeInput.value = 'server';
        }
        
        checkWebGPU();
        
        function updateCharCount() {
            const len = textInput.value.length;
            charCount.textContent = `${len.toLocaleString()} characters`;
            analyzeBtn.disabled = len < <?= MIN_TEXT_LENGTH ?>;
        }
        
        // Compute Tier 1 features client-side before submit
        document.getElementById('analyze-form').addEventListener('htmx:beforeRequest', async (e) => {
            const text = textInput.value;
            
            // Compute lightweight features
            const features = Specho.extractLightweight(text);
            clientFeatures.value = JSON.stringify(features);
        });
    </script>
</body>
</html>
    <?php
}

function handle_analyze(): void {
    header('Content-Type: text/html; charset=utf-8');
    
    $text = $_POST['text'] ?? '';
    $clientFeatures = json_decode($_POST['client_features'] ?? '{}', true);
    $computeMode = $_POST['compute_mode'] ?? 'server';
    
    // Validate
    if (strlen($text) < MIN_TEXT_LENGTH) {
        echo '<div class="result-card"><p>Text too short. Minimum ' . MIN_TEXT_LENGTH . ' characters.</p></div>';
        return;
    }
    
    if (strlen($text) > MAX_TEXT_LENGTH) {
        $text = substr($text, 0, MAX_TEXT_LENGTH);
    }
    
    // If we have client features, use them for Tier 1
    // Only call server for Tier 2 (embeddings) if needed
    
    if ($computeMode === 'server' || empty($clientFeatures)) {
        // Full server-side analysis
        $result = analyze_server($text);
    } else {
        // Hybrid: use client Tier 1, server Tier 2
        $result = analyze_hybrid($text, $clientFeatures);
    }
    
    render_results($result);
}

function analyze_server(string $text): array {
    // Call Python script for full analysis
    $input = json_encode(['text' => $text, 'mode' => 'full']);
    
    $descriptors = [
        0 => ['pipe', 'r'],
        1 => ['pipe', 'w'],
        2 => ['pipe', 'w'],
    ];
    
    $process = proc_open(
        [PYTHON_PATH, SPECHO_SCRIPT],
        $descriptors,
        $pipes,
        __DIR__
    );
    
    if (!is_resource($process)) {
        return ['error' => 'Failed to start analysis process'];
    }
    
    fwrite($pipes[0], $input);
    fclose($pipes[0]);
    
    $output = stream_get_contents($pipes[1]);
    fclose($pipes[1]);
    fclose($pipes[2]);
    
    proc_close($process);
    
    return json_decode($output, true) ?? ['error' => 'Analysis failed'];
}

function analyze_hybrid(string $text, array $clientFeatures): array {
    // Use client features for Tier 1
    // Call server only for Tier 2 (embeddings)
    $input = json_encode([
        'text' => $text,
        'mode' => 'tier2_only',
        'client_features' => $clientFeatures,
    ]);
    
    $descriptors = [
        0 => ['pipe', 'r'],
        1 => ['pipe', 'w'],
        2 => ['pipe', 'w'],
    ];
    
    $process = proc_open(
        [PYTHON_PATH, SPECHO_SCRIPT],
        $descriptors,
        $pipes,
        __DIR__
    );
    
    if (!is_resource($process)) {
        // Fallback to client-only results
        return score_from_features($clientFeatures);
    }
    
    fwrite($pipes[0], $input);
    fclose($pipes[0]);
    
    $output = stream_get_contents($pipes[1]);
    fclose($pipes[1]);
    fclose($pipes[2]);
    
    proc_close($process);
    
    $serverResult = json_decode($output, true);
    if (!$serverResult) {
        return score_from_features($clientFeatures);
    }
    
    // Merge client and server features
    return array_merge($serverResult, ['client_features' => $clientFeatures]);
}

function score_from_features(array $features): array {
    // Heuristic scoring from Tier 1 features only
    $referenceStats = [
        'compression_ratio' => ['human' => 0.564, 'ai' => 0.473, 'dir' => -1, 'weight' => 0.188],
        'sentence_length_cv' => ['human' => 0.822, 'ai' => 0.422, 'dir' => -1, 'weight' => 0.267],
        'paragraph_length_cv' => ['human' => 0.847, 'ai' => 0.493, 'dir' => -1, 'weight' => 0.119],
        'list_marker_rate' => ['human' => 0.010, 'ai' => 0.107, 'dir' => 1, 'weight' => 0.110],
        'type_token_ratio' => ['human' => 0.596, 'ai' => 0.481, 'dir' => -1, 'weight' => 0.085],
        'punctuation_entropy' => ['human' => 1.590, 'ai' => 1.982, 'dir' => 1, 'weight' => 0.068],
        'hapax_rate' => ['human' => 0.417, 'ai' => 0.355, 'dir' => -1, 'weight' => 0.049],
        'sentence_initial_entropy' => ['human' => 0.860, 'ai' => 0.777, 'dir' => -1, 'weight' => 0.059],
    ];
    
    $aiScore = 0;
    $totalWeight = 0;
    $featureSignals = [];
    
    foreach ($features as $name => $value) {
        if (!isset($referenceStats[$name])) continue;
        
        $ref = $referenceStats[$name];
        $humanMean = $ref['human'];
        $aiMean = $ref['ai'];
        $dir = $ref['dir'];
        $weight = $ref['weight'];
        
        if (abs($aiMean - $humanMean) > 0.001) {
            if ($dir < 0) {
                $signal = ($humanMean - $value) / ($humanMean - $aiMean);
            } else {
                $signal = ($value - $humanMean) / ($aiMean - $humanMean);
            }
            $signal = max(0, min(1, $signal));
        } else {
            $signal = 0.5;
        }
        
        $featureSignals[$name] = $signal;
        $aiScore += $signal * $weight;
        $totalWeight += $weight;
    }
    
    $aiProb = $totalWeight > 0 ? $aiScore / $totalWeight : 0.5;
    
    if ($aiProb > 0.6) {
        $label = 'AI';
    } elseif ($aiProb < 0.4) {
        $label = 'HUMAN';
    } else {
        $label = 'UNCERTAIN';
    }
    
    return [
        'label' => $label,
        'confidence' => abs($aiProb - 0.5) * 2,
        'human_prob' => 1 - $aiProb,
        'ai_prob' => $aiProb,
        'features' => $features,
        'feature_signals' => $featureSignals,
        'tier' => 1,
    ];
}

function render_results(array $result): void {
    if (isset($result['error'])) {
        echo '<div class="result-card"><p style="color: var(--ai);">Error: ' . htmlspecialchars($result['error']) . '</p></div>';
        return;
    }
    
    $label = $result['label'] ?? 'UNKNOWN';
    $confidence = ($result['confidence'] ?? 0) * 100;
    $humanProb = ($result['human_prob'] ?? 0.5) * 100;
    $aiProb = ($result['ai_prob'] ?? 0.5) * 100;
    $features = $result['features'] ?? [];
    $featureSignals = $result['feature_signals'] ?? [];
    $tier = $result['tier'] ?? 2;
    $modelProbs = $result['model_probabilities'] ?? [];
    
    $labelClass = strtolower($label);
    ?>
    <div class="results-grid">
        <div class="result-card">
            <h3>Verdict</h3>
            <div class="verdict <?= $labelClass ?>"><?= htmlspecialchars($label) ?></div>
            <div class="confidence"><?= number_format($confidence, 1) ?>% confidence</div>
            
            <div class="prob-bars">
                <div class="prob-bar human">
                    <label>Human</label>
                    <div class="bar"><div class="fill" style="width: <?= $humanProb ?>%"></div></div>
                    <span class="value"><?= number_format($humanProb, 1) ?>%</span>
                </div>
                <div class="prob-bar ai">
                    <label>AI</label>
                    <div class="bar"><div class="fill" style="width: <?= $aiProb ?>%"></div></div>
                    <span class="value"><?= number_format($aiProb, 1) ?>%</span>
                </div>
            </div>
        </div>
        
        <?php if (!empty($modelProbs)): ?>
        <div class="result-card">
            <h3>Model Fingerprint</h3>
            <div class="model-probs">
                <?php 
                arsort($modelProbs);
                foreach ($modelProbs as $model => $prob): 
                    $pct = $prob * 100;
                    $shortName = match(true) {
                        str_contains($model, 'GPT') => 'GPT-4o',
                        str_contains($model, 'gemma') => 'Gemma-2',
                        str_contains($model, 'mistral') => 'Mistral-7B',
                        str_contains($model, 'qwen') => 'Qwen-2',
                        str_contains($model, 'llama') => 'Llama-8B',
                        str_contains($model, 'yi') => 'Yi-Large',
                        $model === 'HUMAN' => 'Human',
                        default => substr($model, 0, 12),
                    };
                ?>
                <div class="model-prob">
                    <span class="name"><?= htmlspecialchars($shortName) ?></span>
                    <div class="bar"><div class="fill" style="width: <?= $pct ?>%"></div></div>
                    <span class="pct"><?= number_format($pct, 1) ?>%</span>
                </div>
                <?php endforeach; ?>
            </div>
        </div>
        <?php endif; ?>
        
        <div class="result-card" style="grid-column: 1 / -1;">
            <h3>Feature Analysis <span class="tier-badge t<?= $tier ?>">Tier <?= $tier ?></span></h3>
            <table class="feature-table">
                <thead>
                    <tr>
                        <th class="signal"></th>
                        <th>Feature</th>
                        <th class="value">Value</th>
                        <th class="bar-cell">AI Signal</th>
                    </tr>
                </thead>
                <tbody>
                    <?php 
                    $featureLabels = [
                        'sentence_length_cv' => 'Sentence Length Variance',
                        'compression_ratio' => 'Compression Ratio',
                        'paragraph_length_cv' => 'Paragraph Length Variance',
                        'list_marker_rate' => 'List Marker Rate',
                        'type_token_ratio' => 'Type-Token Ratio',
                        'punctuation_entropy' => 'Punctuation Entropy',
                        'hapax_rate' => 'Hapax Rate',
                        'sentence_initial_entropy' => 'Sentence Initial Entropy',
                        'function_word_cv' => 'Function Word CV',
                        'em_dash_rate' => 'Em-Dash Rate',
                        'path_tortuosity' => 'Semantic Tortuosity',
                        'phonetic_mean' => 'Phonetic Echo',
                        'semantic_mean' => 'Semantic Similarity',
                    ];
                    
                    foreach ($features as $name => $value): 
                        $signal = $featureSignals[$name] ?? 0.5;
                        $signalPct = $signal * 100;
                        $icon = $signal > 0.6 ? '🤖' : ($signal < 0.4 ? '👤' : '❓');
                        $displayName = $featureLabels[$name] ?? ucfirst(str_replace('_', ' ', $name));
                    ?>
                    <tr>
                        <td class="signal"><?= $icon ?></td>
                        <td><?= htmlspecialchars($displayName) ?></td>
                        <td class="value"><?= number_format($value, 4) ?></td>
                        <td class="bar-cell">
                            <div class="mini-bar">
                                <div class="fill" style="width: <?= $signalPct ?>%; background: <?= $signal > 0.6 ? 'var(--ai)' : ($signal < 0.4 ? 'var(--human)' : 'var(--uncertain)') ?>"></div>
                            </div>
                        </td>
                    </tr>
                    <?php endforeach; ?>
                </tbody>
            </table>
        </div>
    </div>
    <?php
}

function handle_health(): void {
    header('Content-Type: application/json');
    echo json_encode(['status' => 'ok', 'timestamp' => time()]);
}

function serve_static(string $path): void {
    $file = __DIR__ . $path;
    if (!file_exists($file)) {
        http_response_code(404);
        exit('Not found');
    }
    
    $ext = pathinfo($file, PATHINFO_EXTENSION);
    $mimeTypes = [
        'js' => 'application/javascript',
        'css' => 'text/css',
        'wasm' => 'application/wasm',
        'json' => 'application/json',
    ];
    
    header('Content-Type: ' . ($mimeTypes[$ext] ?? 'application/octet-stream'));
    header('Cache-Control: public, max-age=31536000');
    readfile($file);
}
