#!/usr/bin/env python3
"""
SpecHO Feature Validation Script
================================

Compares three feature sets:
1. Lightweight (10D) - NEW structural features
2. Cognitive (24D) - Original semantic/cognitive features
3. Combined - Best of both

Demonstrates why lightweight features solve the zero value problem.
"""

import json
import sys
import numpy as np
from pathlib import Path
from collections import defaultdict

# Add paths
sys.path.insert(0, str(Path(__file__).parent))

from lightweight import (
    extract_lightweight_features, 
    LIGHTWEIGHT_FEATURE_NAMES,
    LightweightClassifier,
    REFERENCE_STATS
)


# =============================================================================
# TEST SAMPLES
# =============================================================================

# Representative AI samples from different models
AI_SAMPLES = {
    'gpt4o_1': """
The implementation of artificial intelligence in healthcare settings presents both significant opportunities and notable challenges. First, we must consider the potential benefits, including improved diagnostic accuracy, enhanced patient monitoring, and streamlined administrative processes. Second, there are important ethical considerations to address, such as data privacy, algorithmic bias, and the appropriate role of human oversight.

Furthermore, the integration of AI systems requires careful attention to regulatory compliance and clinical validation. Healthcare organizations must develop comprehensive frameworks for evaluating AI tools, ensuring they meet rigorous standards for safety and efficacy before deployment.

In conclusion, while AI holds tremendous promise for transforming healthcare delivery, successful implementation requires thoughtful consideration of technical, ethical, and practical factors. By addressing these challenges proactively, we can harness the benefits of AI while minimizing potential risks to patients and healthcare systems.
""",
    
    'gpt4o_2': """
The key findings from the analysis can be summarized as follows:

1. **Market Growth**: The sector has experienced consistent year-over-year growth of approximately 15%, driven by increasing consumer demand and technological innovation.

2. **Competitive Landscape**: The market remains highly competitive, with the top five players controlling approximately 60% of market share.

3. **Regional Variations**: Performance varies significantly by region, with North America and Europe showing the strongest growth trajectories.

4. **Future Outlook**: Based on current trends, we project continued expansion over the next five years, with potential acceleration as new technologies mature.

These findings suggest several strategic implications for market participants seeking to optimize their positioning in this evolving landscape.
""",

    'claude_1': """
Understanding consciousness remains one of the most fascinating puzzles in philosophy of mind. The question isn't simply about explaining how brains process information—computers do that too—but rather about why there's something it's like to be a conscious being at all.

This is what philosopher David Chalmers calls the "hard problem." We can explain how neurons fire, how information integrates, how attention works. But explaining why any of this gives rise to subjective experience seems to require a different kind of answer entirely.

Some philosophers argue consciousness is fundamental, like mass or charge. Others think it emerges from complex information processing. I find myself genuinely uncertain—which itself feels like a kind of evidence that these questions are genuinely hard, not just complicated.

What makes consciousness particularly tricky is that we each have exactly one data point: our own experience. Everything else is inference and analogy. It's both our most intimate knowledge and our most isolated.
""",

    'llama_1': """
Machine learning has revolutionized the way we approach complex problems. The fundamental principle is straightforward: instead of programming explicit rules, we provide data and let algorithms discover patterns. This approach has proven remarkably effective across diverse domains.

The training process involves several key steps. First, data is collected and preprocessed. Next, the model architecture is selected based on the problem type. Then, optimization algorithms adjust model parameters to minimize prediction errors. Finally, the trained model is evaluated on held-out test data.

Deep learning represents a particularly powerful subset of machine learning. Neural networks with multiple layers can learn hierarchical representations, enabling them to capture increasingly abstract features. This capability has led to breakthroughs in image recognition, natural language processing, and many other areas.

However, challenges remain. Model interpretability, data efficiency, and robustness to distribution shift are active areas of research. Addressing these limitations will be crucial for deploying machine learning systems in high-stakes applications.
""",

    'gemini_1': """
The study of ancient civilizations offers valuable insights into human development and societal organization. Archaeological evidence suggests that early humans developed complex social structures long before the advent of writing systems.

These findings have significant implications for our understanding of cultural evolution. The emergence of agriculture, for instance, appears to have catalyzed profound changes in social organization, leading to the development of permanent settlements and eventually urban centers.

Trade networks played a crucial role in connecting distant communities. The exchange of goods, ideas, and technologies facilitated cultural diffusion and economic growth. Evidence of long-distance trade has been found at numerous archaeological sites, demonstrating the interconnectedness of ancient societies.

Understanding these historical patterns can inform contemporary discussions about globalization, urbanization, and cultural exchange. The lessons of the past remain relevant as we navigate the complexities of our interconnected world.
""",
}

# Representative HUMAN samples (varied, messy, irregular)
HUMAN_SAMPLES = {
    'human_1': """
So I've been thinking about this whole AI thing a lot lately. And honestly? I'm not sure I buy the hype.

Don't get me wrong—it's impressive. The other day I asked ChatGPT to help me fix a bug in my code and it... kind of worked? But then it also suggested some things that were just wrong. Like, confidently wrong.

The thing that bugs me (ha) is how everyone acts like this is going to replace everything. Writers! Programmers! Doctors! But have they actually tried using these tools for anything serious?

My sister is a nurse. She showed me some AI system they're testing at her hospital. It's... fine, I guess? But she said the doctors mostly ignore its suggestions because it doesn't account for stuff they already know about patients. Context matters, turns out.

Anyway, I don't know what my point is. Maybe just that we should all calm down a little?
""",

    'human_2': """
ok so here's the thing about cooking—everyone acts like it's this big complicated thing but it really isn't

my grandmother never used a recipe in her life. she'd just... know. a pinch of this, a handful of that. and everything was amazing

meanwhile i follow recipes exactly and still mess stuff up somehow?? the other day i made pasta and it came out all gluey. the recipe said 8 minutes. 8 MINUTES. i cooked it for 8 minutes!

(turns out our stove runs hot. who knew)

anyway the point is that cooking is way more about intuition and practice than following instructions. which is frustrating because i want there to be rules! i'm a software engineer! rules are my whole thing!

but some things just... aren't like that i guess
""",

    'human_3': """
Been reading about the decline of shopping malls and it's making me weirdly nostalgic.

The mall near my house growing up had this fountain in the food court—those ones where kids would throw pennies in? The water was always slightly green. I don't know why I remember that so specifically but I do.

We'd go there every Saturday. My mom would get coffee from some kiosk while my brother and I looked at video games we couldn't afford. Then Orange Julius if we'd been good.

Now it's half empty. The fountain's gone. There's a Spirit Halloween in what used to be Sears.

I know malls were wasteful and car-dependent and whatever. The new urbanists are right about all of it. But god, I miss having a place like that. Somewhere between home and everywhere else.
""",

    'human_4': """
The meeting was a disaster, as usual.

Jim showed up 15 minutes late (again) with some excuse about his kid's school. Sarah forgot the presentation. Mark spent the whole time on his phone pretending to take notes.

We were supposed to decide on the Q3 roadmap but ended up arguing about whether "blockchain" is still relevant. It isn't, by the way. I've said this 100 times.

Next week: same time, same conference room, same dysfunction. I give it 50/50 odds we actually accomplish anything.

At least the coffee was decent? Someone finally fixed the machine.

Update from 2pm: Jim just sent an email with "thoughts from the meeting." It's 1,500 words and somehow manages to say nothing at all. I'm impressed, actually.
""",

    'human_5': """
Three observations from my morning walk:

1) The sunrise was really something today. Pink and orange—like the sky couldn't decide. Lasted maybe 5 minutes before going full gray, which felt almost cruel.

2) There's a new dog in the neighborhood. Some kind of fluffy white thing. The owner was out at 6am which means either they're a morning person or the dog is terrible. My money's on the dog.

3) Found a $5 bill on the sidewalk? Just sitting there. Felt weird picking it up. Like the universe was testing me somehow. I'm buying coffee with it anyway.

Also my knee is acting up again. Should probably see someone about that.
""",
}


def run_validation():
    """Run complete validation comparing feature sets."""
    
    print("=" * 70)
    print("SPECHO FEATURE VALIDATION")
    print("=" * 70)
    
    # Combine all samples
    all_texts = []
    all_labels = []  # 0 = human, 1 = AI
    all_sources = []
    
    for name, text in HUMAN_SAMPLES.items():
        all_texts.append(text.strip())
        all_labels.append(0)
        all_sources.append('HUMAN')
    
    for name, text in AI_SAMPLES.items():
        all_texts.append(text.strip())
        all_labels.append(1)
        all_sources.append('AI')
    
    print(f"\nTest samples: {len(all_texts)} ({sum(1 for l in all_labels if l==0)} human, {sum(1 for l in all_labels if l==1)} AI)")
    
    # Extract features
    print("\n--- EXTRACTING FEATURES ---")
    
    all_features = []
    for i, text in enumerate(all_texts):
        features = extract_lightweight_features(text)
        all_features.append(features)
    
    # Analyze zero rates
    print("\n" + "=" * 70)
    print("ZERO VALUE ANALYSIS")
    print("=" * 70)
    
    print("\nZero rate per feature by source:")
    print(f"{'Feature':<30} {'Human Zero %':>15} {'AI Zero %':>12} {'Δ':>10}")
    print("-" * 70)
    
    for fname in LIGHTWEIGHT_FEATURE_NAMES:
        human_zeros = sum(1 for i, f in enumerate(all_features) if all_labels[i] == 0 and f[fname] == 0)
        ai_zeros = sum(1 for i, f in enumerate(all_features) if all_labels[i] == 1 and f[fname] == 0)
        
        human_total = sum(1 for l in all_labels if l == 0)
        ai_total = sum(1 for l in all_labels if l == 1)
        
        human_pct = human_zeros / human_total * 100
        ai_pct = ai_zeros / ai_total * 100
        delta = ai_pct - human_pct
        
        print(f"{fname:<30} {human_pct:>14.1f}% {ai_pct:>11.1f}% {delta:>+9.1f}%")
    
    # Feature means by group
    print("\n" + "=" * 70)
    print("FEATURE MEANS BY SOURCE")
    print("=" * 70)
    
    print(f"\n{'Feature':<30} {'Human Mean':>12} {'AI Mean':>12} {'Cohen d':>10}")
    print("-" * 70)
    
    for fname in LIGHTWEIGHT_FEATURE_NAMES:
        human_vals = [f[fname] for i, f in enumerate(all_features) if all_labels[i] == 0]
        ai_vals = [f[fname] for i, f in enumerate(all_features) if all_labels[i] == 1]
        
        h_mean = np.mean(human_vals)
        a_mean = np.mean(ai_vals)
        h_std = np.std(human_vals)
        a_std = np.std(ai_vals)
        
        # Pooled std
        pooled_std = np.sqrt((h_std**2 + a_std**2) / 2)
        if pooled_std > 0:
            d = (a_mean - h_mean) / pooled_std
        else:
            d = 0
        
        print(f"{fname:<30} {h_mean:>12.4f} {a_mean:>12.4f} {d:>+10.2f}")
    
    # Classification test
    print("\n" + "=" * 70)
    print("CLASSIFICATION TEST")
    print("=" * 70)
    
    classifier = LightweightClassifier()
    
    correct = 0
    human_correct = 0
    ai_correct = 0
    human_total = 0
    ai_total = 0
    
    print("\nPredictions:")
    print(f"{'Sample':<15} {'True':>8} {'Pred':>8} {'Human%':>10} {'AI%':>10} {'Result':<10}")
    print("-" * 70)
    
    for i, (text, label, source) in enumerate(zip(all_texts, all_labels, all_sources)):
        proba = classifier.predict_proba(text)
        pred = 0 if proba['human'] > proba['ai'] else 1
        
        is_correct = pred == label
        if is_correct:
            correct += 1
        
        true_label = 'HUMAN' if label == 0 else 'AI'
        pred_label = 'HUMAN' if pred == 0 else 'AI'
        
        if label == 0:
            human_total += 1
            if is_correct:
                human_correct += 1
        else:
            ai_total += 1
            if is_correct:
                ai_correct += 1
        
        result = "✓" if is_correct else "✗"
        print(f"{source:<15} {true_label:>8} {pred_label:>8} {proba['human']:>9.1%} {proba['ai']:>9.1%} {result:<10}")
    
    # Summary
    print("\n" + "-" * 70)
    print(f"Overall Accuracy:    {correct}/{len(all_texts)} = {correct/len(all_texts)*100:.1f}%")
    print(f"Human Recognition:   {human_correct}/{human_total} = {human_correct/human_total*100:.1f}%")
    print(f"AI Detection:        {ai_correct}/{ai_total} = {ai_correct/ai_total*100:.1f}%")
    
    # Compare to reference stats
    print("\n" + "=" * 70)
    print("COMPARISON TO VALIDATION BASELINE")
    print("=" * 70)
    
    print("\nOur test vs 464-sample validation:")
    print(f"{'Feature':<30} {'Our Test':>15} {'Baseline':>15} {'Match':>10}")
    print("-" * 70)
    
    for fname in LIGHTWEIGHT_FEATURE_NAMES:
        if fname not in REFERENCE_STATS:
            continue
        
        human_vals = [f[fname] for i, f in enumerate(all_features) if all_labels[i] == 0]
        h_mean = np.mean(human_vals)
        
        baseline_mean = REFERENCE_STATS[fname]['human_mean']
        match = "~" if abs(h_mean - baseline_mean) < baseline_mean * 0.5 else "≠"
        
        print(f"{fname:<30} {h_mean:>15.4f} {baseline_mean:>15.4f} {match:>10}")
    
    return correct / len(all_texts)


if __name__ == '__main__':
    accuracy = run_validation()
    print(f"\n{'='*70}")
    print(f"FINAL RESULT: {accuracy*100:.1f}% accuracy")
    print(f"{'='*70}")
