"""
SpecHO Training & Evaluation Script
====================================

Train and evaluate the SpecHO v2 classifier on sample data.

Usage:
    python train_specho.py --data samples_500.json --output model.pkl
"""

import json
import argparse
import pickle
import numpy as np
from pathlib import Path
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix, accuracy_score, classification_report,
    precision_recall_fscore_support
)

from specho_features_v2 import (
    extract_lightweight_features_array,
    LIGHTWEIGHT_FEATURE_NAMES,
    SpecHOClassifier
)


def load_samples(path: str) -> dict:
    """Load samples from JSON file."""
    with open(path) as f:
        return json.load(f)


def build_feature_matrix(samples: list, models: list = None) -> tuple:
    """
    Build feature matrix from samples.
    
    Args:
        samples: List of sample dicts with 'human' and 'ai_variants'
        models: List of model names to include (None = all)
        
    Returns:
        X: Feature matrix (n_samples, n_features)
        y_binary: Binary labels (0=human, 1=AI)
        y_multi: Model name labels
        sample_ids: Sample IDs for each row
    """
    X = []
    y_binary = []
    y_multi = []
    sample_ids = []
    
    for sample in samples:
        sid = sample.get('id', 'unknown')
        
        # Human
        human_text = sample['human']
        X.append(extract_lightweight_features_array(human_text))
        y_binary.append(0)
        y_multi.append('HUMAN')
        sample_ids.append(sid)
        
        # AI variants
        for model, text in sample['ai_variants'].items():
            if models is not None and model not in models:
                continue
            X.append(extract_lightweight_features_array(text))
            y_binary.append(1)
            y_multi.append(model)
            sample_ids.append(sid)
    
    return np.array(X), np.array(y_binary), np.array(y_multi), sample_ids


def evaluate_binary(X: np.ndarray, y: np.ndarray, cv_folds: int = 5) -> dict:
    """
    Evaluate binary classification with cross-validation.
    
    Returns:
        Dict with accuracy, precision, recall, f1, confusion matrix, etc.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    y_pred = cross_val_predict(clf, X_scaled, y, cv=cv)
    
    cm = confusion_matrix(y, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    precision, recall, f1, _ = precision_recall_fscore_support(y, y_pred, average='binary')
    
    return {
        'accuracy': accuracy_score(y, y_pred),
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'true_positives': int(tp),
        'human_recognition_rate': tn / (tn + fp) if (tn + fp) > 0 else 0,
        'ai_detection_rate': tp / (tp + fn) if (tp + fn) > 0 else 0,
    }


def evaluate_multiclass(X: np.ndarray, y: np.ndarray, cv_folds: int = 5) -> dict:
    """
    Evaluate multi-class model identification.
    
    Returns:
        Dict with overall accuracy and per-class accuracy
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    y_pred = cross_val_predict(clf, X_scaled, y, cv=cv)
    
    # Per-class accuracy
    classes = sorted(set(y))
    per_class = {}
    for cls in classes:
        mask = y == cls
        per_class[cls] = accuracy_score(y[mask], y_pred[mask])
    
    return {
        'accuracy': accuracy_score(y, y_pred),
        'per_class_accuracy': per_class,
    }


def evaluate_pairwise(X: np.ndarray, y_multi: np.ndarray, cv_folds: int = 5) -> dict:
    """
    Evaluate pairwise HUMAN vs each AI model.
    
    Returns:
        Dict with accuracy for each model pair
    """
    models = sorted(set(y_multi) - {'HUMAN'})
    results = {}
    
    scaler = StandardScaler()
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    for model in models:
        mask = np.isin(y_multi, ['HUMAN', model])
        X_pair = X[mask]
        y_pair = (y_multi[mask] != 'HUMAN').astype(int)
        
        X_scaled = scaler.fit_transform(X_pair)
        y_pred = cross_val_predict(clf, X_scaled, y_pair, cv=cv)
        
        # Human recognition rate
        human_mask = y_multi[mask] == 'HUMAN'
        human_correct = (y_pred[human_mask] == 0).sum()
        human_total = human_mask.sum()
        
        results[model] = {
            'accuracy': accuracy_score(y_pair, y_pred),
            'human_recognition': human_correct / human_total if human_total > 0 else 0,
        }
    
    return results


def get_feature_importance(X: np.ndarray, y: np.ndarray) -> list:
    """
    Train classifier and get feature importances.
    
    Returns:
        List of (feature_name, importance) sorted by importance
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf.fit(X_scaled, y)
    
    importances = list(zip(LIGHTWEIGHT_FEATURE_NAMES, clf.feature_importances_))
    return sorted(importances, key=lambda x: -x[1])


def train_and_save(X: np.ndarray, y: np.ndarray, output_path: str):
    """Train classifier and save to disk."""
    classifier = SpecHOClassifier()
    classifier.fit(X, y)
    
    with open(output_path, 'wb') as f:
        pickle.dump(classifier, f)
    
    print(f"Model saved to {output_path}")
    return classifier


def main():
    parser = argparse.ArgumentParser(description='Train SpecHO v2 classifier')
    parser.add_argument('--data', type=str, required=True, help='Path to samples JSON')
    parser.add_argument('--output', type=str, default='specho_model.pkl', help='Output model path')
    parser.add_argument('--cv-folds', type=int, default=5, help='Cross-validation folds')
    args = parser.parse_args()
    
    print("=" * 70)
    print("SpecHO v2 Training & Evaluation")
    print("=" * 70)
    
    # Load data
    print(f"\nLoading data from {args.data}...")
    data = load_samples(args.data)
    samples = data['samples']
    print(f"Loaded {len(samples)} samples")
    
    # Build feature matrix
    print("\nExtracting features...")
    X, y_binary, y_multi, _ = build_feature_matrix(samples)
    print(f"Feature matrix: {X.shape}")
    print(f"Classes: {(y_binary == 0).sum()} human, {(y_binary == 1).sum()} AI")
    
    # Binary evaluation
    print("\n" + "=" * 70)
    print("BINARY CLASSIFICATION (Human vs AI)")
    print("=" * 70)
    
    binary_results = evaluate_binary(X, y_binary, args.cv_folds)
    print(f"\n{args.cv_folds}-Fold CV Results:")
    print(f"  Accuracy:              {binary_results['accuracy']:.3f}")
    print(f"  Precision:             {binary_results['precision']:.3f}")
    print(f"  Recall:                {binary_results['recall']:.3f}")
    print(f"  F1 Score:              {binary_results['f1']:.3f}")
    print(f"  Human Recognition:     {binary_results['human_recognition_rate']:.3f}")
    print(f"  AI Detection:          {binary_results['ai_detection_rate']:.3f}")
    
    print("\nConfusion Matrix:")
    print("                Pred_Human  Pred_AI")
    cm = binary_results['confusion_matrix']
    print(f"True_Human      {cm[0,0]:>10}  {cm[0,1]:>7}")
    print(f"True_AI         {cm[1,0]:>10}  {cm[1,1]:>7}")
    
    # Multi-class evaluation
    print("\n" + "=" * 70)
    print("MULTI-CLASS CLASSIFICATION (Model ID)")
    print("=" * 70)
    
    multi_results = evaluate_multiclass(X, y_multi, args.cv_folds)
    print(f"\n{args.cv_folds}-Fold CV Accuracy: {multi_results['accuracy']:.3f}")
    print("\nPer-Model Accuracy:")
    for model, acc in sorted(multi_results['per_class_accuracy'].items()):
        print(f"  {model[:35]:35}: {acc:.3f}")
    
    # Pairwise evaluation
    print("\n" + "=" * 70)
    print("PAIRWISE: HUMAN vs EACH MODEL")
    print("=" * 70)
    
    pairwise_results = evaluate_pairwise(X, y_multi, args.cv_folds)
    print(f"\n{'Model':<35} {'Accuracy':>10} {'Human Rec':>12}")
    print("-" * 60)
    for model, results in sorted(pairwise_results.items()):
        print(f"{model[:35]:<35} {results['accuracy']:>10.3f} {results['human_recognition']:>12.1%}")
    
    # Feature importance
    print("\n" + "=" * 70)
    print("FEATURE IMPORTANCE")
    print("=" * 70)
    
    importances = get_feature_importance(X, y_binary)
    print(f"\n{'Feature':<30} {'Importance':>12}")
    print("-" * 45)
    for name, imp in importances:
        bar = '█' * int(imp * 40)
        print(f"{name:<30} {imp:>12.3f} {bar}")
    
    # Train and save final model
    print("\n" + "=" * 70)
    print("TRAINING FINAL MODEL")
    print("=" * 70)
    
    classifier = train_and_save(X, y_binary, args.output)
    
    # Demo prediction
    print("\n" + "=" * 70)
    print("DEMO PREDICTIONS")
    print("=" * 70)
    
    demo_human = "I've been thinking about this problem and honestly it's weird but the solution came to me while making coffee yesterday."
    demo_ai = "The problem requires careful analysis. First, we examine the assumptions. Second, we evaluate the evidence. Third, we consider alternatives."
    
    print("\nHuman-like text:")
    result = classifier.predict_text(demo_human)
    print(f"  Prediction: {result['label']} (confidence: {result['confidence']:.3f})")
    
    print("\nAI-like text:")
    result = classifier.predict_text(demo_ai)
    print(f"  Prediction: {result['label']} (confidence: {result['confidence']:.3f})")
    
    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
