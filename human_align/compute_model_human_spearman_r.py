import os, glob, json
import numpy as np
from scipy.stats import spearmanr

# --------------------------------------------------------------------------- #
# Loading functions: collect ability mapping simultaneously
# --------------------------------------------------------------------------- #
def load_human_scores(folder):
    """Load human annotation jsonl files, return:
    - human_scores: {annotator: {key: mos}}
    - ability_map:  {key: ability}
    where key = f"{did}_{iid}"
    """
    human_scores = {}
    ability_map = {}
    for path in glob.glob(os.path.join(folder, '*.jsonl')):
        annot = os.path.splitext(os.path.basename(path))[0]
        scores = {}
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                obj = json.loads(line)
                did, iid = obj.get('id'), obj.get('instruct_id')
                if did is None or iid is None:
                    continue
                key = f"{did}_{iid}"
                # Collect ability
                ability = obj.get('ability')
                if ability and key not in ability_map:
                    ability_map[key] = ability

                mos_field = "mos"
                if mos_field is None:
                    continue
                try:
                    mos = float(obj[mos_field])
                except Exception:
                    continue
                scores[key] = mos
        if scores:
            human_scores[annot] = scores
    return human_scores, ability_map

def load_model_scores(path, ability_map=None):
    """Load model evaluation jsonl file, return:
    - model_scores: {key: score}
    - ability_map:  updated {key: ability}
    """
    if ability_map is None:
        ability_map = {}
    model_scores = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line)
            did, iid = obj.get('id'), obj.get('instruct_id')
            if did is None or iid is None or 'gemini_score' not in obj:
                continue
            key = f"{did}_{iid}"
            # Collect ability (don't overwrite existing)
            ability = obj.get('ability')
            if ability and key not in ability_map:
                ability_map[key] = ability

            try:
                score = float(obj['gemini_score'])
            except Exception:
                continue
            model_scores[key] = score
    return model_scores, ability_map

# --------------------------------------------------------------------------- #
# Correlation Analysis Functions
# --------------------------------------------------------------------------- #
def compute_individual_correlations(human_scores, model_scores):
    """Compute correlations between individual annotators and model"""
    correlations = []
    total_samples = 0
    total_common = 0
    
    for annot, scores in human_scores.items():
        common = set(scores) & set(model_scores)
        n = len(common)
        total_samples += len(scores)
        total_common += n
        
        if n >= 2:
            x = [scores[k] for k in common]
            y = [model_scores[k] for k in common]
            r, p = spearmanr(x, y)
            correlations.append(r)
    
    return correlations, total_samples, total_common

def compute_pairwise_human_correlations(human_scores):
    """Compute pairwise correlations between human annotators"""
    annots = list(human_scores.keys())
    correlations = []
    total_pairs = 0
    valid_pairs = 0
    
    for i in range(len(annots)):
        for j in range(i+1, len(annots)):
            a1, a2 = annots[i], annots[j]
            s1, s2 = human_scores[a1], human_scores[a2]
            common = set(s1) & set(s2)
            n = len(common)
            total_pairs += 1
            
            if n >= 2:
                valid_pairs += 1
                x = [s1[k] for k in common]
                y = [s2[k] for k in common]
                r, p = spearmanr(x, y)
                correlations.append(r)
    
    return correlations, total_pairs, valid_pairs

def compute_overall_correlation(human_scores, model_scores):
    """Compute overall correlation between human average and model"""
    # Calculate average human scores
    item2scores = {}
    for adict in human_scores.values():
        for k, v in adict.items():
            item2scores.setdefault(k, []).append(v)
    avg_human = {k: np.mean(v) for k, v in item2scores.items()}
    
    common_all = set(avg_human) & set(model_scores)
    n_all = len(common_all)
    
    if n_all >= 2:
        x_all = [avg_human[k] for k in common_all]
        y_all = [model_scores[k] for k in common_all]
        r_all, p_all = spearmanr(x_all, y_all)
        return n_all, r_all, p_all
    else:
        return n_all, None, None

def get_major_category(ability):
    """Extract major category from ability string"""
    if not ability or '/' not in ability:
        return ability
    return ability.split('/')[0]

def filter_by_category(data_dict, ability_map, category):
    """Filter data dictionary by category"""
    filtered = {}
    for key, value in data_dict.items():
        if key in ability_map:
            major_cat = get_major_category(ability_map[key])
            if major_cat == category:
                filtered[key] = value
    return filtered

def compute_category_correlations(human_scores, model_scores, ability_map):
    """Compute correlations for each major category"""
    # Get all categories
    categories = set()
    for ability in ability_map.values():
        categories.add(get_major_category(ability))
    
    category_results = {}
    
    for category in categories:
        # Filter human scores by category
        category_human_scores = {}
        for annotator, scores in human_scores.items():
            filtered_scores = filter_by_category(scores, ability_map, category)
            if filtered_scores:
                category_human_scores[annotator] = filtered_scores
        
        # Filter model scores by category
        category_model_scores = filter_by_category(model_scores, ability_map, category)
        
        if not category_human_scores or not category_model_scores:
            category_results[category] = {
                'human_vs_human': (0, None),
                'individual_vs_model': (0, None),
                'overall': (0, None, None)
            }
            continue
        
        # 1. Human vs Human for this category
        human_corrs, _, _ = compute_pairwise_human_correlations(category_human_scores)
        
        # 2. Individual vs Model for this category
        individual_corrs, _, _ = compute_individual_correlations(category_human_scores, category_model_scores)
        
        # 3. Overall correlation for this category
        overall_n, overall_r, overall_p = compute_overall_correlation(category_human_scores, category_model_scores)
        
        category_results[category] = {
            'human_vs_human': (len(human_corrs), np.mean(human_corrs) if human_corrs else None),
            'individual_vs_model': (len(individual_corrs), np.mean(individual_corrs) if individual_corrs else None),
            'overall': (overall_n, overall_r, overall_p)
        }
    
    return category_results

# --------------------------------------------------------------------------- #
# Display Functions
# --------------------------------------------------------------------------- #
def print_summary_table(results_by_language):
    """Print a summary table comparing results across languages"""
    print("\n" + "="*60)
    print("CORRELATION ANALYSIS SUMMARY")
    print("="*60)

    # Header
    print(f"{'Metric':<25} {'English':<15} {'Chinese':<15}")
    print("-" * 55)

    # Overall metrics
    print(f"{'Human vs Human':<25}", end="")
    for lang in results_by_language:
        result = results_by_language[lang]
        human_corrs = result['human_correlations']
        avg_human = np.mean(human_corrs) if human_corrs else None
        if avg_human is not None:
            print(f" {avg_human:.4f}{'':>10}", end="")
        else:
            print(f" {'N/A':>15}", end="")
    print()

    print(f"{'Human Avg vs Model':<25}", end="")
    for lang in results_by_language:
        result = results_by_language[lang]
        overall_r = result['overall_correlation'][1]
        if overall_r is not None:
            print(f" {overall_r:.4f}{'':>10}", end="")
        else:
            print(f" {'N/A':>15}", end="")
    print()

    print(f"{'Individual vs Model':<25}", end="")
    for lang in results_by_language:
        result = results_by_language[lang]
        individual_corrs = result['individual_correlations']
        avg_individual = np.mean(individual_corrs) if individual_corrs else None
        if avg_individual is not None:
            print(f" {avg_individual:.4f}{'':>10}", end="")
        else:
            print(f" {'N/A':>15}", end="")
    print()

    # Get all categories
    all_categories = set()
    for result in results_by_language.values():
        all_categories.update(result['category_correlations'].keys())
    
    if all_categories:
        print("-" * 55)
        print("CATEGORY BREAKDOWN")
        print("-" * 55)
        
        for category in sorted(all_categories):
            print(f"\n{category.upper()} Category:")
            
            # Human vs Human for category
            print(f"{'  Human vs Human':<25}", end="")
            for lang in results_by_language:
                result = results_by_language[lang]
                cat_results = result['category_correlations']
                if category in cat_results:
                    n, r = cat_results[category]['human_vs_human']
                    if r is not None:
                        print(f" {r:.4f}{'':>10}", end="")
                    else:
                        print(f" {'N/A':>15}", end="")
                else:
                    print(f" {'N/A':>15}", end="")
            print()
            
            # Human Avg vs Model for category
            print(f"{'  Human Avg vs Model':<25}", end="")
            for lang in results_by_language:
                result = results_by_language[lang]
                cat_results = result['category_correlations']
                if category in cat_results:
                    n, r, p = cat_results[category]['overall']
                    if r is not None:
                        print(f" {r:.4f}{'':>10}", end="")
                    else:
                        print(f" {'N/A':>15}", end="")
                else:
                    print(f" {'N/A':>15}", end="")
            print()
            
            # Individual vs Model for category
            print(f"{'  Individual vs Model':<25}", end="")
            for lang in results_by_language:
                result = results_by_language[lang]
                cat_results = result['category_correlations']
                if category in cat_results:
                    n, r = cat_results[category]['individual_vs_model']
                    if r is not None:
                        print(f" {r:.4f}{'':>10}", end="")
                    else:
                        print(f" {'N/A':>15}", end="")
                else:
                    print(f" {'N/A':>15}", end="")
            print()

def analyze_language(lang, human_folder, model_file):
    """Analyze correlations for a single language"""
    print(f"\n{'='*15} {lang.upper()} {'='*15}")
    
    # Load data
    human_scores, ability_map = load_human_scores(human_folder)
    model_scores, ability_map = load_model_scores(model_file, ability_map=ability_map)
    
    print(f"Annotators: {len(human_scores)}, Model scores: {len(model_scores)}")
    
    # 1. Individual annotator vs model correlations
    individual_corrs, total_samples, total_common = compute_individual_correlations(human_scores, model_scores)
    
    # 2. Human vs human correlations
    human_corrs, total_pairs, valid_pairs = compute_pairwise_human_correlations(human_scores)
    
    # 3. Overall correlation
    overall_n, overall_r, overall_p = compute_overall_correlation(human_scores, model_scores)
    
    # 4. Category correlations
    category_corrs = compute_category_correlations(human_scores, model_scores, ability_map)
    
    # Print results
    print(f"\nOverall Results:")
    print(f"Human vs Human: {np.mean(human_corrs):.4f} (pairs: {len(human_corrs)})")
    
    if overall_r is not None:
        print(f"Human Avg vs Model: {overall_r:.4f} (n: {overall_n})")
    else:
        print(f"Human Avg vs Model: N/A (n: {overall_n})")
    
    print(f"Individual vs Model: {np.mean(individual_corrs):.4f} (annotators: {len(individual_corrs)})")
    
    print(f"\nCategory Breakdown:")
    for category in sorted(category_corrs.keys()):
        cat_data = category_corrs[category]
        print(f"\n{category}:")
        
        # Human vs Human
        n_h2h, r_h2h = cat_data['human_vs_human']
        if r_h2h is not None:
            print(f"  Human vs Human: {r_h2h:.4f} (pairs: {n_h2h})")
        else:
            print(f"  Human vs Human: N/A (pairs: {n_h2h})")
        
        # Human Avg vs Model
        n_overall, r_overall, p_overall = cat_data['overall']
        if r_overall is not None:
            print(f"  Human Avg vs Model: {r_overall:.4f} (n: {n_overall})")
        else:
            print(f"  Human Avg vs Model: N/A (n: {n_overall})")
        
        # Individual vs Model
        n_ind, r_ind = cat_data['individual_vs_model']
        if r_ind is not None:
            print(f"  Individual vs Model: {r_ind:.4f} (annotators: {n_ind})")
        else:
            print(f"  Individual vs Model: N/A (annotators: {n_ind})")
    
    return {
        'individual_correlations': individual_corrs,
        'human_correlations': human_corrs,
        'overall_correlation': (overall_n, overall_r, overall_p),
        'category_correlations': category_corrs
    }

# --------------------------------------------------------------------------- #
# Main execution
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    languages = ["en", "zh"]
    results_by_language = {}
    
    for lang in languages:
        human_folder = f"./VoiceGenEval-eval-results/human_eval_res/{lang}"
        model_file = f"./VoiceGenEval-eval-results/model_eval_res/{lang}_res.jsonl"
        
        results_by_language[lang] = analyze_language(lang, human_folder, model_file)
    
    print_summary_table(results_by_language)