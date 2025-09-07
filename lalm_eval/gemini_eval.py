#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple, Any

import matplotlib.pyplot as plt
import pandas as pd
import google.generativeai as genai
from tqdm import tqdm

plt.rcParams.update({"font.size": 10})

SUB_CATS = {
    "acoustic_attributes": [
        "acoustic_attributes/age",
        "acoustic_attributes/speed",
        "acoustic_attributes/gender",
        "acoustic_attributes/emotion",
        "acoustic_attributes/pitch",
        "acoustic_attributes/volume",
        "acoustic_attributes/composite_properties"
    ],
    "instruction": [
        "instruction/emotion",
        "instruction/variation",
        "instruction/style",
    ],
    "role_play": [
        "role_play/character",
        "role_play/scenario",
    ],
    "empathy": [
        "empathy/anger",
        "empathy/sadness_disappointment",
        "empathy/anxiety_fear",
        "empathy/joy_excitement",
    ],
}

# --------------------------- Utility Functions --------------------------------

def load_prompts(prompts_dir: Path) -> Dict[str, str]:
    mapping = {}
    for file in prompts_dir.glob("*.txt"):
        mapping[file.stem] = file.read_text(encoding="utf-8")
    if not mapping:
        raise RuntimeError(f"No *.txt prompt files found in {prompts_dir}")
    return mapping

def construct_prompt(template: str, instruction: str, ability: str) -> str:           
    return template.format(
        instruction_type=ability,
        input_instruction=instruction
    )

def call_gemini_api(
    prompt_text: str,
    response_audio_path: Path,
    model_name: str,
    max_retry: int,
    sleep_between_retry: int,
    instruction_audio_path: Path = None
) -> Tuple[bool, str]:
    """
    Call model using official Gemini API
    """
    model = genai.GenerativeModel(model_name=model_name)
    
    last_err_msg = ""
    for attempt in range(1, max_retry + 1):
        try:
            # Upload audio files
            uploaded_files = []
            
            # Upload response audio file
            response_file = genai.upload_file(path=str(response_audio_path))
            uploaded_files.append(response_file)
            content_parts = [prompt_text, response_file]
            
            # Upload instruction audio if exists
            if instruction_audio_path and instruction_audio_path.exists():
                instruction_file = genai.upload_file(path=str(instruction_audio_path))
                uploaded_files.append(instruction_file)
                content_parts.insert(-1, instruction_file)  # Insert instruction audio before response audio
            
            # Generate content with model
            response = model.generate_content(content_parts)
            
            if response.text:
                reply_text = response.text
                
                # Only consider successful if valid numeric score can be parsed
                score_str = parse_score(reply_text)
                if score_str and safe_float(score_str) is not None:
                    return True, reply_text
                
                last_err_msg = "Unable to parse score"
            else:
                last_err_msg = "Model failed to generate response"
                
        except Exception as e:
            last_err_msg = str(e)
            print(f"API call error (attempt {attempt}/{max_retry}): {last_err_msg}")

        if attempt < max_retry:
            time.sleep(sleep_between_retry)

    return False, last_err_msg

def safe_float(x: Any):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def parse_score(model_reply: str) -> str:
    """
    Extract score from model reply:
    1. Find all segments like [[...]]
    2. Search backwards for the first value that can be converted to float
    3. Return that string on success, empty string on failure
    """
    matches = re.findall(r"\[\[(.*?)]]", model_reply, flags=re.S)
    for s in reversed(matches):
        if safe_float(s) is not None:
            return s.strip()
    # Fallback: try treating entire reply as number
    if safe_float(model_reply.strip()) is not None:
        return model_reply.strip()
    return ""

# --------------------------- Evaluation Core --------------------------------

def evaluate_one(
    sample: dict,
    root_dir: Path,
    prompts: Dict[str, str],
    out_dir: Path,
    model_name: str,
    max_retry: int,
    sleep_between_retry: int,
) -> dict:
    """Evaluate single sample and write complete output to txt file"""
    ability = sample["ability"]
    big_cat, small_cat = ability.split("/", 1)
    sample_model_name = sample.get("model_name", "unknown_model")
    sample_id = sample.get("id", "unknown_id")

    if big_cat not in prompts:
        error_msg = f"Prompt template not found for category {big_cat}"
        print(f"⚠️  Sample ID {sample_id}: {error_msg}")
        sample["gemini_score"] = error_msg
        return sample

    # Enhanced audio file validation
    response_audio_path_str = sample.get("response_audio_path", "")
    
    # Check if response_audio_path is empty
    if not response_audio_path_str or response_audio_path_str.strip() == "":
        error_msg = f"Response audio path is empty"
        print(f"⚠️  Sample ID {sample_id} (Model: {sample_model_name}): {error_msg}")
        sample["gemini_score"] = error_msg
        return sample
    
    # Construct full audio file path
    response_audio_path = root_dir / response_audio_path_str.strip()
    
    # Check if audio file exists
    if not response_audio_path.exists():
        error_msg = f"Audio file does not exist: {response_audio_path}"
        print(f"⚠️  Sample ID {sample_id} (Model: {sample_model_name}): {error_msg}")
        sample["gemini_score"] = error_msg
        return sample

    # Check for instruction audio file
    instruction_audio_path = None
    if "instruction_audio_path" in sample and sample["instruction_audio_path"]:
        instruction_audio_path = root_dir / sample["instruction_audio_path"]
        if not instruction_audio_path.exists():
            print(f"ℹ️  Sample ID {sample_id}: Instruction audio file not found: {instruction_audio_path}, proceeding without it")
            instruction_audio_path = None

    prompt_text = construct_prompt(
        prompts[big_cat], 
        instruction=sample["instruct_text"], 
        ability=sample["ability"]
    )
    
    success, reply = call_gemini_api(
        prompt_text, 
        response_audio_path,
        model_name,
        max_retry,
        sleep_between_retry,
        instruction_audio_path=instruction_audio_path
    )
    
    score = parse_score(reply)
    sample["gemini_score"] = score if success else f"API failed: {reply}"
    sample["gemini_raw"] = reply

    # -------- Save complete output ---------- #
    save_path = (
        out_dir / sample_model_name / "gemini_api_res" / big_cat / small_cat / f"{sample['id']}.txt"
    )
    ensure_dir(save_path.parent)
    save_path.write_text(reply, encoding="utf-8")

    return sample

def load_processed_ids_and_clean_failures(scored_jsonl: Path, overwrite: bool) -> Tuple[set, int]:
    """
    Read existing metadata_with_score.jsonl, return set of successfully processed IDs,
    and clean failed records in-place
    
    Returns:
        Tuple[set, int]: (processed_ids, cleaned_count)
    """
    processed = set()
    cleaned_count = 0
    
    if not scored_jsonl.exists() or overwrite:
        return processed, cleaned_count
    
    # Read all lines
    valid_lines = []
    total_lines = 0
    
    with scored_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            total_lines += 1
            line = line.strip()
            if not line:
                continue
                
            try:
                rec = json.loads(line)
                gemini_score = rec.get("gemini_score", "")
                
                # Check if it's a valid numeric score
                score_float = safe_float(gemini_score)
                if score_float is not None:
                    # Valid score, keep this record
                    processed.add(rec.get("id"))
                    valid_lines.append(line)
                else:
                    # Failed score (error message), mark for deletion
                    cleaned_count += 1
                    print(f"🗑️  Removing failed record ID {rec.get('id')}: {gemini_score}")
                    
            except Exception as e:
                # JSON parsing failed, mark for deletion
                cleaned_count += 1
                print(f"🗑️  Removing invalid JSON line: {str(e)}")
    
    # Rewrite file with only valid lines if any cleaning occurred
    if cleaned_count > 0:
        print(f"📝 Cleaning {cleaned_count} failed records from {scored_jsonl}")
        with scored_jsonl.open("w", encoding="utf-8") as f:
            for line in valid_lines:
                f.write(line + "\n")
        print(f"✅ Cleaned file saved. Kept {len(valid_lines)} valid records out of {total_lines} total lines.")
    
    return processed, cleaned_count

def remove_duplicates_from_tasks(tasks: List[dict]) -> Tuple[List[dict], Dict[str, int]]:
    """
    Remove duplicate samples based on ID, keeping the first occurrence
    
    Args:
        tasks: List of sample dictionaries
        
    Returns:
        Tuple[List[dict], Dict[str, int]]: (deduplicated_tasks, stats)
    """
    seen_ids = set()
    unique_tasks = []
    duplicate_count = 0
    duplicate_details = defaultdict(int)
    
    for task in tasks:
        task_id = task.get("id")
        ability = task.get("ability", "unknown")
        
        if task_id not in seen_ids:
            seen_ids.add(task_id)
            unique_tasks.append(task)
        else:
            duplicate_count += 1
            duplicate_details[ability] += 1
            print(f"🔄 Removing duplicate sample ID: {task_id} (ability: {ability})")
    
    stats = {
        "total_duplicates": duplicate_count,
        "by_ability": dict(duplicate_details)
    }
    
    return unique_tasks, stats

def eval_all_samples(
    root_dir: Path,
    metadata_path: Path,
    prompts_dir: Path,
    out_dir: Path,
    max_per_ability: int,
    concurrency: int,
    overwrite: bool,
    model_name: str,
    max_retry: int,
    sleep_between_retry: int,
) -> Path:
    prompts = load_prompts(prompts_dir)

    # --------- Resume from breakpoint: processed IDs and clean failures ---------
    scored_jsonl = out_dir / "metadata_with_score.jsonl"
    processed_ids, cleaned_count = load_processed_ids_and_clean_failures(scored_jsonl, overwrite)

    # 1. 首先收集所有符合条件的任务到 tasks 列表中
    tasks: List[dict] = []
    per_ability_counter = defaultdict(int)

    print(f"📖 Reading samples from: {metadata_path}")
    
    with metadata_path.open(encoding="utf-8") as fin:
        for line in fin:
            try:
                sample = json.loads(line)
            except json.JSONDecodeError:
                continue
                
            # Check required fields
            if not sample.get("id") or not sample.get("ability"):
                continue
                
            ability = sample["ability"]
            
            # Check if already processed
            if sample["id"] in processed_ids:
                continue
            
            # Check max per ability limit
            if per_ability_counter[ability] >= max_per_ability:
                continue
            
            per_ability_counter[ability] += 1
            tasks.append(sample)

    print(f"📊 Step 1 - After initial filtering: {len(tasks)} samples")

    # 2. 记录去重前的任务数量
    original_task_count = len(tasks)
    print(f"📊 Step 2 - Before deduplication: {original_task_count} samples")

    # 3. 执行去重操作
    tasks, duplicate_stats = remove_duplicates_from_tasks(tasks)
    print(f"📊 Step 3 - After deduplication: {len(tasks)} samples (removed {duplicate_stats['total_duplicates']} duplicates)")

    if not tasks:
        print("\n✅ No new samples to process, program ended.")
        return scored_jsonl

    ensure_dir(scored_jsonl.parent)
    write_mode = "a" if scored_jsonl.exists() and not overwrite else "w"

    print(f"\n🎯 Starting evaluation of {len(tasks)} samples...")
    print("Audio file validation will be performed for each sample.")
    print("="*80)

    # Concurrent evaluation
    with scored_jsonl.open(write_mode, encoding="utf-8") as fout, \
            ThreadPoolExecutor(max_workers=concurrency) as executor, \
            tqdm(total=len(tasks), desc="Evaluating") as pbar:

        futures = {
            executor.submit(
                evaluate_one, s, root_dir, prompts, out_dir, model_name, max_retry, sleep_between_retry
            ): s
            for s in tasks
        }

        for fut in as_completed(futures):
            res = fut.result()
            fout.write(json.dumps(res, ensure_ascii=False) + "\n")
            pbar.update(1)

    print(f"\n✅ All processing complete, wrote {len(tasks)} new results -> {scored_jsonl}")
    return scored_jsonl

# --------------------------- Analysis and Visualization ------------------------------

def get_ordered_abilities():
    """Get abilities in the order defined by SUB_CATS"""
    # Flatten the SUB_CATS to get ordered list of abilities
    ordered_abilities = []
    for big_cat in ["acoustic_attributes", "instruction", "role_play", "empathy"]:
        ordered_abilities.extend(SUB_CATS[big_cat])
    
    return ordered_abilities

def analyze_scores(scored_jsonl: Path, out_dir: Path):
    # ========== Read and clean ==========
    records = [
        json.loads(l)
        for l in scored_jsonl.read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]

    rows, parse_fail_ids = [], []
    for rec in records:
        score = safe_float(rec.get("gemini_score"))
        ability = rec.get("ability")

        if score is None:
            parse_fail_ids.append(rec.get("id"))
            continue
        rows.append(
            dict(
                id=rec.get("id"),
                model=rec.get("model_name"),
                ability=rec.get("ability"),
                score=score,
            )
        )
    print("len(rows):", len(rows))

    if not rows:
        print("❌ No usable score data, script terminated.")
        return

    df = pd.DataFrame(rows)

    # ========== Weight table ==========
    BIG_CAT_W = 0.25  # Four major categories each get 0.25
    WEIGHTS = {}

    # 1) acoustic_attributes
    ap_comp = "acoustic_attributes/composite_properties"
    ap_comp_w = BIG_CAT_W * 0.5                     # 0.125
    ap_rem_each = (BIG_CAT_W - ap_comp_w) / (len(SUB_CATS["acoustic_attributes"]) - 1)
    for ab in SUB_CATS["acoustic_attributes"]:
        WEIGHTS[ab] = ap_comp_w if ab == ap_comp else ap_rem_each

    # 2) Other three major categories: equal distribution among subcategories
    for big in ("instruction", "role_play", "empathy"):
        each = BIG_CAT_W / len(SUB_CATS[big])
        for ab in SUB_CATS[big]:
            WEIGHTS[ab] = each

    # Get ordered abilities list
    ordered_abilities = get_ordered_abilities()

    # ========== Print statistics ==========
    print("\n================= Statistical Results =================")
    for model, g in df.groupby("model"):
        print(f"\nModel: {model}")

        # ---- New: weighted overall average score
        sub_means = g.groupby("ability")["score"].mean().to_dict()
        weighted_sum, used_w = 0.0, 0.0
        for ab, w in WEIGHTS.items():
            if ab in sub_means:          # Only include when samples exist
                weighted_sum += sub_means[ab] * w
                used_w += w
        weighted_mean = weighted_sum / used_w if used_w else float("nan")
        print(f"  Weighted overall average score: {weighted_mean:.2f} (weight coverage={used_w:.1%})")
        
        # ---- NEW: Major category average scores
        print("  Major category average scores:")
        for big_cat in ["acoustic_attributes", "instruction", "role_play", "empathy"]:
            # 计算该大类的加权平均分
            big_cat_weighted_sum = 0.0
            big_cat_used_w = 0.0
            big_cat_sample_count = 0
            
            for ability in SUB_CATS[big_cat]:
                if ability in sub_means:
                    weight = WEIGHTS[ability]
                    big_cat_weighted_sum += sub_means[ability] * weight
                    big_cat_used_w += weight
                    big_cat_sample_count += g[g["ability"] == ability].shape[0]
            
            if big_cat_used_w > 0:
                big_cat_avg = big_cat_weighted_sum / big_cat_used_w
                coverage = big_cat_used_w / BIG_CAT_W
                print(f"    {big_cat:<20s}: {big_cat_avg:.2f} (n={big_cat_sample_count}, coverage={coverage:.1%})")
            else:
                print(f"    {big_cat:<20s}: N/A (no data)")
        
        # ---- ability average scores (按SUB_CATS顺序打印)
        print("  Ability average scores:")
        ability_means = g.groupby("ability")["score"].mean().to_dict()
        for ab in ordered_abilities:
            if ab in ability_means:
                cnt = g[g["ability"] == ab].shape[0]
                weight = WEIGHTS.get(ab, 0.0)
                print(f"    {ab:<40s}: {ability_means[ab]:.2f} (n={cnt}, weight={weight:.3f})")

    # ========== Visualization ==========
    print("\nStarting visualization...")
    for model, g in df.groupby("model"):
        model_dir = out_dir / model
        ensure_dir(model_dir)

        # ---- NEW: Major category bar chart
        sub_means = g.groupby("ability")["score"].mean().to_dict()
        big_cat_scores = []
        big_cat_labels = []
        
        for big_cat in ["acoustic_attributes", "instruction", "role_play", "empathy"]:
            big_cat_weighted_sum = 0.0
            big_cat_used_w = 0.0
            
            for ability in SUB_CATS[big_cat]:
                if ability in sub_means:
                    weight = WEIGHTS[ability]
                    big_cat_weighted_sum += sub_means[ability] * weight
                    big_cat_used_w += weight
            
            if big_cat_used_w > 0:
                big_cat_avg = big_cat_weighted_sum / big_cat_used_w
                big_cat_scores.append(big_cat_avg)
                big_cat_labels.append(big_cat.replace("_", " ").title())
        
        if big_cat_scores:
            plt.figure(figsize=(10, 6))
            bars = plt.bar(range(len(big_cat_scores)), big_cat_scores, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
            plt.xticks(range(len(big_cat_labels)), big_cat_labels, rotation=45, ha='right')
            
            # Add value labels on top of bars
            for i, (bar, score) in enumerate(zip(bars, big_cat_scores)):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                        f'{score:.2f}', ha='center', va='bottom', fontweight='bold')
            
            # Draw weighted average line
            w_sum = sum(
                sub_means[ab] * w
                for ab, w in WEIGHTS.items()
                if ab in sub_means
            )
            w_used = sum(w for ab, w in WEIGHTS.items() if ab in sub_means)
            w_avg = w_sum / w_used if w_used else float("nan")
            plt.axhline(w_avg, color="red", linestyle="--", linewidth=2, label=f"Overall average: {w_avg:.2f}")
            
            plt.ylabel("Average Score")
            plt.title(f"{model} - Major Category Scores")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(model_dir / "major_category_scores.png", dpi=200, bbox_inches='tight')
            plt.close()

        # ---- Original ability chart (按SUB_CATS顺序画图)
        ability_mean = g.groupby("ability")["score"].mean()
        
        # Filter and reorder according to SUB_CATS order
        ordered_data = []
        ordered_labels = []
        for ab in ordered_abilities:
            if ab in ability_mean.index:
                ordered_data.append(ability_mean[ab])
                ordered_labels.append(ab)
        
        plt.figure(figsize=(8, 4 + 0.25 * len(ordered_data)))
        y_pos = range(len(ordered_data))
        plt.barh(y_pos, ordered_data, color="#4C72B0")
        plt.yticks(y_pos, ordered_labels)
        
        # Draw weighted average dashed line
        sub_means = g.groupby("ability")["score"].mean().to_dict()
        w_sum = sum(
            sub_means[ab] * w
            for ab, w in WEIGHTS.items()
            if ab in sub_means
        )
        w_used = sum(w for ab, w in WEIGHTS.items() if ab in sub_means)
        w_avg = w_sum / w_used if w_used else float("nan")
        plt.axvline(w_avg, color="red", linestyle="--", label="Weighted overall average")
        plt.xlabel("Average score")
        plt.title(f"{model} - Ability average scores")
        plt.legend()
        plt.tight_layout()
        plt.savefig(model_dir / "ability_scores.png", dpi=200)
        plt.close()

    print(f"✅ Visualization complete, saved to folder: {out_dir}")

    # ========== Parse failures ==========
    print("\n================= Parse Failure Information =================")
    if parse_fail_ids:
        print(f"Total of {len(parse_fail_ids)} records failed gemini_score parsing, ID list:")
        print(parse_fail_ids)
    else:
        print("All records' gemini_score parsed successfully.")

# ----------------------------- CLI ----------------------------------

def main():
    root = "./data/examples"
    
    parser = argparse.ArgumentParser(description="Concurrent Gemini API calls for voice result scoring (supports resume from breakpoint)")
    parser.add_argument("--root_dir", default=f"{root}/model_res/en/wav",
                        help="wav directory path")
    parser.add_argument("--metadata_path", default=f"{root}/model_res/en/metadata.jsonl",
                        help="metadata.jsonl path")
    parser.add_argument("--prompts_dir", default=f"lalm_eval/eval_prompts/en",
                        help="Directory containing four major category prompts")
    parser.add_argument("--out_dir", default=f"{root}/eval_res/en",
                        help="Output root directory (evaluation results & images)")
    parser.add_argument("--gemini_api_key", required=True,
                        help="Gemini API Key; can use environment variable GEMINI_API_KEY")
    parser.add_argument("--model_name", default="gemini-2.5-pro-preview-06-05",
                        help="Gemini model name")
    parser.add_argument("--max_retry_api", type=int, default=5,
                        help="Maximum number of API retry attempts")
    parser.add_argument("--sleep_between_retry", type=int, default=5,
                        help="Sleep time between retries (seconds)")
    parser.add_argument("--max_tokens", type=int, default=4096,
                        help="Maximum number of tokens")
    parser.add_argument("--max_per_ability", type=int, default=100000,
                        help="Maximum number of evaluations per ability (default 100000)")
    parser.add_argument("--concurrency", type=int, default=4,
                        help="Number of8 concurrent threads (default 4)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing results")
    args = parser.parse_args()
    
    args.out_path =f"{args.out_dir}/metadata_with_score.jsonl"
    
    print(f"args: {args}")

    # Configure Gemini API
    if args.gemini_api_key:
        genai.configure(api_key=args.gemini_api_key)
    else:
        print("Error: No Gemini API Key provided, please provide via --gemini_api_key parameter or GEMINI_API_KEY environment variable")
        sys.exit(1)

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    scored_jsonl = eval_all_samples(
        root_dir=Path(f"{args.root_dir}"),
        metadata_path=Path(args.metadata_path),
        prompts_dir=Path(args.prompts_dir),
        out_dir=out_dir,
        max_per_ability=args.max_per_ability,
        concurrency=args.concurrency,
        overwrite=args.overwrite,
        model_name=args.model_name,
        max_retry=args.max_retry_api,
        sleep_between_retry=args.sleep_between_retry,
    )

    analyze_scores(scored_jsonl, out_dir)

if __name__ == "__main__":
    main()