import pandas as pd
import json
import re

df = pd.read_csv("results/multilingual_pilot_predictions.csv")
print(f"Total rows: {len(df)}")

def clean_label(lbl):
    if not isinstance(lbl, str):
        return "INVALID_OUTPUT"
    c = lbl.strip().upper().replace("_", "").replace(" ", "")
    if c in ("SUPPORT", "SUPPORTS", "ENTAILMENT"):
        return "SUPPORTS"
    if c in ("CONTRADICT", "CONTRADICTS", "REFUTE", "REFUTES", "CONTRADICTION"):
        return "REFUTES"
    if c in ("NOTENOUGHINFO", "NEI", "NOINFO", "NEUTRAL"):
        return "NOT_ENOUGH_INFO"
    return "INVALID_OUTPUT"

results = []
for idx, row in df.iterrows():
    raw = str(row["raw_model_output"])
    m = re.search(r'"veracity"\s*:\s*"([^"]+)"', raw, re.I)
    v_field = m.group(1) if m else "NONE"
    norm = clean_label(v_field)
    results.append({
        "row": idx + 1,
        "claim_id": row["claim_id"],
        "lang": row["language"],
        "mode": row["mode"],
        "gold": row["gold_label"],
        "v_field": v_field,
        "clean": norm,
        "old_pred": row["predicted_label"],
        "correct_under_improved": norm == row["gold_label"],
    })
    print(f"Row {idx+1:2d} | Lang: {row['language']} | Mode: {row['mode']} | Gold: {row['gold_label']:15s} | Raw_field: {v_field:15s} | Clean: {norm:15s} | Old_pred: {row['predicted_label']:15s}")

res_df = pd.DataFrame(results)
print("\n--- Accuracy Comparison ---")
for mode in ["A", "B"]:
    for lang in ["en", "es", "hi"]:
        sub = res_df[(res_df["mode"] == mode) & (res_df["lang"] == lang)]
        old_acc = (sub["old_pred"] == sub["gold"]).mean()
        new_acc = (sub["clean"] == sub["gold"]).mean()
        invalid_old = (sub["old_pred"] == "INVALID_OUTPUT").sum()
        invalid_new = (sub["clean"] == "INVALID_OUTPUT").sum()
        print(f"[{lang} - Mode {mode}] Old Acc: {old_acc:.2f} (Inv: {invalid_old}) -> Improved Acc: {new_acc:.2f} (Inv: {invalid_new})")
