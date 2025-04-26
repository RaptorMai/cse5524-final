import argparse
from collections import defaultdict
from functools import reduce
import pandas as pd


def borda_count_ensemble(dfs):
    """
    Perform Borda Count ensemble on a list of DataFrames.

    Args:
        dfs (List[pd.DataFrame]): DataFrames with 'observationId' and 'predictions'.

    Returns:
        pd.DataFrame: 'observationId' and ensembled 'predictions'.
    """
    # Rename predictions columns to merge
    for i, df in enumerate(dfs):
        df.rename(columns={'predictions': f'predictions_{i}'}, inplace=True)

    # Merge all DataFrames on observationId
    merged = reduce(
        lambda left, right: pd.merge(left, right, on='observationId', how='inner'),
        dfs
    )

    final_ids = []
    final_preds = []

    # Iterate over each observation
    for _, row in merged.iterrows():
        score_dict = defaultdict(int)
        # Accumulate scores from each model
        for i in range(len(dfs)):
            preds = str(row[f'predictions_{i}']).split()
            for rank, label in enumerate(preds):
                score_dict[label] += (5 - rank)

        # Sort by score descending, take top 5
        sorted_items = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)
        top5 = [label for label, _ in sorted_items[:5]]

        final_ids.append(row['observationId'])
        final_preds.append(" ".join(top5))

    return pd.DataFrame({
        'observationId': final_ids,
        'predictions': final_preds
    })


def main():
    parser = argparse.ArgumentParser(
        description="Ensemble top-5 model predictions with Borda Count."
    )
    parser.add_argument(
        'inputs',
        type=str,
        nargs='*',
        default=[],
        help='Paths to model prediction CSV files'
    )
    parser.add_argument(
        '-o', '--output', default='submissions/ensemble.csv',
        help='Path to output ensembled CSV'
    )
    args = parser.parse_args()

    # Load all input CSVs
    args.inputs = ['submissions/vit_base_patch14_dinov2_petl/fungi/lora_8_adamw_TrivialAugment_seesaw_loss/classifier/submisson.csv', 'submissions/vit_base_patch14_dinov2_petl/fungi/lora_8_adamw_TrivialAugment_seesaw_loss/prototype/submisson.csv', 
    'submissions/vit_base_patch16_bioclip_224_petl/fungi/linear_adamw_AutoAugment_seesaw_loss/prototype/submisson.csv', 'submissions/vit_base_patch16_bioclip_224_petl/fungi/lora_8_adamw_AutoAugment_seesaw_loss/classifier/submisson.csv', 'submissions/vit_base_patch16_bioclip_224_petl/fungi/lora_8_adamw_AutoAugment_seesaw_loss/prototype/submisson.csv']
    dfs = [pd.read_csv(path) for path in args.inputs]

    # Perform ensemble
    ensemble_df = borda_count_ensemble(dfs)

    # Save to CSV
    ensemble_df.to_csv(args.output, index=False)
    print(f"Ensembled predictions saved to {args.output}")

if __name__ == '__main__':
    main()