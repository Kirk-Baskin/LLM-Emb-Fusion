from utils import get_dataset, serialize

datasets = [
    'adult',
    'bank',
    'blood',
    'car',
    'credit-g',
    'diabetes',
    'heart'
]

for dataset_name in datasets:
    df = get_dataset(dataset_name=dataset_name)
    label_col = 'label'

    # --- Basic stats ---
    num_instances = df.shape[0]
    num_features = df.shape[1] - 1
    num_classes = df[label_col].nunique()

    # --- Class info ---
    class_labels = sorted(df[label_col].unique())
    class_counts = (df[label_col].value_counts(normalize=True) * 100).round(1)
    class_str = ', '.join([f"{cls}: {pct:.1f}%" for cls, pct in class_counts.items()])

    # --- Example text ---
    example_text = serialize(df.head(1).drop(columns=[label_col]))

    # --- Output ---
    print(f"\n📘 Dataset: {dataset_name.capitalize()}")
    print(f"   • Instances: {num_instances:,}")
    print(f"   • Features:  {num_features}")
    print(f"   • Labels:    {num_classes}")
    print(f"   • Unique classes: {class_labels}")
    print(f"   • Class distribution: {class_str}")
    print(f"   • Example (1 record): {example_text}")
