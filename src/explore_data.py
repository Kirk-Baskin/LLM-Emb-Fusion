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

    # Basic stats
    num_instances = df.shape[0]
    num_features = df.shape[1] - 1
    num_classes = df[label_col].nunique()

    # Class distribution (%)
    class_counts = (df[label_col].value_counts(normalize=True) * 100).round(1)

    # Build a readable string for all classes
    class_str = ', '.join([f"{cls}: {pct:.1f}%" for cls, pct in class_counts.items()])

    # Print dataset summary
    print(f"{dataset_name.capitalize():<10} {num_instances:<9} {num_features:<8} {num_classes:<6}")
    print(f"  Class Llbels: {df[label_col].unique()}")
    print(f"  Class distribution: {class_str}")

    # Serialize first few examples (exclude label)
    example_df = df.head(5).drop(columns=[label_col])
    text = serialize(example_df)
    print(f"  Example text: {text}\n")
