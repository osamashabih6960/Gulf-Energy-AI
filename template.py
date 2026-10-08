from pathlib import Path


PROJECT_NAME = "Gulf-Energy-AI"


folders = [
    "data/raw",
    "data/processed",
    "data/features",

    "src",

    "models",

    "notebooks",

    "tests",

    "app",

    "airflow/dags",

    "monitoring",

    "k8s",

    ".github/workflows",
]


files = [
    "requirements.txt",
    "params.yaml",
    "dvc.yaml",
    "Dockerfile",
    ".gitignore",
    "README.md",

    "src/__init__.py",
    "src/data_ingestion.py",
    "src/data_validation.py",
    "src/feature_engineering.py",
    "src/train_xgboost.py",
    "src/train_lstm.py",
    "src/evaluate.py",
    "src/predict.py",

    "tests/__init__.py",
    "app/__init__.py",

    "airflow/dags/__init__.py",
]


def create_project_structure():

    print(f"\nCreating project: {PROJECT_NAME}\n")

    # Create folders
    for folder in folders:
        path = Path(folder)
        path.mkdir(parents=True, exist_ok=True)
        print(f"[DIR]  {path}")

    # Create files
    for file in files:
        path = Path(file)
        path.parent.mkdir(parents=True, exist_ok=True)

        if not path.exists():
            path.touch()

        print(f"[FILE] {path}")

    print("\nProject structure created successfully!")
    print(f"\nProject: {PROJECT_NAME}")


if __name__ == "__main__":
    create_project_structure()