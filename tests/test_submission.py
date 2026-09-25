from pathlib import Path


def test_project_files_exist():
    root = Path(__file__).resolve().parents[1]
    for rel in [
        "README.md",
        "requirements.txt",
        "scripts/validate.py",
        "src/business_entity_resolution/pipeline/train.py",
        "src/business_entity_resolution/pipeline/predict.py",
    ]:
        assert (root / rel).exists(), rel
