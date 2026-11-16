from pathlib import Path

from config import settings


def test_retrieval_models_have_explicit_local_paths():
    assert Path(settings.embedding_model_path).name == "all-MiniLM-L6-v2"
    assert Path(settings.reranker_model_path).name == "ms-marco-MiniLM-L-6-v2"


def test_llm_loaders_are_local_only():
    # This repo's own llm.py, resolved from the test file -- never whatever a
    # shared sys.path (.pth entries from sibling checkouts) or CWD resolves to.
    source = (Path(__file__).resolve().parents[1] / "llm.py").read_text()

    assert source.count("local_files_only=True") == 2
