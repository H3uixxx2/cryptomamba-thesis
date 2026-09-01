import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
ARCHITECTURE_SCREEN = REPO_ROOT / "frontend" / "src" / "screens" / "ArchitectureScreen.tsx"


class TestArchitectureUiContracts(unittest.TestCase):
    def test_happy_path_omits_internal_hashes_and_keeps_useful_runtime_context(self) -> None:
        source = ARCHITECTURE_SCREEN.read_text(encoding="utf-8")

        self.assertIn('eyebrow="Training and runtime context"', source)
        self.assertIn('label="Checkpoint selection"', source)
        self.assertIn('label="Runtime"', source)
        for internal_detail in (
            "Checkpoint SHA-256",
            "Evidence manifest",
            "Source commit",
            "data.model.checkpoint_sha256",
            "data.model.source_commit",
            "data.final_evidence_sha256",
        ):
            with self.subTest(internal_detail=internal_detail):
                self.assertNotIn(internal_detail, source)

    def test_architecture_does_not_repeat_evaluation_or_trading_tables(self) -> None:
        source = ARCHITECTURE_SCREEN.read_text(encoding="utf-8")

        self.assertNotIn("S5-Full paired evidence", source)
        self.assertNotIn("Corrected self-financing trading", source)
        self.assertNotIn("<table", source)
        self.assertNotIn("overflow-x-auto", source)


if __name__ == "__main__":
    unittest.main()
