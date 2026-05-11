from pathlib import Path

_dir = Path(__file__).parent

BRIEF_PARSER_SYSTEM: str = (_dir / "brief_parser.txt").read_text()
CAMPAIGN_PLANNER_SYSTEM: str = (_dir / "campaign_planner.txt").read_text()
PUBLISHER_SCORER_SYSTEM: str = (_dir / "publisher_scorer.txt").read_text()
CREATIVE_WRITER_SYSTEM: str = (_dir / "creative_writer.txt").read_text()
