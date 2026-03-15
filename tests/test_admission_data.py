"""Tests for core.admission_data — CSV loading, GPA normalization, scoring."""

from __future__ import annotations

import csv
import tempfile
from pathlib import Path

import pytest

from core.admission_data import (
    AdmissionRecord,
    classify_background,
    classify_nationality,
    compute_all_program_stats,
    compute_program_stats,
    load_admission_csv,
    normalize_gpa,
    score_internships,
    summarize_records,
)


# ===================================================================
# GPA normalization
# ===================================================================


class TestNormalizeGPA:
    """Tests for normalize_gpa()."""

    def test_scale_4_passthrough(self):
        assert normalize_gpa(3.8, 4) == 3.8

    def test_scale_4_caps_at_4(self):
        assert normalize_gpa(4.2, 4) == 4.0

    def test_scale_100_high(self):
        """91.8/100 should map to ~3.7-3.9 range."""
        result = normalize_gpa(91.8, 100)
        assert 3.7 <= result <= 3.9

    def test_scale_100_90(self):
        result = normalize_gpa(90, 100)
        assert 3.7 <= result <= 3.8

    def test_scale_100_85(self):
        result = normalize_gpa(85, 100)
        assert 3.3 <= result <= 3.5

    def test_scale_100_80(self):
        result = normalize_gpa(80, 100)
        assert 3.0 <= result <= 3.3

    def test_scale_5_high(self):
        result = normalize_gpa(4.5, 5)
        assert 3.7 <= result <= 3.9

    def test_scale_4_3_high(self):
        """3.95/4.3 should map close to 3.9+."""
        result = normalize_gpa(3.95, 4.3)
        assert result >= 3.85

    def test_scale_4_3_caps(self):
        result = normalize_gpa(4.3, 4.3)
        assert result <= 4.0

    def test_unknown_scale_linear(self):
        """Unknown scale should use linear conversion."""
        result = normalize_gpa(8.0, 10)
        assert 3.0 <= result <= 3.5


# ===================================================================
# Background classification
# ===================================================================


class TestClassifyBackground:
    """Tests for classify_background()."""

    @pytest.mark.parametrize(
        "bg_type,expected_tier",
        [
            ("海本(Top10)", 1),
            ("C9", 1),
            ("海本(Top30)", 2),
            ("985", 2),
            ("两财一贸(211)", 3),
            ("211", 3),
            ("双非一本", 4),
        ],
    )
    def test_known_types(self, bg_type, expected_tier):
        assert classify_background(bg_type) == expected_tier

    def test_unknown_defaults_to_4(self):
        assert classify_background("其他学校") == 4

    def test_partial_match(self):
        """Should match '985' within a longer string."""
        assert classify_background("某985高校") == 2


# ===================================================================
# Nationality classification
# ===================================================================


class TestClassifyNationality:
    """Tests for classify_nationality()."""

    @pytest.mark.parametrize(
        "nationality,expected",
        [
            ("美籍", "domestic"),
            ("US", "domestic"),
            ("green card", "domestic"),
            ("绿卡", "domestic"),
            ("中国大陆", "china"),
            ("中国", "china"),
            ("China", "china"),
            ("港澳台", "hk_tw"),
            ("香港", "hk_tw"),
            ("台湾", "hk_tw"),
            ("HK", "hk_tw"),
            ("韩国", "other_intl"),
            ("India", "other_intl"),
        ],
    )
    def test_known_nationalities(self, nationality, expected):
        assert classify_nationality(nationality) == expected

    def test_empty_defaults_to_china(self):
        assert classify_nationality("") == "china"
        assert classify_nationality("不明") == "china"

    def test_case_insensitive(self):
        assert classify_nationality("CHINA") == "china"
        assert classify_nationality("Domestic") == "domestic"


# ===================================================================
# Internship scoring
# ===================================================================


class TestScoreInternships:
    """Tests for score_internships()."""

    def test_empty(self):
        assert score_internships("") == 0.0
        assert score_internships("无") == 0.0

    def test_strong_intern(self):
        desc = "3段量化私募QR(含top百亿)+三中一华金工组"
        score = score_internships(desc)
        assert score >= 5.0

    def test_weak_intern(self):
        desc = "1段银行实习"
        score = score_internships(desc)
        assert 0 < score < 5.0

    def test_top_intern(self):
        desc = "2段顶级量化+1段投行"
        score = score_internships(desc)
        assert score >= 4.0

    def test_capped_at_10(self):
        desc = "3段顶级量化私募QR(含top百亿)+三中一华金工组+高盛+对冲基金"
        score = score_internships(desc)
        assert score <= 10.0


# ===================================================================
# CSV loading
# ===================================================================


class TestLoadAdmissionCSV:
    """Tests for load_admission_csv()."""

    def _write_csv(self, rows: list[dict], tmp_dir: str) -> str:
        path = Path(tmp_dir) / "test.csv"
        fieldnames = [
            "id", "gender", "bg_type", "nationality", "gpa", "gpa_scale",
            "gre", "toefl", "major", "intern_desc", "has_paper",
            "has_research", "courses_note", "program", "result",
            "season", "source",
        ]
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        return str(path)

    def test_load_basic(self, tmp_path):
        path = self._write_csv(
            [
                {
                    "id": "1", "gender": "M", "bg_type": "985",
                    "nationality": "中国大陆",
                    "gpa": "3.8", "gpa_scale": "4",
                    "gre": "332", "toefl": "112", "major": "数学",
                    "intern_desc": "2段量化实习", "has_paper": "是",
                    "has_research": "是", "courses_note": "",
                    "program": "baruch-mfe", "result": "accepted",
                    "season": "2025Fall", "source": "quantnet",
                },
            ],
            str(tmp_path),
        )
        records = load_admission_csv(path)
        assert len(records) == 1
        assert records[0].result == "accepted"
        assert records[0].gpa_normalized == 3.8
        assert records[0].bg_tier == 2
        assert records[0].gre == 332
        assert records[0].gender == "M"
        assert records[0].nationality_canonical == "china"

    def test_skips_pending(self, tmp_path):
        path = self._write_csv(
            [
                {
                    "id": "1", "gender": "", "bg_type": "985",
                    "nationality": "",
                    "gpa": "3.8", "gpa_scale": "4",
                    "gre": "", "toefl": "", "major": "", "intern_desc": "",
                    "has_paper": "", "has_research": "", "courses_note": "",
                    "program": "baruch-mfe", "result": "pending",
                    "season": "", "source": "",
                },
            ],
            str(tmp_path),
        )
        records = load_admission_csv(path)
        assert len(records) == 0

    def test_normalizes_100_scale(self, tmp_path):
        path = self._write_csv(
            [
                {
                    "id": "1", "gender": "F", "bg_type": "211",
                    "nationality": "美籍",
                    "gpa": "91.8", "gpa_scale": "100",
                    "gre": "331", "toefl": "110+", "major": "金工",
                    "intern_desc": "", "has_paper": "不明", "has_research": "不明",
                    "courses_note": "", "program": "cmu-mscf", "result": "accepted",
                    "season": "2025Fall", "source": "test",
                },
            ],
            str(tmp_path),
        )
        records = load_admission_csv(path)
        assert len(records) == 1
        assert 3.7 <= records[0].gpa_normalized <= 3.9
        assert records[0].toefl == 110  # stripped '+'
        assert records[0].has_paper is None  # '不明' -> None
        assert records[0].gender == "F"
        assert records[0].nationality_canonical == "domestic"

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_admission_csv("/nonexistent/path.csv")


# ===================================================================
# Program statistics
# ===================================================================


class TestComputeStats:
    """Tests for compute_program_stats() and compute_all_program_stats()."""

    def _make_records(self) -> list[AdmissionRecord]:
        return [
            AdmissionRecord(
                id="1", gender="M", nationality_canonical="china",
                program="baruch-mfe", result="accepted",
                gpa_normalized=3.8, gre=332, bg_tier=2, intern_score=7.0,
                has_paper=True, has_research=True,
            ),
            AdmissionRecord(
                id="2", gender="F", nationality_canonical="domestic",
                program="baruch-mfe", result="accepted",
                gpa_normalized=3.9, gre=335, bg_tier=1, intern_score=8.0,
                has_paper=True, has_research=True,
            ),
            AdmissionRecord(
                id="3", gender="M", nationality_canonical="china",
                program="baruch-mfe", result="rejected",
                gpa_normalized=3.5, gre=325, bg_tier=4, intern_score=2.0,
                has_paper=False, has_research=False,
            ),
            AdmissionRecord(
                id="4", gender="M", nationality_canonical="china",
                program="cmu-mscf", result="accepted",
                gpa_normalized=3.85, gre=333, bg_tier=2, intern_score=6.0,
                has_paper=None, has_research=None,
            ),
        ]

    def test_program_stats_basic(self):
        records = self._make_records()
        stats = compute_program_stats(records, "baruch-mfe")
        assert stats.total_records == 3
        assert stats.accepted == 2
        assert stats.rejected == 1
        assert stats.observed_acceptance_rate == pytest.approx(2 / 3, rel=0.01)
        assert stats.avg_gpa_accepted == pytest.approx(3.85, rel=0.01)

    def test_empty_program(self):
        records = self._make_records()
        stats = compute_program_stats(records, "nonexistent")
        assert stats.total_records == 0

    def test_all_program_stats(self):
        records = self._make_records()
        all_stats = compute_all_program_stats(records)
        assert "baruch-mfe" in all_stats
        assert "cmu-mscf" in all_stats
        assert all_stats["baruch-mfe"].accepted == 2

    def test_feature_importance_computed(self):
        records = self._make_records()
        stats = compute_program_stats(records, "baruch-mfe")
        # Should have feature importance because there are accepted and rejected
        assert len(stats.feature_importance) > 0
        assert "gpa" in stats.feature_importance
        assert "gender_f" in stats.feature_importance
        assert "domestic" in stats.feature_importance

    def test_gender_stats(self):
        records = self._make_records()
        stats = compute_program_stats(records, "baruch-mfe")
        # 2 accepted: 1 M + 1 F -> female_rate = 0.5
        assert stats.female_rate_accepted == pytest.approx(0.5, rel=0.01)

    def test_nationality_dist(self):
        records = self._make_records()
        stats = compute_program_stats(records, "baruch-mfe")
        # 2 accepted: 1 china + 1 domestic
        assert stats.nationality_dist_accepted["china"] == 1
        assert stats.nationality_dist_accepted["domestic"] == 1

    def test_summarize(self):
        records = self._make_records()
        summary = summarize_records(records)
        assert summary["total_records"] == 4
        assert "baruch-mfe" in summary["programs"]
        assert "gender_dist" in summary
        assert summary["gender_dist"]["M"] == 3
        assert summary["gender_dist"]["F"] == 1
        assert "nationality_dist" in summary
        assert summary["nationality_dist"]["china"] == 3
