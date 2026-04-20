import tempfile
import unittest
from pathlib import Path

from job_application_bot import (
    ApplicationBot,
    CVAnalyzer,
    DocumentTailor,
    FormFiller,
    JobSearchEngine,
    MemoryStore,
    UserProfile,
    _extract_keywords,
    _normalize_job_type,
)


class CVAnalyzerTests(unittest.TestCase):
    def test_parse_extracts_sections(self):
        cv_text = """
        Work History:
        - Software Engineer at ABC
        Education: BS Computer Science
        Skills: Python, SQL, Communication
        """
        profile = CVAnalyzer.parse(cv_text)

        self.assertIn("Software Engineer at ABC", profile.work_history)
        self.assertIn("BS Computer Science", profile.education)
        self.assertEqual(profile.skills, ["Python", "SQL", "Communication"])


class JobSearchTests(unittest.TestCase):
    def test_search_filters_requested_types(self):
        jobs = [
            {"title": "Backend Engineer", "type": "full-time"},
            {"title": "Design Intern", "type": "internship"},
            {"title": "Data Intern", "type": "internship"},
            {"title": "Weekend Support", "type": "part-time"},
            {"title": "Contract QA", "type": "contract"},
        ]

        results = JobSearchEngine.search(jobs, ["internship", "part time"])

        self.assertEqual([job["title"] for job in results], ["Design Intern", "Data Intern", "Weekend Support"])

    def test_normalize_common_type_variants(self):
        self.assertEqual(_normalize_job_type("part time"), "part-time")
        self.assertEqual(_normalize_job_type("interships"), "internship")


class TailoringAndMemoryTests(unittest.TestCase):
    def test_tailored_documents_are_human_like(self):
        profile = UserProfile(
            work_history=["2 years as an automation engineer"],
            education=["BS Computer Science"],
            skills=["Python", "Selenium", "APIs"],
        )
        job = {"title": "QA Engineer", "company": "Acme"}

        cover = DocumentTailor.tailor_cover_letter(profile, job)
        self.assertIn("Dear Hiring Team at Acme", cover)
        self.assertIn("QA Engineer", cover)

    def test_memory_reuses_previous_answer(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            memory_file = Path(temp_dir) / "memory.json"
            bot = ApplicationBot(memory_path=memory_file)
            bot.analyze_cv("Skills: Python")

            first = bot.answer_question("How many skills do I have?")
            second = bot.answer_question("How many skills do I have?")

            self.assertEqual(first, second)


class KeywordExtractionTests(unittest.TestCase):
    def test_strips_stop_words(self):
        keywords = _extract_keywords("What is your expected salary?")
        self.assertIn("expected", keywords)
        self.assertIn("salary", keywords)
        self.assertNotIn("what", keywords)
        self.assertNotIn("your", keywords)

    def test_removes_punctuation(self):
        keywords = _extract_keywords("LinkedIn URL")
        self.assertIn("linkedin", keywords)
        self.assertIn("url", keywords)


class KeywordMemoryTests(unittest.TestCase):
    def test_remember_and_find_by_exact_keywords(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = MemoryStore(Path(temp_dir) / "mem.json")
            store.remember_answer("What is your expected salary?", "50000")
            result = store.find_answer_by_keywords("Expected salary")
            self.assertEqual(result, "50000")

    def test_fuzzy_match_variant_phrasing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = MemoryStore(Path(temp_dir) / "mem.json")
            store.remember_answer("LinkedIn URL", "https://linkedin.com/in/alice")
            # Different phrasing but same keywords should still match.
            result = store.find_answer_by_keywords("Please enter your LinkedIn URL")
            self.assertEqual(result, "https://linkedin.com/in/alice")

    def test_no_match_for_unrelated_label(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = MemoryStore(Path(temp_dir) / "mem.json")
            store.remember_answer("Expected salary", "50000")
            result = store.find_answer_by_keywords("GitHub profile URL")
            self.assertIsNone(result)

    def test_persisted_keywords_survive_reload(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            mem_path = Path(temp_dir) / "mem.json"
            store1 = MemoryStore(mem_path)
            store1.remember_answer("GitHub URL", "https://github.com/alice")
            # Reload from disk.
            store2 = MemoryStore(mem_path)
            result = store2.find_answer_by_keywords("GitHub URL")
            self.assertEqual(result, "https://github.com/alice")


class FormFillerTests(unittest.TestCase):
    def setUp(self):
        self.profile = UserProfile(
            work_history=["Software Engineer at XYZ"],
            education=["BS Computer Science"],
            skills=["Python", "Django"],
        )

    def test_fill_skills_field_from_profile(self):
        value = FormFiller.fill_from_profile("Skills / Technologies", self.profile)
        self.assertIn("Python", value)
        self.assertIn("Django", value)

    def test_fill_education_field_from_profile(self):
        value = FormFiller.fill_from_profile("Education / Degree", self.profile)
        self.assertEqual(value, "BS Computer Science")

    def test_unknown_field_returns_none(self):
        value = FormFiller.fill_from_profile("LinkedIn URL", self.profile)
        self.assertIsNone(value)


class ApplicationBotFormFillTests(unittest.TestCase):
    def test_fill_profile_field_auto(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            bot = ApplicationBot(memory_path=Path(temp_dir) / "mem.json")
            bot.analyze_cv("Skills: Python, Django\nEducation: BS CS\nWork History:\n- Dev at TechCorp")
            value, source = bot.fill_form_field("Skills / Technologies")
            self.assertEqual(source, "profile")
            self.assertIn("Python", value)

    def test_fill_unknown_field_prompts_and_saves_to_memory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            bot = ApplicationBot(memory_path=Path(temp_dir) / "mem.json")
            call_count = {"n": 0}

            def mock_input(label):
                call_count["n"] += 1
                return "https://linkedin.com/in/test"

            value, source = bot.fill_form_field("LinkedIn URL", user_input_func=mock_input)
            self.assertEqual(source, "user")
            self.assertEqual(value, "https://linkedin.com/in/test")
            self.assertEqual(call_count["n"], 1)

            # Second time should use memory, NOT call the prompt again.
            value2, source2 = bot.fill_form_field("Enter your LinkedIn URL", user_input_func=mock_input)
            self.assertEqual(source2, "memory")
            self.assertEqual(value2, "https://linkedin.com/in/test")
            self.assertEqual(call_count["n"], 1)  # prompt was NOT called again

    def test_fill_form_returns_all_fields(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            bot = ApplicationBot(memory_path=Path(temp_dir) / "mem.json")
            bot.analyze_cv("Skills: Python")
            fields = ["Skills", "LinkedIn URL"]
            results = bot.fill_form(fields, user_input_func=lambda _: "")
            self.assertIn("Skills", results)
            self.assertIn("LinkedIn URL", results)


if __name__ == "__main__":
    unittest.main()
