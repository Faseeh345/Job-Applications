import tempfile
import unittest
from pathlib import Path

from job_application_bot import (
    ApplicationBot,
    CVAnalyzer,
    DocumentTailor,
    JobSearchEngine,
    UserProfile,
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


if __name__ == "__main__":
    unittest.main()
