from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class UserProfile:
    work_history: List[str] = field(default_factory=list)
    education: List[str] = field(default_factory=list)
    skills: List[str] = field(default_factory=list)


class MemoryStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.data = {"profile": asdict(UserProfile()), "qa": {}}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            loaded = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return
        if isinstance(loaded, dict):
            self.data["profile"] = loaded.get("profile", self.data["profile"])
            self.data["qa"] = loaded.get("qa", {})

    def _save(self) -> None:
        self.path.write_text(json.dumps(self.data, indent=2), encoding="utf-8")

    def get_profile(self) -> UserProfile:
        profile = self.data.get("profile", {})
        return UserProfile(
            work_history=list(profile.get("work_history", [])),
            education=list(profile.get("education", [])),
            skills=list(profile.get("skills", [])),
        )

    def update_profile(self, profile: UserProfile) -> None:
        existing = self.get_profile()
        merged = UserProfile(
            work_history=_merge_unique(existing.work_history, profile.work_history),
            education=_merge_unique(existing.education, profile.education),
            skills=_merge_unique(existing.skills, profile.skills),
        )
        self.data["profile"] = asdict(merged)
        self._save()

    def get_answer(self, question: str) -> Optional[str]:
        return self.data.get("qa", {}).get(question.strip().lower())

    def remember_answer(self, question: str, answer: str) -> None:
        normalized = question.strip().lower()
        if not normalized:
            return
        self.data.setdefault("qa", {})[normalized] = answer
        self._save()


def _merge_unique(old_values: List[str], new_values: List[str]) -> List[str]:
    seen = {value.strip().lower() for value in old_values if value.strip()}
    merged = [value for value in old_values if value.strip()]
    for value in new_values:
        cleaned = value.strip()
        if not cleaned or cleaned.lower() in seen:
            continue
        seen.add(cleaned.lower())
        merged.append(cleaned)
    return merged


class CVAnalyzer:
    SECTION_KEYS = {
        "work": "work_history",
        "work history": "work_history",
        "experience": "work_history",
        "education": "education",
        "skills": "skills",
    }

    @classmethod
    def parse(cls, cv_text: str) -> UserProfile:
        profile = UserProfile()
        current_field: Optional[str] = None

        for raw_line in cv_text.splitlines():
            line = raw_line.strip()
            if not line:
                continue

            possible_header = line.rstrip(":").lower()
            if possible_header in cls.SECTION_KEYS:
                current_field = cls.SECTION_KEYS[possible_header]
                continue

            if ":" in line:
                header, remainder = line.split(":", 1)
                mapped = cls.SECTION_KEYS.get(header.strip().lower())
                if mapped:
                    current_field = mapped
                    entries = _split_values(remainder)
                    getattr(profile, mapped).extend(entries)
                    continue

            if current_field:
                getattr(profile, current_field).extend(_split_values(line))

        return profile


def _split_values(line: str) -> List[str]:
    stripped = line.strip("-• ")
    if not stripped:
        return []
    if "," in stripped:
        return [segment.strip() for segment in stripped.split(",") if segment.strip()]
    return [stripped]


class JobSearchEngine:
    @classmethod
    def search(cls, jobs: List[Dict[str, str]], desired_types: List[str]) -> List[Dict[str, str]]:
        normalized_desired = {_normalize_job_type(job_type) for job_type in desired_types if job_type}
        if not normalized_desired:
            normalized_desired = {"full-time", "part-time", "internship"}

        matches = []
        for job in jobs:
            job_type = _normalize_job_type(job.get("type", ""))
            if job_type in normalized_desired:
                matches.append(job)
        return matches


def _normalize_job_type(job_type: str) -> str:
    normalized = job_type.strip().lower().replace("_", " ")
    # Deliberately map common user input variants/misspellings to canonical values.
    if normalized == "intership" or normalized == "internships":
        return "internship"
    if normalized == "full time":
        return "full-time"
    if normalized == "part time":
        return "part-time"
    return normalized


class DocumentTailor:
    @staticmethod
    def tailor_resume(profile: UserProfile, job_description: str) -> str:
        skills = ", ".join(profile.skills[:5]) if profile.skills else "problem solving and collaboration"
        experience_line = profile.work_history[0] if profile.work_history else "hands-on project experience"
        education_line = profile.education[0] if profile.education else "a strong academic foundation"
        return (
            f"Professional Summary: I bring {experience_line} and {education_line}. "
            f"My strongest skills include {skills}, which align with this role: {job_description.strip()[:220]}."
        )

    @staticmethod
    def tailor_cover_letter(profile: UserProfile, job: Dict[str, str]) -> str:
        title = job.get("title", "this role")
        company = job.get("company", "your team")
        skills = ", ".join(profile.skills[:3]) if profile.skills else "communication and ownership"
        experience = profile.work_history[0] if profile.work_history else "relevant project work"
        return (
            f"Dear Hiring Team at {company},\n\n"
            f"I am excited to apply for the {title} position. Over time, I have built experience through {experience}, "
            f"and I regularly rely on {skills} to deliver dependable results.\n\n"
            "I am motivated by meaningful work, enjoy learning quickly, and collaborate well across teams. "
            "I would value the opportunity to contribute to your goals and grow with your organization.\n\n"
            "Sincerely,\n"
            "Candidate"
        )


class ApplicationBot:
    def __init__(self, memory_path: Path = Path(".bot_memory.json")) -> None:
        self.memory = MemoryStore(memory_path)

    def analyze_cv(self, cv_text: str) -> UserProfile:
        parsed = CVAnalyzer.parse(cv_text)
        self.memory.update_profile(parsed)
        return self.memory.get_profile()

    def search_jobs(self, jobs: List[Dict[str, str]], desired_types: List[str]) -> List[Dict[str, str]]:
        return JobSearchEngine.search(jobs, desired_types)

    def tailor_documents(self, job: Dict[str, str], job_description: str) -> Dict[str, str]:
        profile = self.memory.get_profile()
        return {
            "resume": DocumentTailor.tailor_resume(profile, job_description),
            "cover_letter": DocumentTailor.tailor_cover_letter(profile, job),
        }

    def answer_question(self, question: str) -> str:
        cached = self.memory.get_answer(question)
        if cached:
            return cached

        profile = self.memory.get_profile()
        answer = (
            f"Based on your saved profile, you have {len(profile.work_history)} work entries, "
            f"{len(profile.education)} education entries, and {len(profile.skills)} listed skills."
        )
        self.memory.remember_answer(question, answer)
        return answer


def _load_jobs(path: Path) -> List[Dict[str, str]]:
    jobs = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(jobs, list):
        raise ValueError("Jobs file must contain a JSON array")
    return [job for job in jobs if isinstance(job, dict)]


def main() -> None:
    parser = argparse.ArgumentParser(description="CV-aware job application bot")
    parser.add_argument("--cv", type=Path, required=True, help="Path to CV text file")
    parser.add_argument("--jobs", type=Path, required=True, help="Path to jobs JSON file")
    parser.add_argument(
        "--types",
        nargs="*",
        default=["full-time", "part-time", "internship"],
        help="Desired job types",
    )
    parser.add_argument("--memory", type=Path, default=Path(".bot_memory.json"), help="Memory file path")
    args = parser.parse_args()

    bot = ApplicationBot(memory_path=args.memory)
    profile = bot.analyze_cv(args.cv.read_text(encoding="utf-8"))
    jobs = _load_jobs(args.jobs)
    matches = bot.search_jobs(jobs, args.types)

    print("Parsed profile:")
    print(json.dumps(asdict(profile), indent=2))
    print(f"\nMatching jobs found: {len(matches)}")

    if matches:
        first = matches[0]
        docs = bot.tailor_documents(first, first.get("description", ""))
        print("\nTailored resume snippet:\n")
        print(docs["resume"])
        print("\nTailored cover letter:\n")
        print(docs["cover_letter"])


if __name__ == "__main__":
    main()
