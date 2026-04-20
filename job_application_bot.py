from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

MAX_JOB_DESC_LENGTH = 220

# Words ignored when building keyword fingerprints for field-label matching.
_STOP_WORDS = frozenset(
    {
        "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "need", "dare", "ought",
        "used", "what", "your", "you", "my", "me", "i", "we", "they", "it",
        "this", "that", "their", "our", "its", "for", "of", "in", "on", "at",
        "to", "by", "as", "or", "and", "but", "if", "not", "no", "yes",
        "please", "enter", "provide", "write", "fill", "type", "put",
        "field", "question", "section", "form",
    }
)


@dataclass
class UserProfile:
    work_history: List[str] = field(default_factory=list)
    education: List[str] = field(default_factory=list)
    skills: List[str] = field(default_factory=list)


class MemoryStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.data = {"profile": asdict(UserProfile()), "qa": {}, "qa_keywords": []}
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
            self.data["qa_keywords"] = loaded.get("qa_keywords", [])

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
        # Also store with keyword fingerprint for fuzzy matching.
        kw_list = self.data.setdefault("qa_keywords", [])
        keywords = list(_extract_keywords(normalized))
        # Update existing entry if same question was stored before.
        for entry in kw_list:
            if entry.get("question") == normalized:
                entry["answer"] = answer
                entry["keywords"] = keywords
                self._save()
                return
        kw_list.append({"question": normalized, "keywords": keywords, "answer": answer})
        self._save()

    def find_answer_by_keywords(self, field_label: str) -> Optional[str]:
        """Return the stored answer whose keyword fingerprint best matches *field_label*.

        At least one keyword must overlap.  When multiple entries tie, the one
        with the highest overlap count wins.
        """
        label_keywords = _extract_keywords(field_label)
        if not label_keywords:
            return None
        best_answer: Optional[str] = None
        best_overlap = 0
        for entry in self.data.get("qa_keywords", []):
            stored_kw = set(entry.get("keywords", []))
            overlap = len(label_keywords & stored_kw)
            if overlap > best_overlap:
                best_overlap = overlap
                best_answer = entry.get("answer")
        return best_answer if best_overlap > 0 else None


def _extract_keywords(text: str) -> frozenset:
    """Return meaningful lowercase words from *text*, ignoring stop words."""
    words = text.lower().replace("?", " ").replace(":", " ").replace("/", " ").split()
    return frozenset(w.strip(".,()[]") for w in words if len(w) >= 2 and w not in _STOP_WORDS)


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
    if normalized.startswith("inter") and normalized.rstrip("s").endswith("ship"):
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
            f"My strongest skills include {skills}, which align with this role: "
            f"{job_description.strip()[:MAX_JOB_DESC_LENGTH]}."
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


class FormFiller:
    """Maps common job-application field labels to values from a UserProfile."""

    _PROFILE_FIELD_KEYWORDS: Dict[str, str] = {
        "experience": "work_history",
        "work": "work_history",
        "history": "work_history",
        "employment": "work_history",
        "job": "work_history",
        "education": "education",
        "degree": "education",
        "qualification": "education",
        "university": "education",
        "college": "education",
        "school": "education",
        "skill": "skills",
        "skills": "skills",
        "technology": "skills",
        "technologies": "skills",
        "language": "skills",
        "languages": "skills",
        "tool": "skills",
        "tools": "skills",
    }

    @classmethod
    def fill_from_profile(cls, field_label: str, profile: UserProfile) -> Optional[str]:
        """Return a profile value whose category keyword appears in *field_label*."""
        label_lower = field_label.lower()
        for keyword, profile_field in cls._PROFILE_FIELD_KEYWORDS.items():
            if keyword in label_lower:
                values: List[str] = getattr(profile, profile_field, [])
                if values:
                    return ", ".join(values) if profile_field == "skills" else values[0]
        return None


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

    # ------------------------------------------------------------------
    # Form-filling helpers
    # ------------------------------------------------------------------

    def fill_form_field(
        self,
        field_label: str,
        user_input_func: Optional[Callable[[str], str]] = None,
    ) -> Tuple[Optional[str], str]:
        """Return *(value, source)* for a single form field.

        Resolution order:
        1. Profile data   – matched by keyword in the field label.
        2. Memory store   – fuzzy keyword match against previously saved answers.
        3. User prompt    – *user_input_func(field_label)* is called when provided
                            and the answer is saved to memory for future reuse.

        *source* is one of ``"profile"``, ``"memory"``, ``"user"``, or ``"unknown"``
        (when no value is available and no prompt function was given).
        """
        profile = self.memory.get_profile()
        value = FormFiller.fill_from_profile(field_label, profile)
        if value is not None:
            return value, "profile"

        value = self.memory.find_answer_by_keywords(field_label)
        if value is not None:
            return value, "memory"

        if user_input_func is not None:
            value = user_input_func(field_label)
            if value is not None and value.strip():
                self.memory.remember_answer(field_label, value.strip())
                return value.strip(), "user"

        return None, "unknown"

    def fill_form(
        self,
        fields: List[str],
        user_input_func: Optional[Callable[[str], str]] = None,
    ) -> Dict[str, Tuple[Optional[str], str]]:
        """Fill every field in *fields* and return a mapping of
        ``{field_label: (value, source)}``.
        """
        return {label: self.fill_form_field(label, user_input_func) for label in fields}


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

        # Interactive form-fill demo: the bot prompts for unknown fields.
        print("\n--- Form-fill demo (press Enter to skip a field) ---")
        sample_fields = [
            "Expected salary",
            "LinkedIn URL",
            "Work experience summary",
            "Skills / Technologies",
            "Education / Degree",
        ]

        def _prompt(label: str) -> str:
            return input(f"  [Bot] What should I enter for '{label}'? ").strip()

        results = bot.fill_form(sample_fields, user_input_func=_prompt)
        print("\nFilled form:")
        for label, (value, source) in results.items():
            status = f"[{source}]" if value else "[skipped]"
            print(f"  {label}: {value or '(no value)'}  {status}")


if __name__ == "__main__":
    main()
