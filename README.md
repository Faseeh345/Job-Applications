# Job-Applications

A minimal Python bot that can:

- Analyze CV text and extract work history, education, and skills
- Search jobs by type (`full-time`, `part-time`, `internship`)
- Tailor a resume summary and cover letter to a job description
- Save memory (profile + repeated Q&A answers) in a local JSON file

## Run

```bash
python job_application_bot.py --cv /absolute/path/to/cv.txt --jobs /absolute/path/to/jobs.json
```

`jobs.json` must be a JSON array with entries like:

```json
[
  {
    "title": "Software Engineer Intern",
    "company": "Acme",
    "type": "internship",
    "description": "Build APIs and collaborate with product teams"
  }
]
```

By default, memory is stored in `.bot_memory.json`.
