#!/usr/bin/env python

import subprocess as sp
import sys
from pathlib import Path

ROOT = Path(__file__).parent
CLUSTERS_FILE = ROOT / "clusters.md"

EMBED_MODEL = "Qwen/Qwen3-Embedding-0.6B"
GEN_MODEL = "anthropic/claude-opus-4-1-20250805"

SYSTEM_PROMPT_FILE = ROOT / "prompts" / "system.md"

CV_FILE = ROOT / ".." / "CV" / "CV_Ivo_Aguiar_Maceira.tex"

PROMPT = r"""
My Name: Ivo Maceira

My CV:

{CV}

More info about me:

{clusters}

Job Post:

{job}
"""

TEMPLATE = r"""
\documentclass[a4paper,11pt]{{letter}}

\usepackage[scale=0.70]{{geometry}}
\usepackage{{fontspec}}
\usepackage{{helvet}}
\usepackage{{microtype}}
\usepackage[dvipsnames]{{xcolor}}

\setmainfont{{Tex Gyre Heros}}

\begin{{document}}

{letter}

\end{{document}}
"""


def notify(title, message):
    sp.run(
        [
            "notify-send",
            "--app-name=Deepgram",
            "--icon=/usr/share/icons/Papirus/128x128/apps/gtranscribe.svg",
            title,
            message,
        ]
    )


def write_cover_letter(job_desc):
    import litellm

    cv = CV_FILE.read_text()
    system = SYSTEM_PROMPT_FILE.read_text()
    clusters = CLUSTERS_FILE.read_text()

    query = PROMPT.format(CV=cv, clusters=clusters, job=job_desc)

    response = litellm.completion(
        model=GEN_MODEL,
        temperature=0.2,
        messages=[
            {
                "role": "system",
                "content": system,
            },
            {
                "role": "user",
                "content": query,
            },
        ],
    )

    message = response.choices[0].message.content.strip()
    print(TEMPLATE.format(letter=message))


def main():
    job_desc = sys.stdin.read().strip()

    if job_desc:
        notify("Cover Letter", "Generating CL...")

        try:
            write_cover_letter(job_desc)
        except Exception as e:
            notify("Cover Letter Error", f"{e}")
            sys.exit(1)

        notify("Cover Letter", "CL generated!")
    else:
        notify("Cover Letter", "No job description provided.")
        sys.exit(1)


if __name__ == "__main__":
    main()
