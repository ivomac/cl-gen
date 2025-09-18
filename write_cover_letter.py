#!/usr/bin/env python

import subprocess as sp
import sys
from pathlib import Path

"""
~25% Me, 50% Company, 25% Us
Break from the old practice of “retelling your CV;” let the recruiter read the actual one, and see
your customized letter instead as convincing evidence of your focused efforts to join their team.
Try to weight your approach to show you are serious: based on your research, 55-60% of the text
should be about the company, department, unit or role you are seeking. For spontaneous offers,
use sector trends, media mentions, and/or research-related insights to anchor your arguments.
Use any personal “hook” available to open the letter (“At your recent presentation at EPFL, I was
quite pleased to hear about...” or, “At last October’s Forum at EPFL, your colleague Melissa
Strathberg, mentioned that ABB was developing a new unit in circular energy storage design...”
or, “In last week’s The Economist, the focus of the “Science and Technology” section was entirely
on new wave-based refractors in cell-level research...” etc.)
Since you are a young professional, you can dare to propose innovative or original solutions
and/or improvements in the “Us” section, where you show that you have imagined being part of
the team, and describe what you would contribute in that capacity.
If you’re not yet comfortable with this, however, a more traditional closing is acceptable:
“I am taking the liberty of sending you a copy of my CV, and I would be very pleased to discuss
this exciting opportunity with you in person at your convenience.”
"""

ROOT = Path(__file__).parent
CLUSTERS_FILE = ROOT / "clusters.md"

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
