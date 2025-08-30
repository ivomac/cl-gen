#!/usr/bin/env python

import subprocess as sp
import sys
from pathlib import Path

CLUSTERS_FILE = Path(__file__).parent / "clusters.md"

EMBED_MODEL = "Qwen/Qwen3-Embedding-0.6B"
GEN_MODEL = "anthropic/claude-sonnet-4-20250514"

PROMPT = r"""
{clusters}

Given the information above, write a cover letter for the job post at the end of this message.

The format should be:
[recipient name]    % For example: Google Hiring Team

[paragraph 1]

[paragraph 2]

Further instructions:
- Output only the recipient name and text body, in plain text.
- Skip greeting (Dear ...,) and closing (Best regards, ...) lines.
- You can write comments preceded with "% " with thoughts and opinions.
- Write a maximum of two paragraphs.
- Be concise and not overly formal.
- Stay within the vocabulary and information provided.
- Do not use punctuation or new words not present in the text like:
  * "delve"
  * "keen"
  * "thrive"
  * "eager"
- Avoid nominalization and "zombie nouns":
  * "I look forward to the possibility of contributing..."
  * "I am excited to apply for the opportunity to work..."
  * "...further strengthening my ability to adapt..."
  * "I possess the ability to provide..."
- Fill in "____" placeholders if used.
- Choose between options "Option 1/Option 2/...".

Job Post:

{job}
"""

TEMPLATE = r"""
Dear {recipient},

{body}

Sincerely,
Ivo Aguiar Maceira
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
    from sentence_transformers import SentenceTransformer, util

    clusters = CLUSTERS_FILE.read_text().split("\n\n")
    encoder = SentenceTransformer(EMBED_MODEL, device="cpu")

    cluster_scores = (
        util.cos_sim(
            encoder.encode(clusters),
            encoder.encode(job_desc),
        )
        .squeeze()
        .tolist()
    )

    sorted_clusters = [
        cluster for cluster, _ in sorted(zip(clusters, cluster_scores), key=lambda x: x[1])
    ]

    query = PROMPT.format(clusters="\n\n".join(sorted_clusters), job=job_desc)

    response = litellm.completion(
        model=GEN_MODEL,
        temperature=0.2,
        messages=[
            {
                "role": "user",
                "content": query,
            },
        ],
    )

    message = response.choices[0].message.content.strip()
    recipient, *paragraphs = message.split("\n\n")
    print(TEMPLATE.format(recipient=recipient, body="\n\n".join(paragraphs)))


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
