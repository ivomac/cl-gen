#!/usr/bin/env python

import argparse
import sys
from pathlib import Path

import litellm
from sentence_transformers import SentenceTransformer, util

EMBED_MODEL = "Qwen/Qwen3-Embedding-0.6B"
GEN_MODEL = "anthropic/claude-sonnet-4-20250514"

PROMPT = r"""
{clusters}

Given the information above, write a cover letter for the job post at the end of this message.

The format should be:
[recipient name]    % For example: Google Hiring Team

[paragraph 1]

[paragraph 2]

...

Further instructions:
- Output only the recipient name and text body, in plain text.
- Skip greeting (Dear ...,) and closing (Best regards, ...) lines.
- You can write comments preceded with "% " with thoughts and opinions.
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
\documentclass[11pt,a4paper,sans]{{article}}

\usepackage[scale=0.80]{{geometry}}

\usepackage[dvipsnames]{{xcolor}}

\usepackage{{fontspec}}
\setmainfont{{Tex Gyre Heros}}

\begin{{document}}

Dear {recipient},

{body}

Sincerely, \\
Ivo Aguiar Maceira

\end{{document}}
"""


def write_cover_letter(post, encoder, clusters):
    with open(post) as f:
        job_desc = f.read()

    if not job_desc:
        raise ValueError(f"{post.name} is empty")

    cl_file = post.parent / "CL.tex"
    if cl_file.is_file():
        print(f"CL for {post.name} already exists", file=sys.stderr)
        return

    print(f"Generating CL for {post.name}...", file=sys.stderr)
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

    if response.content and hasattr(response.content[0], "text"):
        message = getattr(response.content[0], "text").strip()
        recipient, *paragraphs = message.split("\n\n")
        letter = TEMPLATE.format(recipient=recipient, body="\n\n".join(paragraphs))
        cl_file.write_text(letter)
    else:
        raise litellm.exceptions.APIError("Unexpected response from API")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--clusters", type=Path, default="./clusters.txt", help="Path to clusters file"
    )
    parser.add_argument("path", nargs="*", type=Path, help="Path(s) to job posting")
    args = parser.parse_args()

    if args.path:
        clusters = args.clusters.read_text().split("\n\n")

        encoder = SentenceTransformer(EMBED_MODEL, device="cpu")
        for path in args.path:
            write_cover_letter(path, encoder, clusters)


if __name__ == "__main__":
    main()
