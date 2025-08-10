#!/usr/bin/env python

import argparse

import hdbscan
import litellm
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"
SENTENCE_RETRIEVAL_MODEL = "anthropic/claude-3-5-haiku-20241022"
CL_GENERATION_MODEL = "anthropic/claude-sonnet-4-20250514"

SENTENCE_RETRIEVAL_PROMPT = """
{content}

Extract sentences from the body of the text above, skipping preambles and salutations.
Output one sentence per line. Copy the sentences exactly as they are.
"""

CL_GENERATION_PROMPT = """
Given these similar sentences:

{sentences}

Please provide:
1. A short category name (2-4 words) that captures the main theme
2. Representative sentences that synthesize the key message

Format your response as:
[category name]
[representative sentence 1]
[representative sentence 2]

Further instructions:
Stay within the vocabulary and information provided.
Do not use new words not present in the text like "delve" or "keen".
Provide only a single representative sentence if possible.
Provide more sentences (max 3) only if there is minimal information overlap between them.
"""


def cluster_sentences(paths: list[str]) -> dict[int, list[str]]:
    model = SentenceTransformer(EMBEDDING_MODEL, device="cpu")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=2,
        cluster_selection_method="leaf",
        cluster_selection_epsilon=0.01,
        metric="euclidean",
    )

    sentences = []
    for path in paths:
        with open(path) as f:
            content = f.read()

        response = litellm.completion(
            model=SENTENCE_RETRIEVAL_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": SENTENCE_RETRIEVAL_PROMPT.format(content=content),
                }
            ],
        )

        extracted_sentences = response.content[0].text.strip().split("\n")
        sentences.extend(k for sentence in extracted_sentences if (k := sentence.strip()))

    embeddings = model.encode(sentences, convert_to_tensor=True)
    cluster_labels = clusterer.fit_predict(embeddings)

    clusters = {}
    for label, sentence in zip(cluster_labels, sentences):
        clusters.setdefault(label, []).append(sentence)

    return clusters


def synthesize_cluster(sentences: list[str]) -> tuple[str, list[str]]:
    """Synthesize a representative category and sentences for a cluster."""

    response = litellm.completion(
        model=CL_GENERATION_MODEL,
        temperature=0.3,
        messages=[
            {
                "role": "user",
                "content": CL_GENERATION_PROMPT.format(sentences="\n".join(sentences)),
            },
        ],
    )

    content = response.content[0].text
    if content:
        content = content.strip()

        # Parse the response
        category, *rep_sentences = content.split("\n")

        if category and rep_sentences:
            rep_sentences = [rs.strip() for rs in rep_sentences]
            return category.strip(), rep_sentences
    raise Exception


def format_cluster(category: str, representatives: list[str], sentences: list[str]) -> str:
    """Format the output for a cluster."""
    reps = "\n".join(representatives)
    snts = "\n".join(f"  {snt}" for snt in sentences)
    return f"{category}:\n{reps}\n{snts}\n"


def parse_args():
    parser = argparse.ArgumentParser(description="Cluster and classify sentences.")
    parser.add_argument(
        "path",
        nargs="+",
        help="Path(s) to the input file(s) with clustered sentences",
    )
    parser.add_argument("-o", "--output", help="Output file path (default: stdout)")

    return parser.parse_args()


def main():
    args = parse_args()
    clusters = cluster_sentences(args.paths)

    # Process each cluster
    results = []
    for val, cluster in clusters.items():
        if not cluster:
            continue

        if val == -1:
            category = "Unclustered"
            representatives = cluster
            res = format_cluster(category, [], representatives)
        else:
            category, representatives = synthesize_cluster(cluster)
            res = format_cluster(category, representatives, cluster)

        results.append(res)

    return results


if __name__ == "__main__":
    results = main()
    print("\n".join(results))
