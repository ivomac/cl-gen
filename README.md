# Cover Letter Generator

An AI-powered tool that generates personalized cover letters by clustering and analyzing your previous cover letters, then matching them to job postings.

## Overview

This project consists of two main components:

1. **Sentence Clustering** (`cluster_cover_letters.py`) - Groups similar sentences from your cover letters into thematic clusters
2. **Cover Letter Generation** (`write_cover_letter.py`) - Creates tailored cover letters by matching clustered sentences to job requirements

## Features

- **Flexible Input Processing**: Supports multiple text formats (plain text, structured documents, etc.) - a small LLM extracts sentences from any input format
- **Semantic Clustering**: Uses HDBSCAN and sentence embeddings to group related experiences
- **AI Synthesis**: Generates representative sentences for each skill cluster (using Claude by default)
- **Job Matching**: Ranks clusters by relevance to specific job postings
- **LaTeX Output**: Produces template cover letter

## Setup

Install required dependencies:

```bash
pip install hdbscan litellm sentence-transformers
```

Set up the API key environment variable for your chosen model.

## Usage

### 1. Cluster Your Previous Cover Letters

First, create text files with your experiences in any format (plain text, structured documents, etc.). The tool will extract sentences using a small LLM, then cluster them:

```bash
python cluster_cover_letters.py CL1.txt CL2.tex CL3.md > clusters.md
```

The output file serves as the source of sentences to generate future cover letters.
Most likely a group of sentences will be left out "unlabelled".
Go through the output file and edit it appropriately, adding/removing sentences/clusters as you see fit.

### 2. Generate Cover Letters

Select/Copy the job posting, then generate a cover letter:

```bash
xsel | python write_cover_letter.py | xsel
```

This will copy the generated cover letter to X selection.

## File Structure

- `cluster_cover_letter_sentences.py` - Main clustering script
- `write_cover_letter.py` - Cover letter generation script
- `example_clusters.md` - Example clusters file

## Default Models Used

- **Sentence Extraction**: anthropic/claude-3-5-haiku-20241022 (for processing input text formats)
- **Embedding**: Qwen/Qwen3-Embedding-0.6B (for semantic similarity)
- **Generation**: anthropic/claude-sonnet-4-20250514 (for text synthesis)

