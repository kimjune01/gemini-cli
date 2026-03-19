/**
 * @license
 * Copyright 2026 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

export interface Embedder {
  embed(text: string): number[];
  embedQuery?(text: string): number[];
}

function tokenize(text: string): string[] {
  return text
    .toLowerCase()
    .split(/[^a-z0-9_./:-]+/i)
    .map((token) => token.trim())
    .filter(Boolean);
}

function l2Normalize(values: number[]): number[] {
  const norm = Math.sqrt(values.reduce((sum, value) => sum + value * value, 0));
  if (norm === 0) {
    return values;
  }
  return values.map((value) => value / norm);
}

export class TFIDFEmbedder implements Embedder {
  private readonly vocab = new Map<string, number>();
  private docCount = 0;
  private readonly termDocFreq = new Map<string, number>();

  embed(text: string): number[] {
    return this.computeVector(text, true);
  }

  embedQuery(text: string): number[] {
    return this.computeVector(text, false);
  }

  private computeVector(text: string, mutateState: boolean): number[] {
    const tokens = tokenize(text);
    if (tokens.length === 0) {
      return new Array<number>(Math.max(this.vocab.size, 1)).fill(0);
    }

    const termCounts = new Map<string, number>();
    for (const token of tokens) {
      termCounts.set(token, (termCounts.get(token) ?? 0) + 1);
    }

    if (mutateState) {
      this.docCount += 1;
      for (const term of new Set(tokens)) {
        if (!this.vocab.has(term)) {
          this.vocab.set(term, this.vocab.size);
        }
        this.termDocFreq.set(term, (this.termDocFreq.get(term) ?? 0) + 1);
      }
    }

    const vector = new Array(this.vocab.size).fill(0);
    const totalTerms = tokens.length;

    for (const [term, count] of termCounts.entries()) {
      const index = this.vocab.get(term);
      if (index === undefined) {
        continue;
      }
      const docFreq = this.termDocFreq.get(term) ?? 0;
      const idf = Math.log((1 + this.docCount) / (1 + docFreq)) + 1;
      vector[index] = (count / totalTerms) * idf;
    }

    return l2Normalize(vector);
  }
}
