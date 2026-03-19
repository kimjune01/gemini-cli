/**
 * @license
 * Copyright 2026 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, expect, it } from 'vitest';
import { TFIDFEmbedder } from './embeddingService.js';

describe('TFIDFEmbedder', () => {
  it('returns a stable zero vector for empty inputs without mutating state', () => {
    const embedder = new TFIDFEmbedder();

    expect(embedder.embed('')).toEqual([0]);
    expect(embedder.embedQuery('')).toEqual([0]);

    const alpha = embedder.embed('alpha');
    expect(alpha.length).toBe(1);
  });

  it('does not mutate the corpus when embedding queries', () => {
    const embedder = new TFIDFEmbedder();

    const first = embedder.embed('alpha');
    const query = embedder.embedQuery('beta');
    const second = embedder.embed('alpha');

    expect(first.length).toBe(1);
    expect(query.length).toBe(1);
    expect(second.length).toBe(1);
  });

  it('adds new vocabulary only on document embeddings', () => {
    const embedder = new TFIDFEmbedder();

    embedder.embed('alpha');
    const beta = embedder.embed('beta');

    expect(beta.length).toBe(2);
  });
});
