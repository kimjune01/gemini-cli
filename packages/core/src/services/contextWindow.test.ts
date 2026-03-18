/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, it, expect } from 'vitest';
import {
  Forest,
  ContextWindow,
  cosineSimilarity,
  findClosestPair,
  type Embedder,
  type Summarizer,
} from './contextWindow.js';

// -- Helpers --

/** Stub embedder: one-hot encoding based on first char code. */
const stubEmbedder: Embedder = {
  embed(text: string): number[] {
    const vec = new Array(128).fill(0);
    if (text.length > 0) {
      vec[text.charCodeAt(0) % 128] = 1;
    }
    return vec;
  },
};

/** Stub summarizer: joins messages with ' | '. */
const stubSummarizer: Summarizer = {
  async summarize(messages: string[]): Promise<string> {
    return messages.join(' | ');
  },
};

// -- cosineSimilarity --

describe('cosineSimilarity', () => {
  it('should return 1 for identical vectors', () => {
    expect(cosineSimilarity([1, 0, 0], [1, 0, 0])).toBeCloseTo(1.0);
  });

  it('should return 0 for orthogonal vectors', () => {
    expect(cosineSimilarity([1, 0], [0, 1])).toBeCloseTo(0.0);
  });

  it('should return 0 when a vector is zero', () => {
    expect(cosineSimilarity([0, 0], [1, 1])).toBe(0.0);
  });

  it('should handle negative values', () => {
    expect(cosineSimilarity([1, 0], [-1, 0])).toBeCloseTo(-1.0);
  });
});

// -- Forest --

describe('Forest', () => {
  it('should insert a message as a singleton', async () => {
    const forest = new Forest(stubEmbedder, stubSummarizer);
    const id = forest.insert(0, 'hello');
    expect(id).toBe(0);
    expect(forest.find(0)).toBe(0);
    expect(forest.size()).toBe(1);
    expect(forest.clusterCount()).toBe(1);
  });

  it('should find root with path compression', async () => {
    const forest = new Forest(stubEmbedder, stubSummarizer);
    forest.insert(0, 'a');
    forest.insert(1, 'b');
    forest.insert(2, 'c');

    // Manually chain: 2 -> 1 -> 0
    await forest.union(0, 1);
    await forest.union(0, 2);

    // After path compression, find(2) should return root directly
    const root = forest.find(2);
    expect(root).toBe(forest.find(0));
  });

  it('should union by rank', async () => {
    const forest = new Forest(stubEmbedder, stubSummarizer);
    forest.insert(0, 'a');
    forest.insert(1, 'b');
    forest.insert(2, 'c');

    await forest.union(0, 1);
    const root01 = forest.find(0);

    await forest.union(root01, 2);
    // root01 had higher rank, so it should stay root
    expect(forest.find(2)).toBe(root01);
  });

  it('should generate summary on union', async () => {
    const forest = new Forest(stubEmbedder, stubSummarizer);
    forest.insert(0, 'alpha');
    forest.insert(1, 'beta');
    const root = await forest.union(0, 1);
    const summary = forest.summary(root);
    expect(summary).toContain('alpha');
    expect(summary).toContain('beta');
  });

  it('should update centroid on union', async () => {
    const embedder: Embedder = {
      embed(text: string): number[] {
        return text === 'a' ? [1, 0] : [0, 1];
      },
    };
    const forest = new Forest(embedder, stubSummarizer);
    forest.insert(0, 'a');
    forest.insert(1, 'b');
    await forest.union(0, 1);
    // Centroid should be average: [0.5, 0.5]
    const root = forest.find(0);
    const roots = forest.nearest([0.5, 0.5], 1);
    expect(roots).toContain(root);
  });

  it('should return no-op for union of same cluster', async () => {
    const forest = new Forest(stubEmbedder, stubSummarizer);
    forest.insert(0, 'a');
    const root = await forest.union(0, 0);
    expect(root).toBe(0);
    expect(forest.clusterCount()).toBe(1);
  });

  it('should compact a singleton to its content', () => {
    const forest = new Forest(stubEmbedder, stubSummarizer);
    forest.insert(0, 'hello world');
    expect(forest.compact(0)).toBe('hello world');
  });

  it('should compact a merged cluster to its summary', async () => {
    const forest = new Forest(stubEmbedder, stubSummarizer);
    forest.insert(0, 'foo');
    forest.insert(1, 'bar');
    const root = await forest.union(0, 1);
    expect(forest.compact(root)).toContain('foo');
    expect(forest.compact(root)).toContain('bar');
  });

  it('should expand a cluster to source messages', async () => {
    const forest = new Forest(stubEmbedder, stubSummarizer);
    forest.insert(0, 'x');
    forest.insert(1, 'y');
    await forest.union(0, 1);
    const root = forest.find(0);
    const expanded = forest.expand(root);
    expect(expanded).toContain('x');
    expect(expanded).toContain('y');
  });

  it('should retrieve nearest roots by cosine similarity', () => {
    const embedder: Embedder = {
      embed(text: string): number[] {
        if (text.startsWith('cat')) return [1, 0, 0];
        if (text.startsWith('dog')) return [0.9, 0.1, 0];
        return [0, 0, 1];
      },
    };
    const forest = new Forest(embedder, stubSummarizer);
    forest.insert(0, 'cat food');
    forest.insert(1, 'dog park');
    forest.insert(2, 'javascript');

    const results = forest.nearest([1, 0, 0], 2);
    expect(results).toHaveLength(2);
    // 'cat food' should be closest
    expect(results[0]).toBe(0);
  });

  it('should filter by min_sim in nearest', () => {
    const embedder: Embedder = {
      embed(text: string): number[] {
        return text === 'match' ? [1, 0] : [0, 1];
      },
    };
    const forest = new Forest(embedder, stubSummarizer);
    forest.insert(0, 'match');
    forest.insert(1, 'other');

    // Query for [1, 0] with high min_sim
    const results = forest.nearest([1, 0], 5, 0.9);
    expect(results).toHaveLength(1);
    expect(results[0]).toBe(0);
  });

  it('should return nearestRoot', () => {
    const embedder: Embedder = {
      embed(text: string): number[] {
        return text === 'a' ? [1, 0] : [0, 1];
      },
    };
    const forest = new Forest(embedder, stubSummarizer);
    forest.insert(0, 'a');
    forest.insert(1, 'b');

    const result = forest.nearestRoot([1, 0]);
    expect(result).not.toBeNull();
    expect(result![0]).toBe(0);
    expect(result![1]).toBeCloseTo(1.0);
  });

  it('should return null for nearestRoot on empty forest', () => {
    const forest = new Forest(stubEmbedder, stubSummarizer);
    expect(forest.nearestRoot([1, 0])).toBeNull();
  });

  it('should list members of a cluster', async () => {
    const forest = new Forest(stubEmbedder, stubSummarizer);
    forest.insert(0, 'a');
    forest.insert(1, 'b');
    await forest.union(0, 1);
    const root = forest.find(0);
    const members = forest.members(root);
    expect(members).toContain(0);
    expect(members).toContain(1);
  });

  it('should list all roots', async () => {
    const forest = new Forest(stubEmbedder, stubSummarizer);
    forest.insert(0, 'a');
    forest.insert(1, 'b');
    forest.insert(2, 'c');
    await forest.union(0, 1);
    const roots = forest.roots();
    expect(roots).toHaveLength(2);
  });

  it('should sort members by timestamp in summary', async () => {
    const recorder: string[][] = [];
    const recSummarizer: Summarizer = {
      async summarize(messages: string[]): Promise<string> {
        recorder.push([...messages]);
        return messages.join('; ');
      },
    };
    const forest = new Forest(stubEmbedder, recSummarizer);
    forest.insert(0, 'second', undefined, '2024-01-02');
    forest.insert(1, 'first', undefined, '2024-01-01');
    await forest.union(0, 1);
    // The summarizer should have received them in chronological order
    expect(recorder[0][0]).toContain('first');
    expect(recorder[0][1]).toContain('second');
  });
});

// -- findClosestPair --

describe('findClosestPair', () => {
  it('should return null for fewer than 2 roots', () => {
    const forest = new Forest(stubEmbedder, stubSummarizer);
    forest.insert(0, 'a');
    expect(findClosestPair(forest)).toBeNull();
  });

  it('should find the closest pair', () => {
    const embedder: Embedder = {
      embed(text: string): number[] {
        if (text === 'a') return [1, 0, 0];
        if (text === 'b') return [0.95, 0.05, 0];
        return [0, 0, 1];
      },
    };
    const forest = new Forest(embedder, stubSummarizer);
    forest.insert(0, 'a');
    forest.insert(1, 'b');
    forest.insert(2, 'c');

    const pair = findClosestPair(forest);
    expect(pair).not.toBeNull();
    // a and b are closest
    expect(pair).toContain(0);
    expect(pair).toContain(1);
  });
});

// -- ContextWindow --

describe('ContextWindow', () => {
  it('should keep messages in hot zone when under capacity', async () => {
    const cw = new ContextWindow(stubEmbedder, stubSummarizer, {
      hotSize: 5,
    });
    await cw.append('msg1');
    await cw.append('msg2');
    expect(cw.hotCount).toBe(2);
    expect(cw.coldClusterCount).toBe(0);
  });

  it('should graduate oldest messages to cold zone', async () => {
    const cw = new ContextWindow(stubEmbedder, stubSummarizer, {
      hotSize: 2,
      maxColdClusters: 10,
      mergeThreshold: 0.0, // never merge
    });

    await cw.append('msg1');
    await cw.append('msg2');
    await cw.append('msg3'); // msg1 graduates

    expect(cw.hotCount).toBe(2);
    expect(cw.coldClusterCount).toBe(1);
    expect(cw.totalMessages).toBe(3);
  });

  it('should merge graduated message into nearest cluster when similar', async () => {
    // Use an embedder that makes all messages identical
    const sameEmbedder: Embedder = {
      embed(): number[] {
        return [1, 0, 0];
      },
    };

    const cw = new ContextWindow(sameEmbedder, stubSummarizer, {
      hotSize: 2,
      maxColdClusters: 10,
      mergeThreshold: 0.5, // will merge since similarity is 1.0
    });

    await cw.append('a');
    await cw.append('b');
    await cw.append('c'); // 'a' graduates as singleton
    await cw.append('d'); // 'b' graduates, merges with 'a' (sim = 1.0)

    expect(cw.coldClusterCount).toBe(1); // merged into one cluster
  });

  it('should enforce hard cap on cold clusters via forced merging', async () => {
    // Each message gets a unique embedding so nothing merges naturally
    let counter = 0;
    const uniqueEmbedder: Embedder = {
      embed(): number[] {
        const vec = new Array(10).fill(0);
        vec[counter % 10] = 1;
        counter++;
        return vec;
      },
    };

    const cw = new ContextWindow(uniqueEmbedder, stubSummarizer, {
      hotSize: 2,
      maxColdClusters: 3,
      mergeThreshold: 2.0, // never merge naturally (sim max is 1.0)
    });

    // Add 7 messages: 2 stay hot, 5 graduate
    for (let i = 0; i < 7; i++) {
      await cw.append(`msg${i}`);
    }

    expect(cw.hotCount).toBe(2);
    // 5 graduated but max is 3, so forced merges bring it to <= 3
    expect(cw.coldClusterCount).toBeLessThanOrEqual(3);
  });

  it('should render all cold summaries + hot messages without query', async () => {
    const cw = new ContextWindow(stubEmbedder, stubSummarizer, {
      hotSize: 2,
      maxColdClusters: 10,
      mergeThreshold: 0.0,
    });

    await cw.append('old1');
    await cw.append('old2');
    await cw.append('hot1');
    await cw.append('hot2');

    const rendered = cw.render();
    // Should contain cold summaries and hot messages
    expect(rendered.length).toBeGreaterThanOrEqual(2);
    // Hot messages should be at the end
    expect(rendered[rendered.length - 1]).toBe('hot2');
    expect(rendered[rendered.length - 2]).toBe('hot1');
  });

  it('should render with query-based retrieval', async () => {
    const embedder: Embedder = {
      embed(text: string): number[] {
        if (text.includes('cat')) return [1, 0, 0];
        if (text.includes('dog')) return [0.9, 0.1, 0];
        return [0, 0, 1];
      },
    };

    const cw = new ContextWindow(embedder, stubSummarizer, {
      hotSize: 1,
      maxColdClusters: 10,
      mergeThreshold: 0.0,
    });

    await cw.append('cat info');
    await cw.append('javascript info');
    await cw.append('hot message');

    // Query about cats should retrieve cat cluster
    const rendered = cw.render('cat question', 1, 0.5);
    expect(rendered.some((r) => r.includes('cat'))).toBe(true);
  });

  it('should return correct counts', async () => {
    const cw = new ContextWindow(stubEmbedder, stubSummarizer, {
      hotSize: 3,
    });

    await cw.append('a');
    await cw.append('b');
    expect(cw.hotCount).toBe(2);
    expect(cw.coldClusterCount).toBe(0);
    expect(cw.totalMessages).toBe(2);
  });

  it('should expose forest for direct access', () => {
    const cw = new ContextWindow(stubEmbedder, stubSummarizer);
    expect(cw.forest).toBeInstanceOf(Forest);
  });

  it('should expand a cold cluster to source messages', async () => {
    const cw = new ContextWindow(stubEmbedder, stubSummarizer, {
      hotSize: 2,
      maxColdClusters: 10,
      mergeThreshold: 0.0,
    });

    await cw.append('graduated');
    await cw.append('h1');
    await cw.append('h2');

    const roots = cw.forest.roots();
    expect(roots.length).toBe(1);
    const expanded = cw.expand(roots[0]);
    expect(expanded).toContain('graduated');
  });
});
