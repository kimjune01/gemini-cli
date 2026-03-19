/**
 * @license
 * Copyright 2026 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { beforeEach, describe, expect, it, vi } from 'vitest';
import {
  ContextWindow,
  Forest,
  findClosestPair,
  type Summarizer,
} from './contextWindow.js';
import type { Embedder } from './embeddingService.js';

class StubEmbedder implements Embedder {
  embedCalls = 0;
  queryCalls = 0;

  constructor(private readonly vectors: Record<string, number[]>) {}

  embed(text: string): number[] {
    this.embedCalls += 1;
    return [...(this.vectors[text] ?? [0])];
  }

  embedQuery(text: string): number[] {
    this.queryCalls += 1;
    return [...(this.vectors[text] ?? [0])];
  }
}

describe('Forest', () => {
  let embedder: StubEmbedder;
  let summarizer: Summarizer;
  let forest: Forest;

  beforeEach(() => {
    embedder = new StubEmbedder({
      a: [1, 0],
      b: [1, 0],
      c: [0, 1],
      query: [1, 0],
    });
    summarizer = {
      summarize: vi.fn(async (messages: string[]) => messages.join(' | ')),
    };
    forest = new Forest(embedder, summarizer);
  });

  it('compresses paths during find()', () => {
    forest.insert(1, 'a', [1, 0]);
    forest.insert(2, 'b', [1, 0]);
    forest.insert(3, 'c', [0, 1]);

    const root = forest.union(1, 2);
    forest.union(root, 3);

    const message = (
      forest as unknown as { _nodes: Map<number, { _parent: number | null }> }
    )._nodes.get(3);
    forest.find(3);

    expect(message?._parent).toBe(forest.find(3));
  });

  it('returns nearest roots by cosine similarity', () => {
    forest.insert(1, 'a', [1, 0]);
    forest.insert(2, 'c', [0, 1]);

    expect(forest.nearest([1, 0], 1)).toEqual([1]);
    expect(forest.nearestRoot([0, 1])).toEqual([2, 1]);
  });

  it('expands source messages from merged clusters', () => {
    forest.insert(1, 'a', [1, 0]);
    forest.insert(2, 'b', [1, 0]);
    const root = forest.union(1, 2);

    expect(forest.expand(root)).toEqual(['a', 'b']);
  });

  it('guards against stale summaries during in-flight merges', async () => {
    let resolveSummary: ((value: string) => void) | undefined;
    let pendingCall = 0;
    summarizer = {
      summarize: vi.fn((messages: string[]) => {
        pendingCall += 1;
        if (pendingCall === 1) {
          return new Promise<string>((resolve) => {
            resolveSummary = resolve;
          });
        }
        return Promise.resolve(messages.join(' | '));
      }),
    };
    forest = new Forest(embedder, summarizer);

    forest.insert(1, 'a', [1, 0]);
    forest.insert(2, 'b', [1, 0]);
    const root = forest.union(1, 2);

    const resolving = forest.resolveDirty();
    forest.insert(3, 'c', [0.9, 0.1]);
    const nextRoot = forest.union(root, 3);
    resolveSummary?.('stale summary');
    await resolving;

    expect(forest.summary(nextRoot)).toBeUndefined();
    expect(forest.isDirty(nextRoot)).toBe(true);

    await forest.resolveDirty();
    expect(forest.summary(nextRoot)).toContain('a');
  });

  it('leaves failed clusters dirty for retry', async () => {
    summarizer = {
      summarize: vi
        .fn()
        .mockRejectedValueOnce(new Error('boom'))
        .mockResolvedValueOnce('recovered summary'),
    };
    forest = new Forest(embedder, summarizer);

    forest.insert(1, 'a', [1, 0]);
    forest.insert(2, 'b', [1, 0]);
    const root = forest.union(1, 2);

    await forest.resolveDirty();
    expect(forest.isDirty(root)).toBe(true);
    expect(forest.summary(root)).toBeUndefined();

    await forest.resolveDirty();
    expect(forest.isDirty(root)).toBe(false);
    expect(forest.summary(root)).toBe('recovered summary');
  });

  it('preserves timestamps in raw dirty inputs', async () => {
    forest.insert(1, 'a', [1, 0], '2026-03-19T00:00:00.000Z');
    forest.insert(2, 'b', [1, 0], '2026-03-19T00:01:00.000Z');
    forest.union(1, 2);

    await forest.resolveDirty();

    expect(summarizer.summarize).toHaveBeenCalledWith([
      '[2026-03-19T00:00:00.000Z] a',
      '[2026-03-19T00:01:00.000Z] b',
    ]);
  });
});

describe('ContextWindow', () => {
  it('graduates messages into cold storage without summarizing on append', () => {
    const embedder = new StubEmbedder({
      msg1: [1, 0],
      msg2: [1, 0],
      msg3: [1, 0],
    });
    const summarizer = {
      summarize: vi.fn(async (messages: string[]) => messages.join(' | ')),
    };
    const window = new ContextWindow(embedder, summarizer, {
      graduateAt: 2,
      evictAt: 4,
      mergeThreshold: 0.1,
    });

    window.append('msg1');
    window.append('msg2');
    window.append('msg3');

    expect(window.hotCount).toBe(3);
    expect(window.coldClusterCount).toBe(1);
    expect(summarizer.summarize).not.toHaveBeenCalled();
  });

  it('keeps graduated messages in the overlap window', () => {
    const embedder = new StubEmbedder({
      msg0: [1, 0],
      msg1: [1, 0],
      msg2: [1, 0],
      msg3: [1, 0],
    });
    const summarizer = {
      summarize: vi.fn(async (messages: string[]) => messages.join(' | ')),
    };
    const window = new ContextWindow(embedder, summarizer, {
      graduateAt: 3,
      evictAt: 5,
      mergeThreshold: 0.1,
    });

    window.append('msg0');
    window.append('msg1');
    window.append('msg2');
    window.append('msg3');

    expect(window.render()).toContain('msg0');
  });

  it('uses embedQuery for query rendering', () => {
    const embedder = new StubEmbedder({
      alpha: [1, 0],
      beta: [0, 1],
      query: [0, 1],
    });
    const summarizer = {
      summarize: vi.fn(async (messages: string[]) => messages.join(' | ')),
    };
    const window = new ContextWindow(embedder, summarizer, {
      graduateAt: 1,
      evictAt: 2,
      mergeThreshold: 0.1,
    });

    window.append('alpha');
    window.append('beta');
    window.render('query');

    expect(embedder.queryCalls).toBe(1);
    expect(embedder.embedCalls).toBe(2);
  });

  it('batch resolves dirty clusters', async () => {
    const embedder = new StubEmbedder({
      msg0: [1, 0],
      msg1: [1, 0],
      msg2: [0, 1],
      msg3: [0, 1],
    });
    const summarizer = {
      summarize: vi.fn(async (messages: string[]) => messages.join(' | ')),
    };
    const window = new ContextWindow(embedder, summarizer, {
      graduateAt: 1,
      evictAt: 2,
      maxColdClusters: 2,
      mergeThreshold: 0.1,
    });

    window.append('msg0');
    window.append('msg1');
    window.append('msg2');
    window.append('msg3');
    await window.resolveDirty();

    expect(window.forest.dirtyRoots()).toEqual([]);
    expect(summarizer.summarize).toHaveBeenCalled();
  });

  it('rejects invalid overlap configuration', () => {
    const embedder = new StubEmbedder({});
    const summarizer = {
      summarize: vi.fn(async (messages: string[]) => messages.join(' | ')),
    };

    expect(
      () =>
        new ContextWindow(embedder, summarizer, {
          graduateAt: 5,
          evictAt: 4,
        }),
    ).toThrow('evictAt must be greater than or equal to graduateAt');
  });
});

describe('findClosestPair', () => {
  it('finds the most similar pair of roots', () => {
    const embedder = new StubEmbedder({});
    const summarizer = {
      summarize: vi.fn(async (messages: string[]) => messages.join(' | ')),
    };
    const forest = new Forest(embedder, summarizer);
    forest.insert(1, 'a', [1, 0]);
    forest.insert(2, 'b', [0.9, 0.1]);
    forest.insert(3, 'c', [0, 1]);

    expect(findClosestPair(forest)).toEqual([1, 2]);
  });
});
