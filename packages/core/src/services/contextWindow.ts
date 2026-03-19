/**
 * @license
 * Copyright 2026 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import type { Embedder } from './embeddingService.js';

export interface Summarizer {
  summarize(messages: string[]): Promise<string>;
}

export class Message {
  _parent: number | null = null;
  _rank = 0;

  constructor(
    readonly id: number,
    readonly content: string,
    readonly embedding: number[],
    readonly timestamp: string | null = null,
  ) {}
}

export function cosineSimilarity(a: number[], b: number[]): number {
  const dimensions = Math.max(a.length, b.length);
  let dot = 0;
  let normA = 0;
  let normB = 0;

  for (let i = 0; i < dimensions; i++) {
    const av = a[i] ?? 0;
    const bv = b[i] ?? 0;
    dot += av * bv;
    normA += av * av;
    normB += bv * bv;
  }

  if (normA === 0 || normB === 0) {
    return 0;
  }

  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

function mergeCentroids(
  left: number[],
  leftWeight: number,
  right: number[],
  rightWeight: number,
): number[] {
  const dimensions = Math.max(left.length, right.length);
  const totalWeight = leftWeight + rightWeight;
  if (totalWeight === 0) {
    return [];
  }

  const merged: number[] = new Array<number>(dimensions).fill(0);
  for (let i = 0; i < dimensions; i++) {
    const lv = left[i] ?? 0;
    const rv = right[i] ?? 0;
    merged[i] = (lv * leftWeight + rv * rightWeight) / totalWeight;
  }
  return merged;
}

export class Forest {
  private readonly _nodes = new Map<number, Message>();
  private readonly _summaries = new Map<number, string>();
  private readonly _children = new Map<number, number[]>();
  private readonly _centroids = new Map<number, number[]>();
  private readonly _dirtyInputs = new Map<number, string[]>();
  private _resolvePromise: Promise<void> | null = null;

  constructor(
    private readonly _embedder: Embedder,
    private readonly _summarizer: Summarizer,
  ) {}

  insert(
    msgId: number,
    content: string,
    embedding?: number[],
    timestamp?: string,
  ): number {
    const vector = embedding ?? this._embedder.embed(content);
    const message = new Message(msgId, content, vector, timestamp ?? null);
    this._nodes.set(msgId, message);
    this._children.set(msgId, [msgId]);
    this._centroids.set(msgId, [...vector]);
    return msgId;
  }

  find(msgId: number): number {
    const node = this._nodes.get(msgId);
    if (!node) {
      throw new Error(`Unknown message id: ${msgId}`);
    }
    if (node._parent === null) {
      return msgId;
    }
    const root = this.find(node._parent);
    node._parent = root;
    return root;
  }

  union(idA: number, idB: number): number {
    let rootA = this.find(idA);
    let rootB = this.find(idB);
    if (rootA === rootB) {
      return rootA;
    }

    let nodeA = this._nodes.get(rootA)!;
    let nodeB = this._nodes.get(rootB)!;
    if (nodeA._rank < nodeB._rank) {
      [rootA, rootB] = [rootB, rootA];
      [nodeA, nodeB] = [nodeB, nodeA];
    }

    const membersA = this._children.get(rootA) ?? [rootA];
    const membersB = this._children.get(rootB) ?? [rootB];
    const centroidA = this._centroids.get(rootA) ?? [];
    const centroidB = this._centroids.get(rootB) ?? [];
    const dirtyInputs = [
      ...this.getClusterInputs(rootA, membersA),
      ...this.getClusterInputs(rootB, membersB),
    ];

    nodeB._parent = rootA;
    if (nodeA._rank === nodeB._rank) {
      nodeA._rank += 1;
    }

    this._children.set(rootA, [...membersA, ...membersB]);
    this._children.delete(rootB);

    this._centroids.set(
      rootA,
      mergeCentroids(centroidA, membersA.length, centroidB, membersB.length),
    );
    this._centroids.delete(rootB);

    this._summaries.delete(rootA);
    this._summaries.delete(rootB);
    this._dirtyInputs.delete(rootB);
    this._dirtyInputs.set(rootA, dirtyInputs);

    return rootA;
  }

  async resolveDirty(): Promise<void> {
    let ranOwnBatch = false;

    while (true) {
      if (this._resolvePromise) {
        await this._resolvePromise;
      } else {
        this._resolvePromise = this._resolveDirtyBatch();
        ranOwnBatch = true;
        try {
          await this._resolvePromise;
        } finally {
          this._resolvePromise = null;
        }
      }

      if (this.dirtyRoots().length === 0 || ranOwnBatch) {
        return;
      }
    }
  }

  private async _resolveDirtyBatch(): Promise<void> {
    for (const [root, inputs] of [...this._dirtyInputs.entries()]) {
      try {
        if (this.find(root) !== root) {
          continue;
        }
      } catch {
        continue;
      }

      try {
        const summary = (await this._summarizer.summarize(inputs)).trim();
        if (this.find(root) !== root) {
          continue;
        }
        if (this._dirtyInputs.get(root) !== inputs) {
          continue;
        }

        this._dirtyInputs.delete(root);
        if (summary) {
          this._summaries.set(root, summary);
        }
      } catch {
        // Summarization failed; leave the cluster dirty for a later retry.
      }
    }
  }

  compact(rootId: number): string {
    const root = this.find(rootId);
    const summary = this._summaries.get(root);
    if (summary) {
      return summary;
    }
    return this._nodes.get(root)?.content ?? '';
  }

  expand(rootId: number): string[] {
    const root = this.find(rootId);
    return (this._children.get(root) ?? [root])
      .map((id) => this._nodes.get(id)?.content ?? '')
      .filter(Boolean);
  }

  nearest(queryEmbedding: number[], k = 3, minSim = 0): number[] {
    return [...this._children.keys()]
      .map((root) => ({
        root,
        sim: cosineSimilarity(queryEmbedding, this._centroids.get(root) ?? []),
      }))
      .filter((entry) => entry.sim >= minSim)
      .sort((left, right) => right.sim - left.sim)
      .slice(0, k)
      .map((entry) => entry.root);
  }

  nearestRoot(queryEmbedding: number[]): [number, number] | null {
    const roots = this.nearest(queryEmbedding, 1, Number.NEGATIVE_INFINITY);
    if (roots.length === 0) {
      return null;
    }
    const root = roots[0];
    return [
      root,
      cosineSimilarity(queryEmbedding, this._centroids.get(root) ?? []),
    ];
  }

  getCentroid(rootId: number): number[] | undefined {
    return this._centroids.get(this.find(rootId));
  }

  roots(): number[] {
    return [...this._children.keys()];
  }

  members(rootId: number): number[] {
    return [...(this._children.get(this.find(rootId)) ?? [])];
  }

  summary(rootId: number): string | undefined {
    return this._summaries.get(this.find(rootId));
  }

  isDirty(rootId: number): boolean {
    return this._dirtyInputs.has(this.find(rootId));
  }

  dirtyRoots(): number[] {
    return [...this._dirtyInputs.keys()].filter(
      (root) => this.find(root) === root,
    );
  }

  size(): number {
    return this._nodes.size;
  }

  clusterCount(): number {
    return this._children.size;
  }

  private getClusterInputs(rootId: number, memberIds?: number[]): string[] {
    const dirtyInputs = this._dirtyInputs.get(rootId);
    if (dirtyInputs && dirtyInputs.length > 0) {
      return dirtyInputs;
    }

    const summary = this._summaries.get(rootId);
    if (summary) {
      return [summary];
    }

    return (memberIds ?? this._children.get(rootId) ?? [rootId])
      .map((id) => {
        const node = this._nodes.get(id);
        if (!node) {
          return '';
        }
        return node.timestamp
          ? `[${node.timestamp}] ${node.content}`
          : node.content;
      })
      .filter(Boolean);
  }
}

export interface ContextWindowOptions {
  graduateAt?: number;
  evictAt?: number;
  maxColdClusters?: number;
  mergeThreshold?: number;
}

export class ContextWindow {
  private readonly _forest: Forest;
  private readonly _hot: Message[] = [];
  private readonly _graduateAt: number;
  private readonly _evictAt: number;
  private readonly _maxColdClusters: number;
  private readonly _mergeThreshold: number;
  private _nextId = 0;
  private _graduatedIndex = 0;
  private _mergeCount = 0;

  constructor(
    private readonly _embedder: Embedder,
    summarizer: Summarizer,
    options: ContextWindowOptions = {},
  ) {
    this._graduateAt = options.graduateAt ?? 26;
    this._evictAt = options.evictAt ?? 30;
    if (this._evictAt < this._graduateAt) {
      throw new Error(
        `evictAt (${this._evictAt}) must be >= graduateAt (${this._graduateAt})`,
      );
    }
    this._maxColdClusters = options.maxColdClusters ?? 10;
    this._mergeThreshold = options.mergeThreshold ?? 0.15;
    this._forest = new Forest(_embedder, summarizer);
  }

  append(content: string, timestamp?: string): number {
    const id = this._nextId++;
    const message = new Message(
      id,
      content,
      this._embedder.embed(content),
      timestamp ?? null,
    );
    this._hot.push(message);

    while (this._hot.length - this._graduatedIndex > this._graduateAt) {
      if (this._graduate(this._hot[this._graduatedIndex])) {
        this._mergeCount += 1;
      }
      this._graduatedIndex += 1;
    }

    while (this._hot.length > this._evictAt) {
      this._hot.shift();
      if (this._graduatedIndex > 0) {
        this._graduatedIndex -= 1;
      }
    }

    return id;
  }

  render(query?: string | null, k = 3, minSim = 0.05): string[] {
    let coldRoots: number[];
    if (query && this._forest.clusterCount() > 0) {
      const queryEmbedding = this._embedder.embedQuery
        ? this._embedder.embedQuery(query)
        : this._embedder.embed(query);
      coldRoots = this._forest.nearest(queryEmbedding, k, minSim);
    } else {
      coldRoots = this._forest.roots();
    }

    const cold = coldRoots.map((root) => this._forest.compact(root));
    const hot = this._hot.map((message) => message.content);
    return [...cold, ...hot];
  }

  async resolveDirty(): Promise<void> {
    await this._forest.resolveDirty();
  }

  expand(rootId: number): string[] {
    return this._forest.expand(rootId);
  }

  get hotCount(): number {
    return this._hot.length;
  }

  get coldClusterCount(): number {
    return this._forest.clusterCount();
  }

  get totalMessages(): number {
    return this._forest.size() + this._hot.length;
  }

  get forest(): Forest {
    return this._forest;
  }

  drainMergeCount(): number {
    const count = this._mergeCount;
    this._mergeCount = 0;
    return count;
  }

  private _graduate(message: Message): boolean {
    this._forest.insert(
      message.id,
      message.content,
      message.embedding,
      message.timestamp ?? undefined,
    );

    let merged = false;
    const nearest = this.findNearestOtherRoot(message.id, message.embedding);
    if (nearest && nearest[1] >= this._mergeThreshold) {
      this._forest.union(message.id, nearest[0]);
      merged = true;
    }

    while (this._forest.clusterCount() > this._maxColdClusters) {
      const pair = findClosestPair(this._forest);
      if (!pair) {
        break;
      }
      this._forest.union(pair[0], pair[1]);
      merged = true;
    }

    return merged;
  }

  private findNearestOtherRoot(
    selfId: number,
    embedding: number[],
  ): [number, number] | null {
    let bestRoot: number | null = null;
    let bestSim = Number.NEGATIVE_INFINITY;

    for (const root of this._forest.roots()) {
      if (root === selfId) {
        continue;
      }
      const centroid = this._forest.getCentroid(root);
      if (!centroid) {
        continue;
      }
      const sim = cosineSimilarity(embedding, centroid);
      if (sim > bestSim) {
        bestSim = sim;
        bestRoot = root;
      }
    }

    return bestRoot === null ? null : [bestRoot, bestSim];
  }
}

export function findClosestPair(forest: Forest): [number, number] | null {
  const roots = forest.roots();
  if (roots.length < 2) {
    return null;
  }

  let bestPair: [number, number] | null = null;
  let bestSim = Number.NEGATIVE_INFINITY;

  for (let i = 0; i < roots.length; i++) {
    const leftRoot = roots[i];
    const leftCentroid = forest.getCentroid(leftRoot);
    if (!leftCentroid) {
      continue;
    }
    for (let j = i + 1; j < roots.length; j++) {
      const rightRoot = roots[j];
      const rightCentroid = forest.getCentroid(rightRoot);
      if (!rightCentroid) {
        continue;
      }
      const sim = cosineSimilarity(leftCentroid, rightCentroid);
      if (sim > bestSim) {
        bestSim = sim;
        bestPair = [leftRoot, rightRoot];
      }
    }
  }

  return bestPair;
}
