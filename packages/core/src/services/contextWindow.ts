/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * Union-find context compaction with LLM-generated cluster summaries.
 *
 * E-class roots are cache keys for summaries. Union = cheap LLM merge.
 * Graduation merges into nearest e-class if similar enough, else creates
 * a new singleton. Retrieval embeds a query and returns top-k e-classes.
 */

// -- Interfaces --

export interface Embedder {
  embed(text: string): number[];
}

export interface Summarizer {
  summarize(messages: string[]): Promise<string>;
}

// -- Data structures --

export interface Message {
  id: number;
  content: string;
  embedding: number[];
  timestamp: string | null;
  _parent: number | null;
  _rank: number;
}

// -- Helpers --

export function cosineSimilarity(a: number[], b: number[]): number {
  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  normA = Math.sqrt(normA);
  normB = Math.sqrt(normB);
  if (normA === 0 || normB === 0) return 0.0;
  return dot / (normA * normB);
}

export function findClosestPair(forest: Forest): [number, number] | null {
  const roots = forest.roots();
  if (roots.length < 2) return null;

  let bestSim = -1.0;
  let bestPair: [number, number] = [roots[0], roots[1]];

  for (let i = 0; i < roots.length; i++) {
    const ca = forest.getCentroid(roots[i]);
    if (!ca) continue;
    for (let j = i + 1; j < roots.length; j++) {
      const cb = forest.getCentroid(roots[j]);
      if (!cb) continue;
      const sim = cosineSimilarity(ca, cb);
      if (sim > bestSim) {
        bestSim = sim;
        bestPair = [roots[i], roots[j]];
      }
    }
  }

  return bestPair;
}

// -- Forest --

export class Forest {
  private _nodes: Map<number, Message> = new Map();
  private _summaries: Map<number, string> = new Map();
  private _children: Map<number, number[]> = new Map();
  private _centroids: Map<number, number[]> = new Map();
  private _embedder: Embedder;
  private _summarizer: Summarizer;

  constructor(embedder: Embedder, summarizer: Summarizer) {
    this._embedder = embedder;
    this._summarizer = summarizer;
  }

  insert(
    msgId: number,
    content: string,
    embedding?: number[],
    timestamp?: string | null,
  ): number {
    if (embedding === undefined) {
      embedding = this._embedder.embed(content);
    }
    const msg: Message = {
      id: msgId,
      content,
      embedding,
      timestamp: timestamp ?? null,
      _parent: null,
      _rank: 0,
    };
    this._nodes.set(msgId, msg);
    this._children.set(msgId, [msgId]);
    this._centroids.set(msgId, [...embedding]);
    return msgId;
  }

  find(msgId: number): number {
    const node = this._nodes.get(msgId);
    if (!node) throw new Error(`Node ${msgId} not found`);
    if (node._parent === null) return msgId;
    const root = this.find(node._parent);
    node._parent = root; // path compression
    return root;
  }

  async union(idA: number, idB: number): Promise<number> {
    let rootA = this.find(idA);
    let rootB = this.find(idB);
    if (rootA === rootB) return rootA;

    let nodeA = this._nodes.get(rootA)!;
    let nodeB = this._nodes.get(rootB)!;

    // Union by rank
    if (nodeA._rank < nodeB._rank) {
      [rootA, rootB] = [rootB, rootA];
      [nodeA, nodeB] = [nodeB, nodeA];
    }
    nodeB._parent = rootA;
    if (nodeA._rank === nodeB._rank) {
      nodeA._rank += 1;
    }

    // Merge children lists
    const membersB = this._children.get(rootB) ?? [];
    this._children.delete(rootB);
    const membersA = this._children.get(rootA) ?? [];
    membersA.push(...membersB);
    this._children.set(rootA, membersA);

    // Update centroid (weighted average)
    const ca = this._centroids.get(rootA);
    const cb = this._centroids.get(rootB);
    this._centroids.delete(rootB);

    if (ca && cb) {
      const na = membersA.length - membersB.length;
      const nb = membersB.length;
      const merged = ca.map((v, i) => (v * na + cb[i] * nb) / (na + nb));
      this._centroids.set(rootA, merged);
    }

    // Summarize merged e-class via summarizer
    const memberIds = this._children.get(rootA)!;
    const sortedIds = [...memberIds].sort((a, b) => {
      const ta = this._nodes.get(a)?.timestamp ?? '';
      const tb = this._nodes.get(b)?.timestamp ?? '';
      return ta.localeCompare(tb);
    });

    const memberTexts: string[] = [];
    for (const mid of sortedIds) {
      const node = this._nodes.get(mid)!;
      if (node.timestamp) {
        memberTexts.push(`[${node.timestamp}] ${node.content}`);
      } else {
        memberTexts.push(node.content);
      }
    }
    this._summaries.set(rootA, await this._summarizer.summarize(memberTexts));

    return rootA;
  }

  compact(rootId: number): string {
    const root = this.find(rootId);
    const summary = this._summaries.get(root);
    if (summary === undefined) {
      return this._nodes.get(root)!.content;
    }
    return summary;
  }

  expand(rootId: number): string[] {
    const root = this.find(rootId);
    const memberIds = this._children.get(root) ?? [root];
    return memberIds.map((mid) => this._nodes.get(mid)!.content);
  }

  nearest(
    queryEmbedding: number[],
    k: number = 3,
    minSim: number = 0.0,
  ): number[] {
    const scored: Array<[number, number]> = [];
    for (const root of this._children.keys()) {
      const centroid = this._centroids.get(root);
      if (centroid) {
        const sim = cosineSimilarity(queryEmbedding, centroid);
        if (sim >= minSim) {
          scored.push([sim, root]);
        }
      }
    }
    scored.sort((a, b) => b[0] - a[0]);
    return scored.slice(0, k).map(([, root]) => root);
  }

  nearestRoot(queryEmbedding: number[]): [number, number] | null {
    let bestSim = -1.0;
    let bestRoot: number | null = null;
    for (const root of this._children.keys()) {
      const centroid = this._centroids.get(root);
      if (centroid) {
        const sim = cosineSimilarity(queryEmbedding, centroid);
        if (sim > bestSim) {
          bestSim = sim;
          bestRoot = root;
        }
      }
    }
    if (bestRoot === null) return null;
    return [bestRoot, bestSim];
  }

  roots(): number[] {
    return [...this._children.keys()];
  }

  members(rootId: number): number[] {
    const root = this.find(rootId);
    return [...(this._children.get(root) ?? [root])];
  }

  summary(rootId: number): string | undefined {
    const root = this.find(rootId);
    return this._summaries.get(root);
  }

  size(): number {
    return this._nodes.size;
  }

  clusterCount(): number {
    return this._children.size;
  }

  getCentroid(rootId: number): number[] | undefined {
    return this._centroids.get(rootId);
  }
}

// -- ContextWindow --

export interface ContextWindowOptions {
  hotSize?: number;
  maxColdClusters?: number;
  mergeThreshold?: number;
}

export class ContextWindow {
  private _embedder: Embedder;
  private _forest: Forest;
  private _hot: Message[] = [];
  private _hotSize: number;
  private _maxColdClusters: number;
  private _mergeThreshold: number;
  private _nextId = 0;

  constructor(
    embedder: Embedder,
    summarizer: Summarizer,
    options: ContextWindowOptions = {},
  ) {
    this._embedder = embedder;
    this._forest = new Forest(embedder, summarizer);
    this._hotSize = options.hotSize ?? 30;
    this._maxColdClusters = options.maxColdClusters ?? 10;
    this._mergeThreshold = options.mergeThreshold ?? 0.15;
  }

  async append(content: string, timestamp?: string | null): Promise<number> {
    const msgId = this._nextId++;
    const embedding = this._embedder.embed(content);
    const msg: Message = {
      id: msgId,
      content,
      embedding,
      timestamp: timestamp ?? null,
      _parent: null,
      _rank: 0,
    };
    this._hot.push(msg);

    while (this._hot.length > this._hotSize) {
      const graduated = this._hot.shift()!;
      await this._graduate(graduated);
    }

    return msgId;
  }

  private async _graduate(msg: Message): Promise<void> {
    this._forest.insert(msg.id, msg.content, msg.embedding, msg.timestamp);

    if (this._forest.clusterCount() <= 1) return;

    // Find nearest existing e-class (excluding the singleton we just inserted)
    const match = this._forest.nearestRoot(msg.embedding);
    if (!match) return;

    let [nearestRoot, sim] = match;

    // Don't merge with self
    if (nearestRoot === msg.id) {
      const scored: Array<[number, number]> = [];
      for (const root of this._forest.roots()) {
        if (root === msg.id) continue;
        const centroid = this._forest.getCentroid(root);
        if (centroid) {
          const s = cosineSimilarity(msg.embedding, centroid);
          scored.push([s, root]);
        }
      }
      if (scored.length === 0) return;
      scored.sort((a, b) => b[0] - a[0]);
      [sim, nearestRoot] = scored[0];
    }

    if (sim >= this._mergeThreshold) {
      await this._forest.union(msg.id, nearestRoot);
    }

    // Enforce hard cap on cluster count
    while (this._forest.clusterCount() > this._maxColdClusters) {
      const pair = findClosestPair(this._forest);
      if (!pair) break;
      await this._forest.union(pair[0], pair[1]);
    }
  }

  render(
    query?: string | null,
    k: number = 3,
    minSim: number = 0.05,
  ): string[] {
    let cold: string[];

    if (query != null && this._forest.clusterCount() > 0) {
      const queryEmb = this._embedder.embed(query);
      const topRoots = this._forest.nearest(queryEmb, k, minSim);
      cold = topRoots.map((r) => this._forest.compact(r));
    } else {
      cold = this._forest.roots().map((r) => this._forest.compact(r));
    }

    const hot = this._hot.map((m) => m.content);
    return [...cold, ...hot];
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
}
