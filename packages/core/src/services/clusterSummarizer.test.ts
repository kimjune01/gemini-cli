/**
 * @license
 * Copyright 2026 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, expect, it, vi } from 'vitest';
import type { GenerateContentResponse } from '@google/genai';
import type { BaseLlmClient } from '../core/baseLlmClient.js';
import { ClusterSummarizer } from './clusterSummarizer.js';

describe('ClusterSummarizer', () => {
  it('returns single messages without calling the model', async () => {
    const llmClient = {
      generateContent: vi.fn(),
    } as unknown as BaseLlmClient;

    const summarizer = new ClusterSummarizer(
      llmClient,
      'chat-compression-3-pro',
    );
    const summary = await summarizer.summarize(['msg 1']);

    expect(summary).toBe('msg 1');
    expect(llmClient.generateContent).not.toHaveBeenCalled();
  });

  it('builds a cluster summarization request', async () => {
    const llmClient = {
      generateContent: vi.fn().mockResolvedValue({
        candidates: [{ content: { parts: [{ text: 'cluster summary' }] } }],
      } as unknown as GenerateContentResponse),
    } as unknown as BaseLlmClient;

    const summarizer = new ClusterSummarizer(
      llmClient,
      'chat-compression-3-pro',
    );
    const summary = await summarizer.summarize(['msg 1', 'msg 2']);

    expect(summary).toBe('cluster summary');
    expect(llmClient.generateContent).toHaveBeenCalledTimes(1);
    const request = vi.mocked(llmClient.generateContent).mock.calls[0][0];
    expect(request.modelConfigKey).toEqual({ model: 'chat-compression-3-pro' });
    expect(request.contents[0].parts?.[0].text).toContain('[1] msg 1');
    expect(request.contents[0].parts?.[0].text).toContain('[2] msg 2');
  });

  it('falls back to joined messages when the model returns no text', async () => {
    const llmClient = {
      generateContent: vi.fn().mockResolvedValue({
        candidates: [{ content: { parts: [] } }],
      } as unknown as GenerateContentResponse),
    } as unknown as BaseLlmClient;

    const summarizer = new ClusterSummarizer(
      llmClient,
      'chat-compression-3-pro',
    );
    const summary = await summarizer.summarize(['msg 1', 'msg 2']);

    expect(summary).toBe('msg 1\n---\nmsg 2');
  });

  it('falls back to joined messages when the model call fails', async () => {
    const llmClient = {
      generateContent: vi.fn().mockRejectedValue(new Error('boom')),
    } as unknown as BaseLlmClient;

    const summarizer = new ClusterSummarizer(
      llmClient,
      'chat-compression-3-pro',
    );
    const summary = await summarizer.summarize(['msg 1', 'msg 2']);

    expect(summary).toBe('msg 1\n---\nmsg 2');
  });
});
