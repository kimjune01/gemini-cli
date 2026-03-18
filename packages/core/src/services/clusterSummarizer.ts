/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import type { BaseLlmClient } from '../core/baseLlmClient.js';
import { getResponseText } from '../utils/partUtils.js';
import { LlmRole } from '../telemetry/types.js';
import type { Summarizer } from './contextWindow.js';

/**
 * Cluster summarizer using BaseLlmClient for LLM-generated summaries.
 *
 * Single-phase summarization (no verification) since clusters are small.
 */
export class ClusterSummarizer implements Summarizer {
  private _client: BaseLlmClient;
  private _modelConfigKey: string;

  constructor(client: BaseLlmClient, modelConfigKey: string) {
    this._client = client;
    this._modelConfigKey = modelConfigKey;
  }

  async summarize(messages: string[]): Promise<string> {
    const fallback = messages.join('\n---\n');

    try {
      const response = await this._client.generateContent({
        modelConfigKey: { model: this._modelConfigKey },
        contents: [
          {
            role: 'user',
            parts: [
              {
                text: `Summarize the following conversation messages into a concise, information-dense paragraph. Preserve all specific technical details, file paths, tool results, variable names, and user constraints.\n\nMessages:\n${messages.map((m, i) => `[${i + 1}] ${m}`).join('\n\n')}`,
              },
            ],
          },
        ],
        promptId: 'cluster-summarize',
        role: LlmRole.UTILITY_COMPRESSOR,
        abortSignal: new AbortController().signal,
      });

      const text = getResponseText(response)?.trim();
      if (!text) return fallback;
      return text;
    } catch {
      return fallback;
    }
  }
}
