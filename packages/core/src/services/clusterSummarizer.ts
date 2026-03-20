/**
 * @license
 * Copyright 2026 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import type { BaseLlmClient } from '../core/baseLlmClient.js';
import { getResponseText } from '../utils/partUtils.js';
import { LlmRole } from '../telemetry/types.js';
import type { Summarizer } from './contextWindow.js';

export class ClusterSummarizer implements Summarizer {
  constructor(
    private readonly llmClient: BaseLlmClient,
    private readonly modelConfigKey: string,
  ) {}

  async summarize(
    messages: string[],
    abortSignal?: AbortSignal,
  ): Promise<string> {
    if (messages.length === 0) {
      return '';
    }
    if (messages.length === 1) {
      return messages[0];
    }

    const fallback = messages.join('\n---\n');

    try {
      const response = await this.llmClient.generateContent({
        modelConfigKey: { model: this.modelConfigKey },
        contents: [
          {
            role: 'user',
            parts: [{ text: this.buildClusterPrompt(messages) }],
          },
        ],
        promptId: 'cluster-summarize',
        role: LlmRole.UTILITY_COMPRESSOR,
        abortSignal,
      });

      return getResponseText(response)?.trim() || fallback;
    } catch {
      return fallback;
    }
  }

  private buildClusterPrompt(messages: string[]): string {
    const numberedMessages = messages
      .map((message, index) => `[${index + 1}] ${message}`)
      .join('\n');

    return [
      'Summarize the following conversation messages into a concise, information-dense paragraph.',
      'Preserve specific technical details, file paths, tool results, variable names, and user constraints.',
      '',
      numberedMessages,
    ].join('\n');
  }
}
