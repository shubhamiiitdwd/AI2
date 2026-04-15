/** Maps backend `source` on AI responses to UI label and badge style. */
export function aiSourceDisplay(source: string): { label: string; badgeClass: string } {
  if (source === 'huggingface') return { label: 'HuggingFace', badgeClass: 'aw-badge--green' };
  if (source === 'azure') return { label: 'Azure OpenAI', badgeClass: 'aw-badge--green' };
  return { label: 'Rule-based fallback', badgeClass: 'aw-badge--orange' };
}
