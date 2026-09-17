// Splits a generated answer into paragraph/code blocks (fenced ```
// blocks vs. everything else) per the design brief's `Answer.blocks`
// shape -- no markdown library, since this is the only structure the
// v2-cited prompt actually produces (see rag/prompts/v2_cited.py).

export type AnswerBlock = { type: "paragraph"; text: string } | { type: "code"; code: string };

export function splitAnswerBlocks(answer: string): AnswerBlock[] {
  const blocks: AnswerBlock[] = [];
  const parts = answer.split(/```(?:\w*\n)?([\s\S]*?)```/g);
  parts.forEach((part, i) => {
    if (i % 2 === 1) {
      blocks.push({ type: "code", code: part.replace(/\n$/, "") });
      return;
    }
    part
      .split(/\n{2,}/)
      .map((p) => p.trim())
      .filter(Boolean)
      .forEach((text) => blocks.push({ type: "paragraph", text }));
  });
  return blocks;
}
