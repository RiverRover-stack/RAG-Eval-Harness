// Shared inline-text rendering for the question heading and answer body:
// backtick code spans everywhere, plus `[n]` citation markers (only present
// in the answer body) swapped for the SourceTile they were generated next
// to -- extract_citations() in rag/citations.py places the marker right
// before the sentence's closing punctuation, so substituting in place
// already puts the tile "at the end of the claim it supports".
//
// Also matches the fullwidth `【n】` form some providers (observed: Groq)
// emit instead of the prompted `[n]` -- mirrors rag/citations.py's
// `_CITATION_RE = r"[\[【](\d+)[\]】]"` exactly, so a citation that the
// backend already extracts/scores correctly doesn't silently render as
// inert bracket text here just because of which bracket glyph the model
// used.

import type { ReactNode } from "react";
import { SourceTile } from "@/components/SourceTile";
import type { CitationOut } from "./types";

export function renderInlineCode(text: string, codeClassName: string): ReactNode[] {
  return text.split(/(`[^`]+`)/g).map((part, i) =>
    part.length > 1 && part.startsWith("`") && part.endsWith("`") ? (
      <code key={i} className={codeClassName}>
        {part.slice(1, -1)}
      </code>
    ) : (
      <span key={i}>{part}</span>
    )
  );
}

export function renderAnswerInline(text: string, citations: CitationOut[]): ReactNode[] {
  const byIndex = new Map(citations.map((c) => [c.index, c]));
  return text.split(/(`[^`]+`|[\[【]\d+[\]】])/g).map((part, i) => {
    if (part.length > 1 && part.startsWith("`") && part.endsWith("`")) {
      return (
        <code key={i} className="font-mono text-[14px] text-accent-signal">
          {part.slice(1, -1)}
        </code>
      );
    }
    const marker = /^[\[【](\d+)[\]】]$/.exec(part);
    if (marker) {
      const citation = byIndex.get(Number(marker[1]));
      // An unknown/unresolved index drops silently rather than leaking a
      // raw "[n]" into prose -- validate_citations() already excludes these
      // from the model's own coverage score.
      return citation ? <SourceTile key={i} citation={citation} /> : null;
    }
    return <span key={i}>{part}</span>;
  });
}
