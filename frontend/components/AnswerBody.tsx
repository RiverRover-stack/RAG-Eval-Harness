import { splitAnswerBlocks } from "@/lib/parseAnswer";
import { renderAnswerInline } from "@/lib/inline";
import type { CitationOut } from "@/lib/types";
import type { ReactNode } from "react";

function highlightCode(code: string): ReactNode[] {
  return code.split(/(try:|finally:|yield)/g).map((part, i) => {
    if (part === "try:" || part === "finally:") {
      return (
        <span key={i} className="text-text-dim">
          {part}
        </span>
      );
    }
    if (part === "yield") {
      return (
        <span key={i} className="text-accent-signal">
          {part}
        </span>
      );
    }
    return <span key={i}>{part}</span>;
  });
}

export function AnswerBody({
  answer,
  citations,
  onCitationClick,
}: {
  answer: string;
  citations: CitationOut[];
  onCitationClick?: (chunkId: string) => void;
}) {
  const blocks = splitAnswerBlocks(answer);
  return (
    <div className="flex flex-col gap-4 text-[15.5px] leading-[1.76] text-text">
      {blocks.map((block, i) =>
        block.type === "code" ? (
          <pre
            key={i}
            className="bg-surface border border-border rounded-[6px] px-[18px] py-[15px] font-mono text-[13px] leading-[1.75] whitespace-pre overflow-auto"
          >
            <code>{highlightCode(block.code)}</code>
          </pre>
        ) : (
          <p key={i}>{renderAnswerInline(block.text, citations, onCitationClick)}</p>
        )
      )}
    </div>
  );
}
