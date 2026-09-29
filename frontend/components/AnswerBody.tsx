import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { renderAnswerInline } from "@/lib/inline";
import type { CitationOut } from "@/lib/types";
import { Children, useMemo } from "react";
import type { Components } from "react-markdown";
import type { ReactElement, ReactNode } from "react";

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

// react-markdown hands every text-bearing element (p, li, strong, ...) its
// children as a mix of plain strings and already-rendered elements. Citation
// markers like `[1]` / `【1†source】` live in the string children, and must
// become SourceTiles via renderAnswerInline -- while every non-string child
// (a nested <strong>, <a>, ...) must pass through untouched.
function withCitations(children: ReactNode, citations: CitationOut[]): ReactNode {
  return Children.map(children, (child) =>
    typeof child === "string" ? renderAnswerInline(child, citations) : child
  );
}

function markdownComponents(citations: CitationOut[]): Components {
  const inline = (children: ReactNode) => withCitations(children, citations);
  return {
    p: ({ children }) => <p>{inline(children)}</p>,
    li: ({ children }) => <li className="pl-1">{inline(children)}</li>,
    strong: ({ children }) => <strong className="font-semibold">{inline(children)}</strong>,
    em: ({ children }) => <em>{inline(children)}</em>,
    h1: ({ children }) => <h3 className="text-[18px] font-semibold mt-2">{inline(children)}</h3>,
    h2: ({ children }) => <h3 className="text-[17px] font-semibold mt-2">{inline(children)}</h3>,
    h3: ({ children }) => <h3 className="text-[16px] font-semibold mt-2">{inline(children)}</h3>,
    h4: ({ children }) => <h4 className="text-[15.5px] font-semibold mt-1">{inline(children)}</h4>,
    ul: ({ children }) => <ul className="list-disc pl-6 flex flex-col gap-1.5">{children}</ul>,
    ol: ({ children }) => <ol className="list-decimal pl-6 flex flex-col gap-1.5">{children}</ol>,
    a: ({ href, children }) => (
      <a href={href} target="_blank" rel="noopener noreferrer" className="text-accent-signal underline">
        {children}
      </a>
    ),
    table: ({ children }) => (
      <div className="overflow-auto">
        <table className="border-collapse text-[14px]">{children}</table>
      </div>
    ),
    th: ({ children }) => (
      <th className="border border-border px-3 py-1.5 text-left font-semibold">{inline(children)}</th>
    ),
    td: ({ children }) => <td className="border border-border px-3 py-1.5">{inline(children)}</td>,
    // Fenced blocks: react-markdown emits <pre><code>. Handling it at `pre`
    // keeps block code on the existing highlightCode path and means the
    // `code` override below only ever sees inline code.
    pre: ({ children }) => {
      const codeEl = children as ReactElement<{ children?: ReactNode }>;
      const text = String(codeEl.props.children ?? "").replace(/\n$/, "");
      return (
        <pre className="bg-surface border border-border rounded-[6px] px-[18px] py-[15px] font-mono text-[13px] leading-[1.75] whitespace-pre overflow-auto">
          <code>{highlightCode(text)}</code>
        </pre>
      );
    },
    code: ({ children }) => <code className="font-mono text-[14px] text-accent-signal">{children}</code>,
  };
}

export function AnswerBody({ answer, citations }: { answer: string; citations: CitationOut[] }) {
  // Memoised because AnswerBody re-renders on every streamed token: a fresh
  // `components` map gives every override a new function identity, which
  // React treats as a new element type and remounts the whole answer
  // (including SourceTiles) each time.
  const components = useMemo(() => markdownComponents(citations), [citations]);
  return (
    <div className="flex flex-col gap-4 text-[15.5px] leading-[1.76] text-text">
      {/* img disallowed: answers are built from untrusted GitHub Discussions text,
          and a model-emitted image URL would be fetched by the browser with no click. */}
      <ReactMarkdown remarkPlugins={[remarkGfm]} components={components} disallowedElements={["img"]}>
        {answer}
      </ReactMarkdown>
    </div>
  );
}
