import { describe, expect, it } from "vitest";
import { renderAnswerInline, renderInlineCode } from "./inline";
import type { CitationOut } from "./types";

const citation = (index: number): CitationOut => ({
  index,
  chunk_id: `c${index}`,
  url: `https://x/${index}`,
  path: `path-${index}`,
  gold: false,
});

// Duck-typed rather than importing SourceTile by reference -- avoids a
// same-name import collision with the `citation` test helper above and
// keeps this file decoupled from SourceTile's own props/JSX shape. A
// rendered SourceTile is the only node type here with a `citation` prop,
// so checking for that is equivalent to a type check.
function citationChunkIds(nodes: ReturnType<typeof renderAnswerInline>): string[] {
  return nodes
    .filter(
      (n): n is React.ReactElement<{ citation: CitationOut }> =>
        typeof n === "object" && n !== null && "props" in n && (n.props as { citation?: CitationOut }).citation !== undefined
    )
    .map((n) => n.props.citation.chunk_id);
}

describe("renderAnswerInline", () => {
  it("resolves ASCII [n] markers to a SourceTile", () => {
    const nodes = renderAnswerInline("served concurrently [2].", [citation(2)]);
    expect(citationChunkIds(nodes)).toEqual(["c2"]);
  });

  it("resolves fullwidth 【n】 markers to a SourceTile -- Groq sometimes emits this bracket style", () => {
    const nodes = renderAnswerInline("served concurrently 【2】.", [citation(2)]);
    expect(citationChunkIds(nodes)).toEqual(["c2"]);
  });

  it("resolves mixed ASCII and fullwidth markers in the same string", () => {
    const nodes = renderAnswerInline("first [1] then 【2】 then [3].", [citation(1), citation(2), citation(3)]);
    expect(citationChunkIds(nodes)).toEqual(["c1", "c2", "c3"]);
  });

  it("drops an unresolved fullwidth index silently instead of leaking raw brackets", () => {
    const nodes = renderAnswerInline("unresolved 【9】.", [citation(1)]);
    expect(citationChunkIds(nodes)).toEqual([]);
  });

  it("still treats backtick code spans as code, not as a citation marker", () => {
    const nodes = renderAnswerInline("call `get_db()` then 【1】.", [citation(1)]);
    const code = nodes.find((n) => typeof n === "object" && n !== null && "type" in n && n.type === "code");
    expect(code).toBeDefined();
    expect(citationChunkIds(nodes)).toEqual(["c1"]);
  });
});

describe("renderInlineCode", () => {
  it("still splits backtick spans unaffected by the citation regex change", () => {
    const nodes = renderInlineCode("the `response_model` param", "code-class");
    const code = nodes.find((n) => typeof n === "object" && n !== null && "type" in n && n.type === "code");
    expect(code).toBeDefined();
  });
});
