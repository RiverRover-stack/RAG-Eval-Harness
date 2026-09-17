// POST /api/ask/stream via fetch + ReadableStream -- EventSource can't POST
// a body, and this route needs the question in the request. Parses the
// `event: <name>\ndata: <json>\n\n` blocks src/rag_eval/api/routes/ask.py
// writes (see its module docstring for the exact event sequence).

import type { SSEEvent } from "./types";

function parseBlock(block: string): SSEEvent | null {
  const lines = block.split("\n");
  const eventLine = lines.find((l) => l.startsWith("event: "));
  const dataLine = lines.find((l) => l.startsWith("data: "));
  if (!eventLine || !dataLine) return null;
  const event = eventLine.slice("event: ".length).trim();
  const data = JSON.parse(dataLine.slice("data: ".length));
  return { event, data } as SSEEvent;
}

export async function* streamAsk(question: string, apiBase = ""): AsyncGenerator<SSEEvent> {
  let res: Response;
  try {
    res = await fetch(`${apiBase}/api/ask/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
  } catch (e) {
    yield { event: "error", data: { detail: `request failed: ${(e as Error).message}` } };
    return;
  }

  if (!res.ok || !res.body) {
    yield { event: "error", data: { detail: `request failed: ${res.status}` } };
    return;
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let sepIndex: number;
    while ((sepIndex = buffer.indexOf("\n\n")) !== -1) {
      const block = buffer.slice(0, sepIndex);
      buffer = buffer.slice(sepIndex + 2);
      const parsed = parseBlock(block);
      if (parsed) yield parsed;
    }
  }
}
