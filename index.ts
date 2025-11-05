import { ChromaClient } from "chromadb";
import fetch from "node-fetch";
import { readFileSync } from "fs";

// Initialize Chroma client
const chroma = new ChromaClient({ host: "localhost", port: 8000, ssl: false });

// --- Embed text using Ollama ---
async function embed(text: string) {
  const res = await fetch("http://localhost:11434/api/embeddings", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model: "embeddinggemma:300m", prompt: text }),
  });

  if (!res.ok) throw new Error(`Embedding failed: ${res.statusText}`);
  const data = await res.json();
  return data.embedding as number[];
}

// --- Generate completion from Ollama ---
async function generate(prompt: string) {
  const res = await fetch("http://localhost:11434/api/generate?stream=false", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model: "llama3.2:1b", prompt }),
  });

  if (!res.ok) throw new Error(`Generation failed: ${res.statusText}`);

  // Bun returns full response as text
  const text = await res.text();

  // Ollama outputs newline-delimited JSON objects
  const lines = text.split("\n").filter(Boolean);
  let fullText = "";

  for (const line of lines) {
    try {
      const json = JSON.parse(line);
      if (json.response) fullText += json.response;
    } catch {
      // ignore malformed lines
    }
  }

  return fullText.trim();
}

// --- Load and prepare data ---
function loadData(path: string): string[] {
  const raw = readFileSync(path, "utf-8");
  const json = JSON.parse(raw);

  if (!Array.isArray(json)) throw new Error("Data must be an array of paragraphs.");
  return json.map((p: string) => p.trim()).filter(Boolean);
}

// --- Main workflow ---
async function main() {
  console.log("🔹 Initializing RAG...");

  // Load your data.json
  const docs = loadData("./data.json");

  let collection;
  try {
    collection = await chroma.getCollection({ name: "local_docs" });
  } catch {
    console.log("Collection not found — creating new one...");
    collection = await chroma.createCollection({
      name: "local_docs",
      metadata: { embedding_function: "manual" }
    });
  }

  const count = (await collection.count()) ?? 0;
  if (count === 0) {
    console.log("📚 Indexing documents...");
    for (let i = 0; i < docs.length; i++) {
      const text = docs[i];
      const emb = await embed(text);
      await collection.add({
        ids: [`doc-${i}`],
        embeddings: [emb],
        documents: [text],
      });
      console.log(`Indexed doc-${i + 1}/${docs.length}`);
    }
  } else {
    console.log(`📦 Existing collection detected (${count} docs). Skipping indexing.`);
  }

  // Define your query
  const query = "tell me about authrix";

  console.log(`\n🔍 Searching for relevant context...`);
  const queryEmb = await embed(query);
  const results = await collection.query({
    queryEmbeddings: [queryEmb],
    nResults: 5,
  });

  const context = results.documents?.flat().join("\n") || "";
  const prompt = `You are an expert assistant. Use the following context to answer clearly.\n\nContext:\n${context}\n\nQuestion: ${query}\nAnswer:`;

  console.log("\n🤖 Generating answer...\n");
  const answer = await generate(prompt);
  console.log(answer);
}

main().catch(console.error);
