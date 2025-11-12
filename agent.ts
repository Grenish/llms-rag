import { ChromaClient, Collection } from "chromadb";
import fetch, { Response } from "node-fetch";
import { readFileSync } from "fs";
import { performance } from "perf_hooks";
import 'dotenv/config';

/* ----------------------- CONFIG ----------------------- */
const CONFIG = {
  OLLAMA: {
    BASE_URL: "http://localhost:11434",
    GENERATE_MODEL: "granite4:1b-h",
    DECISION_MODEL: "granite4:1b-h", // Upgraded for better decision making
    EMBEDDING_MODEL: "embeddinggemma:300m",
    MAX_RETRIES: 3,
    RETRY_DELAY: 1000,
    GENERATION_TIMEOUT: 60000, // 60 seconds for generation
    EMBEDDING_TIMEOUT: 15000,  // 15 seconds for embeddings
  },
  CHROMA: {
    HOST: "localhost",
    PORT: 8000,
    SSL: false,
    COLLECTION_NAME: "local_docs",
  },
  RAG: {
    TOP_K_RESULTS: 5, // Increased for better context
    BATCH_SIZE: 10,
    MIN_SIMILARITY_SCORE: 0.3, // Add similarity threshold
  },
  WEB_SEARCH: {
    MAX_RESULTS: 5, // Increased for better coverage
    TIMEOUT: 20000, // Increased to 20 seconds
    RETRY_ON_TIMEOUT: true,
    CACHE_DURATION: 300000, // 5 minutes cache
  },
} as const;

/* ----------------------- TYPES ------------------------ */
interface EmbeddingResponse { embedding: number[] }
interface OllamaGenerateResponse { response?: string; done?: boolean }
type Tool = "RAG" | "WebSearch" | "Both";

interface SearchCache {
  query: string;
  result: string;
  timestamp: number;
}

/* ----------------------- UTIL ------------------------- */
/**
 * A custom error class for operations that can be retried.
 * @class
 * @extends Error
 */
class RetryableError extends Error {
  /**
   * Creates an instance of RetryableError.
   * @param {string} message - The error message.
   * @param {boolean} [retryable=true] - Indicates if the error is retryable.
   */
  constructor(message: string, public readonly retryable = true) {
    super(message);
  }
}

/**
 * Retries an async function with exponential backoff.
 * @template T
 * @param {() => Promise<T>} fn - The async function to retry.
 * @param {number} [maxRetries=CONFIG.OLLAMA.MAX_RETRIES] - The maximum number of retries.
 * @param {number} [delay=CONFIG.OLLAMA.RETRY_DELAY] - The initial delay in milliseconds.
 * @returns {Promise<T>} The result of the async function.
 * @throws Will throw the last error if all retries fail.
 */
async function retryWithBackoff<T>(
  fn: () => Promise<T>,
  maxRetries = CONFIG.OLLAMA.MAX_RETRIES,
  delay = CONFIG.OLLAMA.RETRY_DELAY
): Promise<T> {
  let lastError: any;
  for (let i = 0; i < maxRetries; i++) {
    try {
      return await fn();
    } catch (error: any) {
      lastError = error;
      if (i === maxRetries - 1 || (error instanceof RetryableError && !error.retryable)) {
        throw error;
      }
      const waitTime = delay * Math.pow(1.5, i); // Less aggressive backoff
      console.log(`Retry ${i + 1}/${maxRetries} after ${waitTime}ms...`);
      await new Promise(res => setTimeout(res, waitTime));
    }
  }
  throw lastError || new Error("Max retries reached");
}

/* -------------------- OLLAMA SERVICE ------------------ */
/**
 * A service for interacting with the Ollama API.
 * @class
 */
class OllamaService {
  /**
   * Fetches a URL with a timeout.
   * @private
   * @param {string} url - The URL to fetch.
   * @param {any} options - The fetch options.
   * @param {number} [timeout] - The timeout in milliseconds.
   * @returns {Promise<Response>} The fetch response.
   * @throws {RetryableError} If the request times out.
   */
  private async fetchWithTimeout(url: string, options: any, timeout?: number): Promise<Response> {
    const actualTimeout = timeout || CONFIG.OLLAMA.GENERATION_TIMEOUT;
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), actualTimeout);

    try {
      const response = await fetch(url, { ...options, signal: controller.signal });
      clearTimeout(timeoutId);
      return response;
    } catch (error: any) {
      clearTimeout(timeoutId);
      if (error.name === "AbortError") {
        throw new RetryableError(`Request timeout after ${actualTimeout}ms`);
      }
      throw error;
    }
  }

  /**
   * Generates a completion from the Ollama API.
   * @param {string} model - The model to use for generation.
   * @param {string} prompt - The prompt for the generation.
   * @param {number} [temperature=0.3] - The temperature for the generation.
   * @returns {Promise<string>} The generated text.
   */
  async generate(model: string, prompt: string, temperature = 0.3): Promise<string> {
    return retryWithBackoff(async () => {
      const res = await this.fetchWithTimeout(
        `${CONFIG.OLLAMA.BASE_URL}/api/generate`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            model,
            prompt,
            stream: false,
            options: { 
              temperature, 
              top_p: 0.9,
              num_predict: 512,
              stop: ["User:", "Question:", "CONTEXT:", "JSON Response:"]
            },
          }),
        },
        CONFIG.OLLAMA.GENERATION_TIMEOUT
      );

      if (!res.ok) {
        throw new RetryableError(`Generation failed: ${res.status} ${res.statusText}`);
      }

      const data = await res.json() as OllamaGenerateResponse;
      return (data.response || "").trim();
    });
  }

  /**
   * Generates embeddings for a given text from the Ollama API.
   * @param {string} text - The text to embed.
   * @returns {Promise<number[]>} The generated embeddings.
   */
  async embed(text: string): Promise<number[]> {
    return retryWithBackoff(async () => {
      const res = await this.fetchWithTimeout(
        `${CONFIG.OLLAMA.BASE_URL}/api/embeddings`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ 
            model: CONFIG.OLLAMA.EMBEDDING_MODEL, 
            prompt: text.slice(0, 2048) // Limit text length for embeddings
          }),
        },
        CONFIG.OLLAMA.EMBEDDING_TIMEOUT
      );

      if (!res.ok) {
        throw new RetryableError(`Embedding failed: ${res.status} ${res.statusText}`);
      }

      const data = await res.json() as EmbeddingResponse;
      if (!data.embedding || !Array.isArray(data.embedding)) {
        throw new Error("Invalid embedding response");
      }
      return data.embedding;
    });
  }
}

/* ---------------------- RAG SERVICE ------------------- */
/**
 * A service for Retrieval-Augmented Generation (RAG).
 * @class
 */
class RAGService {
  private collection?: Collection;
  private embedCache = new Map<string, number[]>();

  /**
   * Creates an instance of RAGService.
   * @param {ChromaClient} chroma - The ChromaDB client.
   * @param {OllamaService} ollama - The Ollama service.
   */
  constructor(private chroma: ChromaClient, private ollama: OllamaService) {}

  /**
   * Gets a cached embedding or generates a new one.
   * @private
   * @param {string} text - The text to embed.
   * @returns {Promise<number[]>} The embedding.
   */
  private async getCachedEmbedding(text: string): Promise<number[]> {
    const cacheKey = text.slice(0, 100); // Use first 100 chars as cache key
    const cached = this.embedCache.get(cacheKey);
    if (cached) return cached;
    const embedding = await this.ollama.embed(text);
    this.embedCache.set(cacheKey, embedding);
    return embedding;
  }

  /**
   * Initializes the RAG service by setting up the ChromaDB collection.
   * @param {string} dataPath - The path to the data file.
   * @returns {Promise<void>}
   */
  async initialize(dataPath: string): Promise<void> {
    try {
      this.collection = await this.chroma.getCollection({ name: CONFIG.CHROMA.COLLECTION_NAME });
    } catch {
      this.collection = await this.chroma.createCollection({ name: CONFIG.CHROMA.COLLECTION_NAME });
    }

    const count = await this.collection.count();
    if (count === 0) {
      await this.indexDocuments(dataPath);
    } else {
      console.log(`📦 Collection has ${count} documents. Ready for queries.`);
    }
  }

  /**
   * Indexes documents from a data file into the ChromaDB collection.
   * @private
   * @param {string} dataPath - The path to the data file.
   * @returns {Promise<void>}
   */
  private async indexDocuments(dataPath: string): Promise<void> {
    console.log("📚 Indexing local documents...");
    const docs = this.loadData(dataPath);
    const startTime = performance.now();

    const batches: string[][] = [];
    for (let i = 0; i < docs.length; i += CONFIG.RAG.BATCH_SIZE) {
      batches.push(docs.slice(i, i + CONFIG.RAG.BATCH_SIZE));
    }

    let processed = 0;
    for (const batch of batches) {
      const embeddings = await Promise.all(batch.map(doc => this.getCachedEmbedding(doc)));
      const ids = batch.map((_, idx) => `doc-${processed + idx}`);
      await this.collection!.add({
        ids,
        embeddings,
        documents: batch,
        metadatas: batch.map((doc, idx) => ({ 
          index: processed + idx, 
          timestamp: new Date().toISOString(),
          length: doc.length,
          preview: doc.slice(0, 100)
        })),
      });
      processed += batch.length;
      console.log(`Indexed ${processed}/${docs.length} documents`);
    }

    const elapsed = ((performance.now() - startTime) / 1000).toFixed(2);
    console.log(`✅ Indexing completed in ${elapsed}s`);
  }

  /**
   * Loads data from a JSON file.
   * @private
   * @param {string} path - The path to the JSON file.
   * @returns {string[]} The loaded data.
   * @throws {Error} If the data is not a JSON array of paragraphs.
   */
  private loadData(path: string): string[] {
    try {
      const raw = readFileSync(path, "utf-8");
      const json = JSON.parse(raw);
      if (!Array.isArray(json)) throw new Error("data.json must be a JSON array of paragraphs.");
      return json.map((p: any) => String(p).trim()).filter(Boolean);
    } catch (err: any) {
      throw new Error(`Failed to load data: ${err.message}`);
    }
  }

  /**
   * Searches the ChromaDB collection for relevant documents.
   * @param {string} query - The search query.
   * @param {number} [maxResults] - The maximum number of results to return.
   * @returns {Promise<string>} The search results as a single string.
   */
  async search(query: string, maxResults?: number): Promise<string> {
    if (!this.collection) throw new Error("RAG not initialized");
    
    const queryEmb = await this.getCachedEmbedding(query);
    const nResults = maxResults || CONFIG.RAG.TOP_K_RESULTS;
    
    const results = await this.collection.query({ 
      queryEmbeddings: [queryEmb], 
      nResults 
    });
    
    const documents = results.documents?.flat().filter(Boolean) || [];
    const distances = results.distances?.flat() || [];
    
    // Filter by similarity score if distances are available
    const relevantDocs = documents.filter((_, idx) => {
      const distance = distances[idx];
      return !distance || distance < (1 - CONFIG.RAG.MIN_SIMILARITY_SCORE);
    });
    
    if (relevantDocs.length === 0) {
      return "No relevant documents found in local knowledge base.";
    }
    
    return relevantDocs.slice(0, nResults).join("\n\n---\n\n");
  }
}

/* -------------------- TAVILY WEB SEARCH -------------------- */
/**
 * A service for performing web searches using the Tavily API.
 * @class
 */
class WebSearchService {
  private apiKey: string;
  private searchCache: Map<string, SearchCache> = new Map();

  /**
   * Creates an instance of WebSearchService.
   * @throws {Error} If the TAVILY_API_KEY is not set.
   */
  constructor() {
    this.apiKey = String(process.env.TAVILY_API_KEY || "");
    if (!this.apiKey) {
      throw new Error(
        "TAVILY_API_KEY not set. Please add it to your .env file:\nTAVILY_API_KEY=your_api_key_here"
      );
    }
  }

  /**
   * Gets the cache key for a given query.
   * @private
   * @param {string} query - The search query.
   * @returns {string} The cache key.
   */
  private getCacheKey(query: string): string {
    return query.toLowerCase().trim();
  }

  /**
   * Gets a cached search result.
   * @private
   * @param {string} query - The search query.
   * @returns {string | null} The cached result or null if not found.
   */
  private getCachedResult(query: string): string | null {
    const key = this.getCacheKey(query);
    const cached = this.searchCache.get(key);
    
    if (cached && (Date.now() - cached.timestamp) < CONFIG.WEB_SEARCH.CACHE_DURATION) {
      console.log("📋 Using cached web search result");
      return cached.result;
    }
    
    return null;
  }

  /**
   * Caches a search result.
   * @private
   * @param {string} query - The search query.
   * @param {string} result - The search result.
   * @returns {void}
   */
  private setCachedResult(query: string, result: string): void {
    const key = this.getCacheKey(query);
    this.searchCache.set(key, {
      query: key,
      result,
      timestamp: Date.now()
    });
  }

  /**
   * Performs a web search using the Tavily API.
   * @param {string} query - The search query.
   * @param {number} [retryCount=0] - The current retry count.
   * @returns {Promise<string>} The search results as a single string.
   */
  async search(query: string, retryCount = 0): Promise<string> {
    // Check cache first
    const cached = this.getCachedResult(query);
    if (cached) return cached;

    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), CONFIG.WEB_SEARCH.TIMEOUT);

      const payload = {
        api_key: this.apiKey,
        query,
        search_depth: "advanced",
        max_results: CONFIG.WEB_SEARCH.MAX_RESULTS,
        include_answer: true,
        include_raw_content: false,
        include_domains: [],
        exclude_domains: [],
      };

      const res = await fetch("https://api.tavily.com/search", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(payload),
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      if (!res.ok) {
        const errorText = await res.text().catch(() => "");
        
        // Handle rate limiting
        if (res.status === 429) {
          console.log("⏳ Rate limited, waiting before retry...");
          await new Promise(resolve => setTimeout(resolve, 5000));
          if (retryCount < 2) {
            return this.search(query, retryCount + 1);
          }
        }
        
        return `Web search failed: ${res.status} ${res.statusText}. ${errorText}`;
      }

      const data = await res.json().catch(() => null);
      if (!data) return "No results from web search.";

      const result = this.parseTavilyResponse(data);
      
      // Cache successful result
      this.setCachedResult(query, result);
      
      return result;
      
    } catch (error: any) {
      if (error.name === "AbortError") {
        // Retry on timeout if configured
        if (CONFIG.WEB_SEARCH.RETRY_ON_TIMEOUT && retryCount < 1) {
          console.log("⏱️ Search timed out, retrying...");
          return this.search(query, retryCount + 1);
        }
        return "Web search timed out. Please try again.";
      }
      return `Web search error: ${error.message}`;
    }
  }

  /**
   * Parses the response from the Tavily API.
   * @private
   * @param {any} data - The response data.
   * @returns {string} The parsed search results.
   */
  private parseTavilyResponse(data: any): string {
    const sections: string[] = [];

    // Use the AI-generated answer if available
    if (data.answer && typeof data.answer === "string" && data.answer.trim().length > 0) {
      sections.push(`**Summary:** ${data.answer.trim()}`);
    }

    // Add detailed results
    if (Array.isArray(data.results) && data.results.length > 0) {
      const results = data.results.slice(0, CONFIG.WEB_SEARCH.MAX_RESULTS);
      
      sections.push("\n**Sources:**");
      for (let i = 0; i < results.length; i++) {
        const r = data.results[i];
        const title = r.title || "Untitled";
        const url = r.url || "";
        const content = r.content || r.snippet || "";
        const score = r.score ? ` (relevance: ${(r.score * 100).toFixed(0)}%)` : "";
        
        sections.push(`${i + 1}. ${title}${score}\n   ${url}\n   ${content.slice(0, 200)}...`);
      }
    }

    // Include query-related info if available
    if (data.query) {
      sections.push(`\n*Search query: "${data.query}"*`);
    }

    return sections.join("\n\n") || "No relevant web results found.";
  }
}

/* ------------------------ AGENT ------------------------- */
/**
 * An AI agent that can use RAG and web search to answer questions.
 * @class
 */
class AIAgent {
  private decisionCache = new Map<string, Tool>();

  /**
   * Creates an instance of AIAgent.
   * @param {OllamaService} ollama - The Ollama service.
   * @param {RAGService} rag - The RAG service.
   * @param {WebSearchService} webSearch - The web search service.
   */
  constructor(
    private ollama: OllamaService,
    private rag: RAGService,
    private webSearch: WebSearchService
  ) {}

  /**
   * Decides which tool to use for a given query.
   * @param {string} query - The user's query.
   * @returns {Promise<Tool>} The selected tool.
   */
  async decideTool(query: string): Promise<Tool> {
    // Check cache
    const cached = this.decisionCache.get(query.toLowerCase());
    if (cached) return cached;

    const decisionPrompt = `You are an expert tool selector AI. Analyze the question carefully and choose the most appropriate tool.

AVAILABLE TOOLS:
1. "RAG" - Local Knowledge Base
   Use for questions about:
   - Specific known entities: Authrix, AI Cookbook, FuelDev, Grenish Rai, Detoxify
   - Technical documentation, tutorials, guides
   - Historical or archived information
   - Internal system knowledge
   - Previously stored facts and data

2. "WebSearch" - Live Internet Search
   Use for questions about:
   - Current events, news, recent happenings
   - Dates/years mentioned (2023, 2024, 2025, etc.)
   - Real-time information (weather, stocks, prices)
   - Latest updates, versions, releases
   - General world knowledge not in local database
   - Questions with temporal words: "latest", "current", "today", "recently", "now", "trending"
   - Comparisons with external products/services
   - Facts about people, places, or things not in local knowledge

3. "Both" - Combined Search
   Use when:
   - Question requires both local context AND current information
   - Comparing internal knowledge with external updates
   - Need comprehensive answer from multiple sources

DECISION RULES:
- If question mentions specific years or dates → prefer "WebSearch"
- If question asks for "latest" or "current" → prefer "WebSearch"
- If question is about known local entities → prefer "RAG"
- If unclear or needs broad coverage → use "Both"

IMPORTANT: Respond with ONLY a valid JSON object, nothing else.
Format: {"tool": "RAG"} or {"tool": "WebSearch"} or {"tool": "Both"}

Question: ${query}

Analyze the question and respond with JSON:`;

    try {
      const decision = await this.ollama.generate(
        CONFIG.OLLAMA.DECISION_MODEL, 
        decisionPrompt,
        0.1 // Low temperature for consistent decisions
      );
      
      // Extract JSON from response
      const jsonMatch = decision.match(/\{[^}]*"tool"[^}]*\}/);
      if (jsonMatch) {
        const parsed = JSON.parse(jsonMatch[0]);
        if (["RAG", "WebSearch", "Both"].includes(parsed.tool)) {
          this.decisionCache.set(query.toLowerCase(), parsed.tool);
          return parsed.tool;
        }
      }
    } catch (e) {
      console.warn("⚠️ Tool decision via LLM failed, using advanced heuristics");
    }
    
    // Advanced heuristic fallback
    const tool = this.advancedHeuristicDecision(query);
    this.decisionCache.set(query.toLowerCase(), tool);
    return tool;
  }

  /**
   * An advanced heuristic to decide which tool to use.
   * @private
   * @param {string} query - The user's query.
   * @returns {Tool} The selected tool.
   */
  private advancedHeuristicDecision(query: string): Tool {
    const lower = query.toLowerCase();
    
    // Expanded keyword lists with weights
    const webSearchIndicators = {
      strong: ['latest', 'current', 'today', 'yesterday', 'tomorrow', 'breaking', 
               'trending', 'live', 'real-time', 'now', 'recent', 'update'],
      medium: ['news', '2023', '2024', '2025', '2026', 'price', 'weather', 
               'stock', 'market', 'election', 'announcement'],
      weak: ['compare', 'versus', 'vs', 'difference', 'best', 'top']
    };
    
    const ragIndicators = {
      strong: ['authrix', 'fueldev', 'grenish', 'detoxify', 'ai cookbook', 
               'our system', 'our documentation', 'internal'],
      medium: ['documentation', 'guide', 'tutorial', 'api', 'configuration', 
               'setup', 'installation', 'troubleshoot'],
      weak: ['explain', 'describe', 'what is', 'how to', 'define']
    };

    // Calculate weighted scores
    let webScore = 0;
    let ragScore = 0;

    // Check for strong indicators (weight: 3)
    webScore += webSearchIndicators.strong.filter(k => lower.includes(k)).length * 3;
    ragScore += ragIndicators.strong.filter(k => lower.includes(k)).length * 3;

    // Check for medium indicators (weight: 2)
    webScore += webSearchIndicators.medium.filter(k => lower.includes(k)).length * 2;
    ragScore += ragIndicators.medium.filter(k => lower.includes(k)).length * 2;

    // Check for weak indicators (weight: 1)
    webScore += webSearchIndicators.weak.filter(k => lower.includes(k)).length;
    ragScore += ragIndicators.weak.filter(k => lower.includes(k)).length;

    // Check for year patterns (strong indicator for web search)
    if (/\b20\d{2}\b/.test(lower)) {
      webScore += 4;
    }

    // Check for question patterns
    if (/^(what|who|when|where|why|how)\s+(is|are|was|were|has|have)\s+the\s+(latest|current|recent)/.test(lower)) {
      webScore += 3;
    }

    // Decision logic
    if (webScore > 0 && ragScore > 0 && Math.abs(webScore - ragScore) < 3) {
      return "Both"; // Close scores suggest both might be needed
    }
    
    if (webScore > ragScore) {
      return "WebSearch";
    }
    
    if (ragScore > webScore) {
      return "RAG";
    }

    // Default to RAG for general questions
    return "RAG";
  }

  /**
   * Answers a query using the appropriate tool(s).
   * @param {string} query - The user's query.
   * @returns {Promise<string>} The generated answer.
   */
  async answer(query: string): Promise<string> {
    const tool = await this.decideTool(query);
    console.log(`🧠 Selected Tool: ${tool}`);

    let context = "";
    let contextSource = "";

    if (tool === "Both") {
      console.log("🔄 Fetching from both sources...");
      
      // Parallel fetch from both sources
      const [ragContext, webContext] = await Promise.allSettled([
        this.rag.search(query),
        this.webSearch.search(query)
      ]);

      const ragResult = ragContext.status === "fulfilled" ? ragContext.value : "RAG search failed.";
      const webResult = webContext.status === "fulfilled" ? webContext.value : "Web search failed.";

      context = `**Local Knowledge Base:**\n${ragResult}\n\n**Web Search Results:**\n${webResult}`;
      contextSource = "both local knowledge and web search";
      
    } else if (tool === "RAG") {
      context = await this.rag.search(query);
      contextSource = "local knowledge base";
      
    } else {
      context = await this.webSearch.search(query);
      contextSource = "web search";
    }

    // Enhanced answer prompt
    const answerPrompt = `You are an intelligent, helpful AI assistant. Your task is to provide accurate, comprehensive answers based on the provided context.

CONTEXT SOURCE: ${contextSource}

INSTRUCTIONS:
1. Answer the user's question directly and thoroughly
2. Use ONLY information from the provided context
3. If context is insufficient, acknowledge this honestly
4. Structure your response with clear paragraphs
5. Be concise but complete
6. Cite sources when they're provided in the context
7. If multiple sources conflict, mention the discrepancy
8. Maintain a professional, friendly tone

CONTEXT:
${context}

USER QUESTION:
${query}

COMPREHENSIVE ANSWER:`;

    const answer = await this.ollama.generate(
      CONFIG.OLLAMA.GENERATE_MODEL, 
      answerPrompt,
      0.7 // Balanced temperature for generation
    );

    // Post-process to ensure quality
    return this.postProcessAnswer(answer, tool);
  }

  /**
   * Post-processes the generated answer.
   * @private
   * @param {string} answer - The generated answer.
   * @param {Tool} tool - The tool used to generate the answer.
   * @returns {string} The post-processed answer.
   */
  private postProcessAnswer(answer: string, tool: Tool): string {
    // Clean up any residual prompt artifacts
    answer = answer.replace(/^(COMPREHENSIVE ANSWER:|RESPONSE:|ANSWER:)/i, "").trim();
    
    // Add source attribution if not present
    if (!answer.includes("source") && !answer.includes("according to")) {
      const attribution = tool === "WebSearch" 
        ? "\n\n*[Source: Web Search]*"
        : tool === "RAG"
        ? "\n\n*[Source: Local Knowledge Base]*"
        : "\n\n*[Sources: Local Knowledge Base & Web Search]*";
      
      // Only add if answer doesn't already have attribution
      if (!answer.includes("[Source:")) {
        answer += attribution;
      }
    }
    
    return answer;
  }
}

/* ------------------------ MAIN -------------------------- */
/**
 * The main function of the application.
 * @async
 * @returns {Promise<void>}
 */
async function main() {
  try {
    console.log("🚀 Initializing Enhanced AI Agent with Tavily Web Search + RAG...\n");

    // Validate environment
    if (!process.env.TAVILY_API_KEY) {
      console.error("❌ TAVILY_API_KEY not found in environment!");
      console.error("Please create a .env file with: TAVILY_API_KEY=your_key_here");
      process.exit(1);
    }

    const chroma = new ChromaClient({
      path: `http://${CONFIG.CHROMA.HOST}:${CONFIG.CHROMA.PORT}`,
    });

    const ollama = new OllamaService();
    const rag = new RAGService(chroma, ollama);
    const webSearch = new WebSearchService();
    const agent = new AIAgent(ollama, rag, webSearch);

    await rag.initialize("./data.json");

    // Get query from arguments or use default
    const query = process.argv.slice(2).join(" ") || 
      "What are the latest AI developments in 2025?";
    
    console.log(`\n📝 Query: "${query}"\n`);
    console.log("🤖 Processing your request...\n");

    const startTime = performance.now();
    const answer = await agent.answer(query);
    const elapsed = ((performance.now() - startTime) / 1000).toFixed(2);

    console.log("=".repeat(70));
    console.log("📊 ANSWER");
    console.log("=".repeat(70));
    console.log(answer);
    console.log("=".repeat(70));
    console.log(`⏱️  Response generated in ${elapsed} seconds`);
    console.log("=".repeat(70));
    
  } catch (error: any) {
    console.error("\n❌ Fatal Error:", error.message || error);
    if (error.stack && process.env.DEBUG === "true") {
      console.error("\n📋 Stack Trace:");
      console.error(error.stack);
    }
    console.error("\n💡 Tip: Check your .env file and ensure all services are running");
    process.exit(1);
  }
}

if (require.main === module) main();

export { AIAgent, RAGService, WebSearchService, OllamaService };
