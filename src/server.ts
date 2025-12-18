import { routeAgentRequest, type Schedule } from "agents";
import { getSchedulePrompt } from "agents/schedule";
import { AIChatAgent } from "agents/ai-chat-agent";
import {
  generateId,
  streamText,
  type StreamTextOnFinishCallback,
  stepCountIs,
  createUIMessageStream,
  convertToModelMessages,
  createUIMessageStreamResponse,
  type ToolSet
} from "ai";
import { createWorkersAI } from "workers-ai-provider";
import { processToolCalls, cleanupMessages } from "./utils";
import { tools, executions } from "./tools";

// ============================================================================
// TYPE DEFINITIONS FOR PERCIFY AVATAR CO-PILOT
// ============================================================================

/**
 * Avatar profile representing the user's AI persona
 */
export type AvatarProfile = {
  id: string;
  displayName: string;
  bio: string;
  tone: "casual" | "professional" | "playful" | "technical";
  expertiseTags: string[];
};

/**
 * Memory item for storing user preferences, tasks, and notes
 */
export type MemoryItem = {
  id: string;
  createdAt: string;
  type: "task" | "preference" | "note";
  content: string;
};

/**
 * Complete agent state including avatar and memories
 */
export type AgentState = {
  avatar?: AvatarProfile;
  memories: MemoryItem[];
};

// ============================================================================
// PERCIFY AVATAR AGENT IMPLEMENTATION
// ============================================================================

/**
 * PercifyAvatarAgent - AI Avatar Co-Pilot that maintains persistent persona and memory
 * 
 * Features:
 * - Persistent avatar profile with customizable tone and expertise
 * - Long-term memory storage for preferences and tasks
 * - Multi-step task execution with LLM orchestration
 * - Web research capabilities
 * 
 * Uses Cloudflare Workers AI with Llama 3.3 70B for inference
 */
export class PercifyAvatarAgent extends AIChatAgent<Env, AgentState> {
  
  /**
   * Initialize default state on agent start
   */
  initialState: AgentState = {
    avatar: undefined,
    memories: []
  };

  /**
   * Called when the agent starts - ensures state is properly initialized
   */
  onStart(): void {
    console.log("[PercifyAvatarAgent] Agent starting, initializing state...");
    
    // Re-initialize state if missing or corrupted
    if (!this.state || typeof this.state !== "object") {
      console.log("[PercifyAvatarAgent] State missing, initializing defaults");
      this.setState({
        avatar: undefined,
        memories: []
      });
    }
    
    // Ensure memories array exists
    if (!Array.isArray(this.state.memories)) {
      console.log("[PercifyAvatarAgent] Memories array corrupted, resetting");
      this.setState({
        ...this.state,
        memories: []
      });
    }
  }

  /**
   * Get the current avatar state (for API endpoint)
   */
  getAvatarState(): AgentState {
    return {
      avatar: this.state.avatar,
      memories: this.state.memories.slice(-5) // Return last 5 memories
    };
  }

  /**
   * Save or update the avatar profile
   */
  saveAvatarProfile(profileInput: Partial<AvatarProfile>): AvatarProfile {
    console.log("[PercifyAvatarAgent] Updating avatar profile:", profileInput);
    
    const currentAvatar = this.state.avatar;
    const updatedAvatar: AvatarProfile = {
      id: currentAvatar?.id || generateId(),
      displayName: profileInput.displayName || currentAvatar?.displayName || "Unnamed Avatar",
      bio: profileInput.bio || currentAvatar?.bio || "",
      tone: profileInput.tone || currentAvatar?.tone || "casual",
      expertiseTags: profileInput.expertiseTags || currentAvatar?.expertiseTags || []
    };

    this.setState({
      ...this.state,
      avatar: updatedAvatar
    });

    console.log("[PercifyAvatarAgent] Avatar updated:", updatedAvatar);
    return updatedAvatar;
  }

  /**
   * Save a memory item with automatic cleanup of old entries
   */
  saveMemory(type: MemoryItem["type"], content: string): MemoryItem[] {
    console.log("[PercifyAvatarAgent] Storing memory:", { type, content });
    
    const newMemory: MemoryItem = {
      id: generateId(),
      createdAt: new Date().toISOString(),
      type,
      content
    };

    let updatedMemories = [...this.state.memories, newMemory];
    
    // Cap at 50 memories, remove oldest if exceeded
    if (updatedMemories.length > 50) {
      updatedMemories = updatedMemories.slice(-50);
      console.log("[PercifyAvatarAgent] Memory cap reached, trimmed to 50 items");
    }

    this.setState({
      ...this.state,
      memories: updatedMemories
    });

    console.log("[PercifyAvatarAgent] Memory stored, total count:", updatedMemories.length);
    return updatedMemories.slice(-5); // Return last 5 memories
  }

  /**
   * Percify Documentation Database
   * Simulates docs.percify.io content for research queries
   */
  private readonly percifyDocs: Record<string, { title: string; content: string; path: string }> = {
    avatar: {
      title: "Avatar Setup Guide",
      content: "Percify Avatar Co-Pilot lets you create a personalized AI persona. Your avatar includes: displayName (your preferred name), bio (a short description), tone (casual/professional/playful/technical), and expertiseTags (your areas of expertise). Use the command 'create my avatar as [name]' or 'set my tone to professional' to customize your experience. Avatars persist across sessions using Cloudflare Durable Objects.",
      path: "/docs/avatar-guide"
    },
    memory: {
      title: "Memory System",
      content: "Percify stores three types of memories: tasks (things to do), preferences (your likes/settings), and notes (general information). Say 'remember that I prefer dark mode' or 'note: meeting at 3pm'. Memories are stored persistently and limited to 50 items with automatic cleanup of oldest entries. Access recent memories anytime - they're included in your avatar context.",
      path: "/docs/memory-system"
    },
    tone: {
      title: "Tone Customization",
      content: "Choose from 4 tone styles: CASUAL (friendly, relaxed, conversational), PROFESSIONAL (formal, precise, business-like), PLAYFUL (fun, energetic, humorous), TECHNICAL (detailed, accurate, uses technical terms). Change anytime with 'set my tone to [style]'. Your tone affects how Percify responds to all your messages.",
      path: "/docs/customization/tone"
    },
    tools: {
      title: "Available Tools",
      content: "Percify has 4 core tools: 1) saveAvatarProfile - Create/update your avatar (name, bio, tone, expertise). 2) saveMemory - Store preferences, tasks, or notes. 3) researchWeb - Look up information from Percify docs. 4) getAvatarState - View current avatar and recent memories. Plus scheduling tools for reminders and recurring tasks.",
      path: "/docs/tools-reference"
    },
    schedule: {
      title: "Scheduling & Reminders",
      content: "Schedule tasks with natural language: 'remind me in 1 hour to check email' or 'schedule daily standup at 9am'. Percify uses Cloudflare's scheduling system for reliable task execution. View scheduled tasks with 'show my schedules' and cancel with 'cancel schedule [id]'.",
      path: "/docs/scheduling"
    },
    getting_started: {
      title: "Getting Started",
      content: "Welcome to Percify! Start by creating your avatar: tell me your name and preferred tone. Then save some preferences so I remember your style. Try: '1) Create avatar named Alex with professional tone, 2) Remember I prefer TypeScript, 3) Research how memory works'. Your data persists across sessions!",
      path: "/docs/getting-started"
    },
    architecture: {
      title: "Technical Architecture",
      content: "Percify runs on Cloudflare's edge network using: Workers AI (Llama 3.3 70B model for inference), Durable Objects (persistent state storage), WebSockets (real-time chat), and the Agents SDK (orchestration). State is stored in SQLite within Durable Objects for low-latency global access.",
      path: "/docs/architecture"
    },
    api: {
      title: "API Reference",
      content: "REST Endpoints: GET /api/avatar-state returns current avatar and last 5 memories. GET /check-open-ai-key returns provider status. WebSocket connects at /chat for real-time messaging. All endpoints return JSON. Authentication is handled per-session via Durable Object IDs.",
      path: "/docs/api-reference"
    },
    expertise: {
      title: "Expertise Tags",
      content: "Add expertise tags to your avatar to help Percify understand your background. Examples: 'TypeScript', 'React', 'DevOps', 'Machine Learning'. Set with 'my expertise is [tag1], [tag2], [tag3]'. Tags help contextualize responses and can be used for personalized recommendations.",
      path: "/docs/customization/expertise"
    },
    troubleshooting: {
      title: "Troubleshooting",
      content: "Common issues: 1) Avatar not saving? Ensure you provide at least a name. 2) Memories full? Oldest are auto-deleted after 50 items. 3) Slow responses? Check network connection. 4) Wrong tone? Say 'change tone to [style]'. 5) Reset everything? Say 'reset my avatar' to start fresh.",
      path: "/docs/troubleshooting"
    }
  };

  /**
   * Perform web research from Percify documentation
   */
  async researchWeb(query: string): Promise<{ query: string; snippet: string; sourceUrl: string }> {
    console.log("[PercifyAvatarAgent] Searching docs.percify.io for:", query);
    
    const baseUrl = "https://docs.percify.io";
    const queryLower = query.toLowerCase();
    
    // Find matching documentation
    let bestMatch: { title: string; content: string; path: string } | null = null;
    let matchScore = 0;
    
    for (const [key, doc] of Object.entries(this.percifyDocs)) {
      let score = 0;
      
      // Check for keyword matches
      if (queryLower.includes(key)) score += 10;
      if (doc.title.toLowerCase().includes(queryLower)) score += 5;
      if (doc.content.toLowerCase().includes(queryLower)) score += 3;
      
      // Check individual words
      const queryWords = queryLower.split(/\s+/);
      for (const word of queryWords) {
        if (word.length > 2) {
          if (key.includes(word)) score += 2;
          if (doc.content.toLowerCase().includes(word)) score += 1;
        }
      }
      
      if (score > matchScore) {
        matchScore = score;
        bestMatch = doc;
      }
    }
    
    // Default to getting started if no good match
    if (!bestMatch || matchScore < 2) {
      bestMatch = this.percifyDocs.getting_started;
    }
    
    const sourceUrl = `${baseUrl}${bestMatch.path}`;
    
    console.log("[PercifyAvatarAgent] Found doc:", bestMatch.title, "at", sourceUrl);
    
    return {
      query,
      snippet: `📚 **${bestMatch.title}** (docs.percify.io)\n\n${bestMatch.content}`,
      sourceUrl
    };
  }

  /**
   * Build context string for LLM from current state
   */
  private buildContextString(): string {
    const avatar = this.state.avatar;
    const recentMemories = this.state.memories.slice(-3);
    
    let context = "## Current Avatar State\n";
    
    if (avatar) {
      context += `- Name: ${avatar.displayName}\n`;
      context += `- Bio: ${avatar.bio || "Not set"}\n`;
      context += `- Tone: ${avatar.tone}\n`;
      context += `- Expertise: ${avatar.expertiseTags.length > 0 ? avatar.expertiseTags.join(", ") : "None specified"}\n`;
    } else {
      context += "- No avatar profile set yet. The user can create one.\n";
    }
    
    context += "\n## Recent Memories\n";
    if (recentMemories.length > 0) {
      for (const mem of recentMemories) {
        context += `- [${mem.type}] ${mem.content} (${new Date(mem.createdAt).toLocaleDateString()})\n`;
      }
    } else {
      context += "- No memories stored yet.\n";
    }
    
    return context;
  }

  /**
   * System prompt for the Percify Avatar Co-Pilot
   */
  private getSystemPrompt(): string {
    const tone = this.state.avatar?.tone || "casual";
    const toneInstructions: Record<string, string> = {
      casual: "Be friendly, relaxed, and approachable. Use conversational language.",
      professional: "Be formal, precise, and business-like. Maintain a professional demeanor.",
      playful: "Be fun, energetic, and use humor when appropriate. Keep things light.",
      technical: "Be detailed, accurate, and use technical terminology. Focus on precision."
    };

    return `You are Percify Avatar Co-Pilot, a friendly AI assistant that remembers each user's persona and preferences.

## IMPORTANT: Be Helpful and Conversational
- ALWAYS respond helpfully to ANY message, even simple greetings like "hey" or "hello"
- For greetings, introduce yourself warmly and explain what you can do
- NEVER ask for "more details" unless absolutely necessary - be proactive and helpful
- If unsure what the user wants, offer suggestions instead of asking for clarification

## Your Capabilities
1. **Avatar Setup**: Help users create their AI persona (name, bio, tone, expertise)
2. **Memory**: Remember preferences, tasks, and notes for the user
3. **Research**: Look up information from docs.percify.io (official Percify documentation)
4. **Chat**: Have friendly conversations

## How to Respond
- For "hey", "hello", "hi": Greet warmly and introduce your capabilities
- For "create avatar", "set up avatar", etc.: Use the saveAvatarProfile tool to help them create one
- For "remember X", "note that X": Use the saveMemory tool
- For "research X", "look up X", "what is percify", "how does X work": Use the researchWeb tool to search docs.percify.io
- For questions about Percify features: ALWAYS use researchWeb to fetch from docs.percify.io
- Maintain the avatar's tone (${tone}): ${toneInstructions[tone]}

## Example Responses
- User: "hey" → "Hey there! 👋 I'm Percify, your AI co-pilot! I can help you create a personalized avatar, remember your preferences, and look up our docs. Want to set up your avatar? Just tell me your name and what kind of tone you prefer (casual, professional, playful, or technical)!"
- User: "create an avatar" → Use saveAvatarProfile tool with a friendly default, then confirm
- User: "I prefer TypeScript" → Use saveMemory tool with type "preference"
- User: "what is percify" → Use researchWeb tool to fetch from docs.percify.io
- User: "how do memories work" → Use researchWeb tool with query "memory"

## Tools Available
- saveAvatarProfile: Set name, bio, tone (casual/professional/playful/technical), expertiseTags
- saveMemory: Store preferences, tasks, or notes  
- researchWeb: Search docs.percify.io for documentation on Percify features

${getSchedulePrompt({ date: new Date() })}

${this.buildContextString()}`;
  }

  /**
   * Handles incoming chat messages and manages the response stream
   * This is the main "brain" method that orchestrates LLM calls and tool execution
   */
  async onChatMessage(
    onFinish: StreamTextOnFinishCallback<ToolSet>,
    _options?: { abortSignal?: AbortSignal }
  ) {
    // Initialize Workers AI with Llama 3.3 70B
    const workersAI = createWorkersAI({ binding: this.env.AI });
    // @ts-expect-error - Model name is valid but not in type definitions yet
    const model = workersAI("@cf/meta/llama-3.3-70b-instruct-fp8-fast");

    // Use our defined tools (MCP not used in this project)
    const allTools = tools;

    const stream = createUIMessageStream({
      execute: async ({ writer }) => {
        // Clean up incomplete tool calls
        const cleanedMessages = cleanupMessages(this.messages);

        // Process any pending tool calls from previous messages
        const processedMessages = await processToolCalls({
          messages: cleanedMessages,
          dataStream: writer,
          tools: allTools,
          executions
        });

        const result = streamText({
          system: this.getSystemPrompt(),
          messages: convertToModelMessages(processedMessages),
          model,
          tools: allTools,
          maxTokens: 1024,
          temperature: 0.7,
          onFinish: onFinish as unknown as StreamTextOnFinishCallback<typeof allTools>,
          stopWhen: stepCountIs(10)
        });

        writer.merge(result.toUIMessageStream());
      }
    });

    return createUIMessageStreamResponse({ stream });
  }

  /**
   * Execute a scheduled task
   */
  async executeTask(description: string, _task: Schedule<string>) {
    await this.saveMessages([
      ...this.messages,
      {
        id: generateId(),
        role: "user",
        parts: [
          {
            type: "text",
            text: `Running scheduled task: ${description}`
          }
        ],
        metadata: {
          createdAt: new Date()
        }
      }
    ]);
  }
}

// Keep the old Chat export for backward compatibility during migration
export { PercifyAvatarAgent as Chat };

// ============================================================================
// WORKER ENTRY POINT
// ============================================================================

/**
 * Worker entry point that routes incoming requests to the appropriate handler
 */
export default {
  async fetch(request: Request, env: Env, _ctx: ExecutionContext) {
    const url = new URL(request.url);

    // Health check endpoint for Workers AI
    if (url.pathname === "/check-open-ai-key") {
      // We're using Workers AI, so always return success
      return Response.json({
        success: true,
        provider: "workers-ai"
      });
    }

    // API endpoint to get current avatar state
    if (url.pathname === "/api/avatar-state") {
      try {
        // Get the agent instance and return state
        // Note: This requires the agent to be accessed via the Durable Object
        return Response.json({
          message: "Use WebSocket connection to access avatar state"
        });
      } catch (error) {
        return Response.json({ error: "Failed to get avatar state" }, { status: 500 });
      }
    }

    return (
      // Route the request to our agent or return 404 if not found
      (await routeAgentRequest(request, env)) ||
      new Response("Not found", { status: 404 })
    );
  }
} satisfies ExportedHandler<Env>;
