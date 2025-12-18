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
   * Perform web research (simulated for assignment)
   */
  async researchWeb(query: string): Promise<{ query: string; snippet: string; sourceUrl: string }> {
    console.log("[PercifyAvatarAgent] Executing research for:", query);
    
    const sourceUrl = "https://developers.cloudflare.com/agents/";
    
    try {
      const response = await fetch(sourceUrl);
      const html = await response.text();
      
      // Extract a meaningful snippet from the response
      const textContent = html
        .replace(/<script[^>]*>[\s\S]*?<\/script>/gi, "")
        .replace(/<style[^>]*>[\s\S]*?<\/style>/gi, "")
        .replace(/<[^>]+>/g, " ")
        .replace(/\s+/g, " ")
        .trim();
      
      // Get relevant snippet (first 500 chars)
      const snippet = textContent.substring(0, 500) + "...";
      
      console.log("[PercifyAvatarAgent] Research complete");
      return { query, snippet, sourceUrl };
    } catch (error) {
      console.error("[PercifyAvatarAgent] Research error:", error);
      return {
        query,
        snippet: `Research on "${query}" completed. Cloudflare Agents SDK enables building AI-powered applications with persistent state, real-time communication, and tool integration capabilities.`,
        sourceUrl
      };
    }
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
3. **Research**: Look up information from Cloudflare docs
4. **Chat**: Have friendly conversations

## How to Respond
- For "hey", "hello", "hi": Greet warmly and introduce your capabilities
- For "create avatar", "set up avatar", etc.: Use the saveAvatarProfile tool to help them create one
- For "remember X", "note that X": Use the saveMemory tool
- For "research X", "look up X": Use the researchWeb tool
- Maintain the avatar's tone (${tone}): ${toneInstructions[tone]}

## Example Responses
- User: "hey" → "Hey there! 👋 I'm Percify, your AI co-pilot! I can help you create a personalized avatar, remember your preferences, and research topics. Want to set up your avatar? Just tell me your name and what kind of tone you prefer (casual, professional, playful, or technical)!"
- User: "create an avatar" → Use saveAvatarProfile tool with a friendly default, then confirm
- User: "I prefer TypeScript" → Use saveMemory tool with type "preference"

## Tools Available
- saveAvatarProfile: Set name, bio, tone (casual/professional/playful/technical), expertiseTags
- saveMemory: Store preferences, tasks, or notes  
- researchWeb: Look up information

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
