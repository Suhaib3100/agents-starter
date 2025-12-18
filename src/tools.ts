/**
 * Tool definitions for the Percify Avatar Co-Pilot
 * Includes avatar management, memory storage, and web research tools
 */
import { tool, type ToolSet } from "ai";
import { z } from "zod/v3";

import type { PercifyAvatarAgent } from "./server";
import { getCurrentAgent } from "agents";
import { scheduleSchema } from "agents/schedule";

// ============================================================================
// AVATAR PROFILE TOOL
// ============================================================================

/**
 * Save or update the user's avatar profile
 * Allows setting display name, bio, tone, and expertise tags
 */
const saveAvatarProfile = tool({
  description: `Update the user's avatar profile. Use this when the user wants to:
    - Set or change their display name
    - Update their bio/description
    - Change their communication tone (casual, professional, playful, technical)
    - Add or modify expertise tags
    Example triggers: "Set my avatar as...", "Call me...", "I'm a...", "Change my tone to..."`,
  inputSchema: z.object({
    displayName: z.string().optional().describe("The avatar's display name"),
    bio: z.string().optional().describe("A short bio or description for the avatar"),
    tone: z.enum(["casual", "professional", "playful", "technical"]).optional()
      .describe("The communication tone: casual, professional, playful, or technical"),
    expertiseTags: z.array(z.string()).optional()
      .describe("List of expertise areas or skills")
  }),
  execute: async ({ displayName, bio, tone, expertiseTags }) => {
    const { agent } = getCurrentAgent<PercifyAvatarAgent>();
    
    if (!agent) {
      return { error: "Agent not available" };
    }

    try {
      const updatedAvatar = agent.saveAvatarProfile({
        displayName,
        bio,
        tone,
        expertiseTags
      });

      return {
        success: true,
        message: `Avatar profile updated successfully!`,
        avatar: updatedAvatar
      };
    } catch (error) {
      console.error("Error saving avatar profile:", error);
      return { error: `Failed to update avatar profile: ${error}` };
    }
  }
});

// ============================================================================
// MEMORY STORAGE TOOL
// ============================================================================

/**
 * Store a memory item for long-term retention
 * Supports tasks, preferences, and general notes
 */
const saveMemory = tool({
  description: `Store information in the user's memory for future reference. Use this when:
    - User asks to remember something: "Remember that I...", "Don't forget..."
    - User shares a preference: "I prefer...", "I like..."
    - User mentions an important task or project
    Memory types: 'preference' for likes/dislikes, 'task' for work items, 'note' for general info`,
  inputSchema: z.object({
    type: z.enum(["task", "preference", "note"])
      .describe("The type of memory: task, preference, or note"),
    content: z.string()
      .describe("The content to remember")
  }),
  execute: async ({ type, content }) => {
    const { agent } = getCurrentAgent<PercifyAvatarAgent>();
    
    if (!agent) {
      return { error: "Agent not available" };
    }

    try {
      const recentMemories = agent.saveMemory(type, content);

      return {
        success: true,
        message: `Memory stored successfully! I'll remember: "${content}"`,
        recentMemories
      };
    } catch (error) {
      console.error("Error saving memory:", error);
      return { error: `Failed to store memory: ${error}` };
    }
  }
});

// ============================================================================
// WEB RESEARCH TOOL
// ============================================================================

/**
 * Perform web research by fetching content from docs.percify.io
 * The official Percify documentation site
 */
const researchWeb = tool({
  description: `Search the official Percify documentation at docs.percify.io. Use this when the user asks:
    - "What is Percify?", "How does X work?"
    - "Research...", "Look up...", "Find information about..."
    - Questions about avatars, memories, tones, tools, scheduling
    - Any question about Percify features or capabilities
    This searches docs.percify.io and returns official documentation.`,
  inputSchema: z.object({
    query: z.string().describe("The search query for docs.percify.io (e.g., 'avatar', 'memory', 'tone', 'tools')")
  }),
  execute: async ({ query }) => {
    const { agent } = getCurrentAgent<PercifyAvatarAgent>();
    
    if (!agent) {
      return { error: "Agent not available" };
    }

    try {
      const result = await agent.researchWeb(query);

      return {
        success: true,
        message: `Research completed for: "${query}"`,
        ...result
      };
    } catch (error) {
      console.error("Error performing research:", error);
      return { error: `Failed to complete research: ${error}` };
    }
  }
});

// ============================================================================
// GET AVATAR STATE TOOL
// ============================================================================

/**
 * Retrieve the current avatar state including profile and memories
 * Useful for the LLM to check current state before making decisions
 */
const getAvatarState = tool({
  description: "Get the current avatar profile and recent memories",
  inputSchema: z.object({}),
  execute: async () => {
    const { agent } = getCurrentAgent<PercifyAvatarAgent>();
    
    if (!agent) {
      return { error: "Agent not available" };
    }

    try {
      const state = agent.getAvatarState();
      return {
        success: true,
        avatar: state.avatar || null,
        recentMemories: state.memories
      };
    } catch (error) {
      console.error("Error getting avatar state:", error);
      return { error: `Failed to get avatar state: ${error}` };
    }
  }
});

// ============================================================================
// SCHEDULING TOOLS (from original starter)
// ============================================================================

/**
 * Schedule a task to be executed at a later time
 */
const scheduleTask = tool({
  description: "A tool to schedule a task to be executed at a later time",
  inputSchema: scheduleSchema,
  execute: async ({ when, description }) => {
    const { agent } = getCurrentAgent<PercifyAvatarAgent>();

    function throwError(msg: string): string {
      throw new Error(msg);
    }
    if (when.type === "no-schedule") {
      return "Not a valid schedule input";
    }
    const input =
      when.type === "scheduled"
        ? when.date
        : when.type === "delayed"
          ? when.delayInSeconds
          : when.type === "cron"
            ? when.cron
            : throwError("not a valid schedule input");
    try {
      agent!.schedule(input!, "executeTask", description);
    } catch (error) {
      console.error("error scheduling task", error);
      return `Error scheduling task: ${error}`;
    }
    return `Task scheduled for type "${when.type}" : ${input}`;
  }
});

/**
 * List all scheduled tasks
 */
const getScheduledTasks = tool({
  description: "List all tasks that have been scheduled",
  inputSchema: z.object({}),
  execute: async () => {
    const { agent } = getCurrentAgent<PercifyAvatarAgent>();

    try {
      const tasks = agent!.getSchedules();
      if (!tasks || tasks.length === 0) {
        return "No scheduled tasks found.";
      }
      return tasks;
    } catch (error) {
      console.error("Error listing scheduled tasks", error);
      return `Error listing scheduled tasks: ${error}`;
    }
  }
});

/**
 * Cancel a scheduled task by its ID
 */
const cancelScheduledTask = tool({
  description: "Cancel a scheduled task using its ID",
  inputSchema: z.object({
    taskId: z.string().describe("The ID of the task to cancel")
  }),
  execute: async ({ taskId }) => {
    const { agent } = getCurrentAgent<PercifyAvatarAgent>();
    try {
      await agent!.cancelSchedule(taskId);
      return `Task ${taskId} has been successfully canceled.`;
    } catch (error) {
      console.error("Error canceling scheduled task", error);
      return `Error canceling task ${taskId}: ${error}`;
    }
  }
});

// ============================================================================
// WEATHER TOOL (Human-in-the-loop example from starter)
// ============================================================================

/**
 * Weather information tool that requires human confirmation
 */
const getWeatherInformation = tool({
  description: "Show the weather in a given city to the user",
  inputSchema: z.object({ city: z.string() })
  // No execute function = requires human confirmation
});

/**
 * Local time tool
 */
const getLocalTime = tool({
  description: "Get the local time for a specified location",
  inputSchema: z.object({ location: z.string() }),
  execute: async ({ location }) => {
    console.log(`Getting local time for ${location}`);
    return "10am";
  }
});

// ============================================================================
// EXPORTS
// ============================================================================

/**
 * All available tools for the Percify Avatar Co-Pilot
 */
export const tools = {
  // Avatar tools (main features)
  saveAvatarProfile,
  saveMemory,
  researchWeb,
  getAvatarState,
  // Scheduling tools
  scheduleTask,
  getScheduledTasks,
  cancelScheduledTask,
  // Utility tools
  getWeatherInformation,
  getLocalTime
} satisfies ToolSet;

/**
 * Execution handlers for tools requiring human confirmation
 */
export const executions = {
  getWeatherInformation: async ({ city }: { city: string }) => {
    console.log(`Getting weather information for ${city}`);
    return `The weather in ${city} is sunny`;
  }
};
