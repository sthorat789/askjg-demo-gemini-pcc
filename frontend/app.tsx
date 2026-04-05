/**
 * Gemini Voice Agent — LiveKit React Frontend
 *
 * A React component that connects to a LiveKit room and displays
 * real-time agent state (speaking, listening, barge-in) driven by
 * data channel messages from the Python backend.
 *
 * The backend (livekit_ui_integration.py) broadcasts JSON payloads
 * via LiveKit's data channel with the shape:
 *   { "status": "speaking" | "listening" | "barge_in_active" | "idle",
 *     "message": "Human-readable description" }
 *
 * This component renders:
 * - A central avatar/visualizer with state-driven border colors and glow
 * - Status text showing the current agent state
 * - A local microphone activity indicator
 *
 * Dependencies:
 *   npm install @livekit/components-react livekit-client react
 *
 * Usage:
 *   <GeminiVoiceApp wsUrl="wss://your-livekit.cloud" token="<jwt>" />
 */

import React, { useState, useEffect } from 'react';
import {
  LiveKitRoom,
  RoomAudioRenderer,
  useDataChannel,
  useTracks,
  useParticipant,
} from '@livekit/components-react';
import { Track } from 'livekit-client';

// ---------------------------------------------------------------------------
// 1. MAIN WRAPPER COMPONENT
// ---------------------------------------------------------------------------

interface GeminiVoiceAppProps {
  /** LiveKit server WebSocket URL */
  wsUrl?: string;
  /** LiveKit participant access token (JWT) */
  token?: string;
}

export default function GeminiVoiceApp({
  wsUrl = process.env.NEXT_PUBLIC_LIVEKIT_URL || "",
  token = "",
}: GeminiVoiceAppProps) {

  if (!wsUrl || !token) {
    return (
      <div className="flex h-screen items-center justify-center bg-gray-900 text-white font-mono">
        Please provide a valid LiveKit WebSocket URL and Participant Token.
      </div>
    );
  }

  return (
    <LiveKitRoom
      serverUrl={wsUrl}
      token={token}
      connect={true}
      audio={true}
    >
      {/* Invisible component that auto-plays the Agent's incoming WebRTC audio */}
      <RoomAudioRenderer />

      {/* Our Custom UI */}
      <AgentInterface />
    </LiveKitRoom>
  );
}

// ---------------------------------------------------------------------------
// 2. THE VISUAL INTERFACE & DATA CHANNEL LISTENER
// ---------------------------------------------------------------------------

type AgentStatus = 'idle' | 'listening' | 'speaking' | 'barge_in_active';

function AgentInterface() {
  const [uiState, setUiState] = useState<AgentStatus>('idle');
  const [message, setMessage] = useState('Waiting for connection...');

  // Hook 1: Listen to Data Channels beamed from the Python Backend
  const { message: dataChannelMessage } = useDataChannel();

  // Hook 2: Monitor if the local user is speaking (Frontend Mic VAD)
  const tracks = useTracks([Track.Source.Microphone]);
  const userTrack = tracks.find((t) => t.participant.isLocal);
  const { isSpeaking: isUserSpeaking } = useParticipant(
    userTrack?.participant ?? ({} as any)
  );

  // Process incoming state updates from the Python Backend Data Channel
  useEffect(() => {
    if (dataChannelMessage) {
      const decoder = new TextDecoder();
      const rawString = decoder.decode(dataChannelMessage.payload);
      try {
        const parsed = JSON.parse(rawString);
        if (parsed.status) {
          setUiState(parsed.status as AgentStatus);
        }
        if (parsed.message) {
          setMessage(parsed.message);
        }
      } catch (e) {
        console.error("Failed to parse backend UI state:", e);
      }
    }
  }, [dataChannelMessage]);

  // Dynamic visual logic based on the synchronized states
  const getBorderColor = (): string => {
    if (uiState === 'barge_in_active') return 'border-red-500';
    if (uiState === 'speaking') return 'border-cyan-400';
    if (isUserSpeaking) return 'border-green-400';
    return 'border-gray-600';
  };

  const getEmoji = (): string => {
    if (uiState === 'barge_in_active') return '🛑';
    if (uiState === 'speaking') return '🤖🔊';
    if (isUserSpeaking) return '🗣️';
    return '🤖';
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-gray-900 text-white p-8 font-sans">

      {/* The Central Visualizer Avatar */}
      <div
        className={`mb-8 p-12 border-4 rounded-full transition-all duration-300 shadow-2xl flex items-center justify-center w-64 h-64 ${getBorderColor()}`}
        style={{
          transform: uiState === 'speaking' ? 'scale(1.1)' : 'scale(1)',
          boxShadow: uiState === 'speaking'
            ? '0 0 40px rgba(34, 211, 238, 0.4)'
            : uiState === 'barge_in_active'
              ? '0 0 40px rgba(239, 68, 68, 0.4)'
              : 'none',
        }}
      >
        <h1 className="text-8xl text-center select-none">
          {getEmoji()}
        </h1>
      </div>

      {/* State Text & Subtitle */}
      <div className="text-center space-y-4 max-w-md">
        <h2
          className={`text-2xl font-bold tracking-widest uppercase ${
            uiState === 'barge_in_active' ? 'text-red-400' : 'text-white'
          }`}
        >
          {uiState.replace(/_/g, ' ')}
        </h2>

        <p className="text-gray-400 text-lg h-8">
          {message}
        </p>

        {/* Local Microphone indicator */}
        <div className="h-6">
          {isUserSpeaking && (
            <p className="text-green-400 animate-pulse text-sm font-mono mt-4">
              [Mic Active: Transmitting]
            </p>
          )}
        </div>
      </div>

    </div>
  );
}
